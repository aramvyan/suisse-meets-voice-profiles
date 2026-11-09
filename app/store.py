"""Per-customer (tenant) vault for voice profiles.

Isolation rules this module enforces:
  * every tenant has its own directory AND its own encryption key, so one
    customer's data is unreadable with another customer's key;
  * there is no API that loads profiles across tenants — matching always
    receives exactly one tenant's profiles;
  * only the voice fingerprint (embedding) is stored, never the raw audio;
  * the consent log holds no biometric data, so it can outlive a deletion
    as proof that consent was given and later withdrawn.

In production the key would live in a KMS/HSM (e.g. per-tenant keys in a
Swiss-region KMS), not next to the data. A file is used here to keep the
prototype self-contained.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet

from .engine import ROOT

TENANTS_DIR = ROOT / "data" / "tenants"
TENANT_ID = re.compile(r"^[a-z0-9-]{3,40}$")
PROFILE_TTL_DAYS = 365  # profile expires unless the person re-confirms

_invite_lock = threading.Lock()

CONSENT_VERSION = "v1-2026-11"
CONSENT_TEXT = (
    "I agree that Suisse Meets creates a voice profile from my voice sample "
    "for the sole purpose of showing my name in this organisation's meeting "
    "transcripts. Only a mathematical voice fingerprint is stored, not the "
    "recording. It is stored in Switzerland, is used only within my "
    "organisation, and is never used to train models. Giving this consent is "
    "voluntary and has no effect on my employment. I can delete my voice "
    "profile at any time, and it is deleted automatically after 12 months "
    "unless I renew it."
)
CONSENT_HASH = hashlib.sha256(CONSENT_TEXT.encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class TenantVault:
    def __init__(self, tenant_id: str):
        if not TENANT_ID.match(tenant_id):
            raise ValueError("invalid tenant id")
        self.tenant_id = tenant_id
        self.dir = TENANTS_DIR / tenant_id
        self.dir.mkdir(parents=True, exist_ok=True)
        key_file = self.dir / "tenant.key"
        if not key_file.exists():
            key_file.write_bytes(Fernet.generate_key())
            key_file.chmod(0o600)
        self._fernet = Fernet(key_file.read_bytes())

    # --- encrypted profile storage -------------------------------------
    def _load(self) -> dict:
        f = self.dir / "profiles.enc"
        if not f.exists():
            return {}
        return json.loads(self._fernet.decrypt(f.read_bytes()))

    def _save(self, profiles: dict) -> None:
        tmp = self.dir / "profiles.enc.tmp"
        tmp.write_bytes(self._fernet.encrypt(json.dumps(profiles).encode()))
        tmp.replace(self.dir / "profiles.enc")

    def _purge_expired(self, profiles: dict) -> dict:
        now = datetime.now(timezone.utc)
        expired = [pid for pid, p in profiles.items()
                   if datetime.fromisoformat(p["expires_at"]) < now]
        for pid in expired:
            self.log("profile_expired_deleted", profile_id=pid)
            del profiles[pid]
        if expired:
            self._save(profiles)
        return profiles

    def list_profiles(self) -> list[dict]:
        profiles = self._purge_expired(self._load())
        return [{k: v for k, v in p.items() if k != "embedding"} | {"id": pid}
                for pid, p in profiles.items()]

    def embeddings(self) -> tuple[list[str], list[str], np.ndarray]:
        """(ids, names, matrix) for matching — this tenant only."""
        profiles = self._purge_expired(self._load())
        ids = list(profiles)
        names = [profiles[i]["name"] for i in ids]
        mat = (np.array([profiles[i]["embedding"] for i in ids], dtype=np.float32)
               if ids else np.zeros((0, 0), dtype=np.float32))
        return ids, names, mat

    def add_profile(self, name: str, email: str, embedding: np.ndarray,
                    seconds: float, consent_id: str) -> str:
        profiles = self._load()
        # Re-enrolling replaces the old profile of the same person.
        for pid in [p for p, v in profiles.items() if v["email"] == email]:
            del profiles[pid]
        pid = uuid.uuid4().hex[:12]
        profiles[pid] = {
            "name": name, "email": email,
            "embedding": embedding.round(6).tolist(),
            "sample_seconds": round(seconds, 1),
            "consent_id": consent_id,
            "enrolled_at": _now(),
            "expires_at": (datetime.now(timezone.utc)
                           + timedelta(days=PROFILE_TTL_DAYS)).isoformat(timespec="seconds"),
        }
        self._save(profiles)
        self.log("profile_created", profile_id=pid, email=email)
        return pid

    def delete_profile(self, profile_id: str, by: str) -> bool:
        profiles = self._load()
        if profile_id not in profiles:
            return False
        email = profiles.pop(profile_id)["email"]
        self._save(profiles)
        self.log("profile_deleted", profile_id=profile_id, email=email, by=by)
        return True

    # --- invites: only the person themself can give consent ------------
    def create_invite(self, name: str, email: str) -> str:
        token = secrets.token_urlsafe(24)
        invites = self._invites()
        invites[_hash_token(token)] = {"name": name, "email": email,
                                       "created": time.time()}
        (self.dir / "invites.json").write_text(json.dumps(invites))
        self.log("invite_created", email=email)
        return token

    def _invites(self) -> dict:
        f = self.dir / "invites.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def peek_invite(self, token: str) -> dict | None:
        inv = self._invites().get(_hash_token(token))
        if inv and time.time() - inv["created"] > 7 * 86400:
            return None  # invite links expire after 7 days
        return inv

    def take_invite(self, token: str) -> dict | None:
        """Atomically claim an invite so it can't be used twice in parallel."""
        with _invite_lock:
            inv = self.peek_invite(token)
            if inv:
                invites = self._invites()
                invites.pop(_hash_token(token), None)
                (self.dir / "invites.json").write_text(json.dumps(invites))
            return inv

    def restore_invite(self, token: str, inv: dict) -> None:
        """Give the invite back when enrollment failed (e.g. audio too short)."""
        with _invite_lock:
            invites = self._invites()
            invites[_hash_token(token)] = inv
            (self.dir / "invites.json").write_text(json.dumps(invites))

    # --- self-service management link (stored hashed, like invites) ----
    def create_manage_token(self, profile_id: str) -> str:
        token = secrets.token_urlsafe(24)
        f = self.dir / "manage.json"
        tokens = json.loads(f.read_text()) if f.exists() else {}
        tokens[_hash_token(token)] = profile_id
        f.write_text(json.dumps(tokens))
        return token

    def profile_for_manage_token(self, token: str) -> str | None:
        f = self.dir / "manage.json"
        tokens = json.loads(f.read_text()) if f.exists() else {}
        return tokens.get(_hash_token(token))

    # --- append-only audit / consent log (no biometric data) -----------
    def log(self, event: str, **fields) -> str:
        entry_id = uuid.uuid4().hex[:12]
        with (self.dir / "audit.jsonl").open("a") as f:
            f.write(json.dumps({"id": entry_id, "at": _now(), "event": event,
                                **fields}) + "\n")
        return entry_id

    def audit(self) -> list[dict]:
        f = self.dir / "audit.jsonl"
        if not f.exists():
            return []
        return [json.loads(line) for line in f.read_text().splitlines()][::-1]
