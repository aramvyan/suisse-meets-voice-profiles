"""Fast tests (no ML models) for the rules that must never break:
consent, deletion, encryption at rest and tenant isolation."""

import json

import numpy as np
import pytest

from app import recognizer, store


@pytest.fixture(autouse=True)
def tmp_tenants(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "TENANTS_DIR", tmp_path)
    return tmp_path


def unit(seed):
    v = np.random.default_rng(seed).normal(size=256).astype(np.float32)
    return v / np.linalg.norm(v)


def test_profile_file_is_encrypted_and_has_no_plaintext(tmp_tenants):
    v = store.TenantVault("alpine-dental")
    v.add_profile("Anna Keller", "anna@alpine.ch", unit(1), 20, "c1")
    raw = (tmp_tenants / "alpine-dental" / "profiles.enc").read_bytes()
    assert b"Anna" not in raw and b"anna@alpine.ch" not in raw


def test_other_tenant_key_cannot_decrypt(tmp_tenants):
    store.TenantVault("alpine-dental").add_profile("Anna", "a@a.ch", unit(1), 20, "c1")
    store.TenantVault("zurich-legal")
    # Copy A's encrypted file into B's folder: B's key must not open it.
    (tmp_tenants / "zurich-legal" / "profiles.enc").write_bytes(
        (tmp_tenants / "alpine-dental" / "profiles.enc").read_bytes())
    with pytest.raises(Exception):
        store.TenantVault("zurich-legal").list_profiles()


def test_tenants_never_see_each_others_profiles():
    a, b = store.TenantVault("alpine-dental"), store.TenantVault("zurich-legal")
    a.add_profile("Anna", "anna@a.ch", unit(1), 20, "c1")
    assert b.list_profiles() == []
    ids, names, mat = b.embeddings()
    assert names == [] and mat.shape[0] == 0


def test_same_voice_in_other_tenant_gets_no_name():
    a, b = store.TenantVault("alpine-dental"), store.TenantVault("zurich-legal")
    anna = unit(1)
    a.add_profile("Anna", "anna@a.ch", anna, 20, "c1")
    b.add_profile("Bob", "bob@b.ch", unit(2), 20, "c2")
    ids, names, mat = b.embeddings()
    m = recognizer.identify({0: anna}, {0: 30.0}, ids, names, mat, 0.5)
    assert m[0].name is None


@pytest.mark.parametrize("bad", ["../etc", "A", "x" * 60, "a/b", ""])
def test_tenant_id_cannot_escape_its_folder(bad):
    with pytest.raises(ValueError):
        store.TenantVault(bad)


def test_delete_removes_biometrics_but_keeps_consent_trail(tmp_tenants):
    v = store.TenantVault("alpine-dental")
    cid = v.log("consent_given", email="anna@a.ch", version=store.CONSENT_VERSION)
    pid = v.add_profile("Anna", "anna@a.ch", unit(1), 20, cid)
    assert v.delete_profile(pid, by="self")
    assert v.list_profiles() == [] and v.embeddings()[2].shape[0] == 0
    events = [e["event"] for e in v.audit()]
    assert {"consent_given", "profile_created", "profile_deleted"} <= set(events)
    log = (tmp_tenants / "alpine-dental" / "audit.jsonl").read_text()
    assert "embedding" not in log


def test_expired_profiles_are_purged(tmp_tenants):
    v = store.TenantVault("alpine-dental")
    pid = v.add_profile("Anna", "anna@a.ch", unit(1), 20, "c1")
    profiles = v._load()
    profiles[pid]["expires_at"] = "2000-01-01T00:00:00+00:00"
    v._save(profiles)
    assert v.list_profiles() == []
    assert any(e["event"] == "profile_expired_deleted" for e in v.audit())


def test_reenrolling_replaces_old_profile():
    v = store.TenantVault("alpine-dental")
    v.add_profile("Anna", "anna@a.ch", unit(1), 20, "c1")
    v.add_profile("Anna", "anna@a.ch", unit(2), 20, "c2")
    assert len(v.list_profiles()) == 1


def test_invite_token_is_stored_hashed_and_single_use(tmp_tenants):
    v = store.TenantVault("alpine-dental")
    token = v.create_invite("Anna", "anna@a.ch")
    assert token not in (tmp_tenants / "alpine-dental" / "invites.json").read_text()
    assert v.peek_invite(token)["email"] == "anna@a.ch"
    v.consume_invite(token)
    assert v.peek_invite(token) is None


def mix(a, b, w):
    v = a * w + b * (1 - w)
    return v / np.linalg.norm(v)


def test_matching_prefers_best_cluster_and_respects_threshold():
    anna, bob = unit(1), unit(2)
    ids, names, mat = ["p1", "p2"], ["Anna", "Bob"], np.stack([anna, bob])
    best = mix(anna, unit(3), 0.95)
    medium = mix(anna, unit(4), 0.45)  # above threshold, below split margin
    assert 0.5 <= float(medium @ anna) < 0.5 + recognizer.SPLIT_MARGIN
    m = recognizer.identify({0: best, 1: medium}, {0: 30, 1: 30}, ids, names, mat, 0.5)
    assert m[0].name == "Anna" and m[1].name is None
    # A stranger stays unnamed.
    m = recognizer.identify({0: unit(9)}, {0: 30}, ids, names, mat, 0.5)
    assert m[0].name is None


def test_split_cluster_of_same_person_is_merged_only_with_high_score():
    anna = unit(1)
    ids, names, mat = ["p1"], ["Anna"], anna[None]
    part1, part2 = mix(anna, unit(3), 0.95), mix(anna, unit(4), 0.9)
    assert float(part2 @ anna) >= 0.5 + recognizer.SPLIT_MARGIN
    m = recognizer.identify({0: part1, 1: part2}, {0: 30, 1: 10}, ids, names, mat, 0.5)
    assert m[0].name == "Anna" and m[1].name == "Anna"


def test_too_little_speech_is_never_named():
    anna = unit(1)
    m = recognizer.identify({0: anna}, {0: 1.0}, ["p1"], ["Anna"], anna[None], 0.5)
    assert m[0].name is None


def test_calibrated_threshold_read_from_eval(tmp_path, monkeypatch):
    (tmp_path / "data" / "eval").mkdir(parents=True)
    (tmp_path / "data" / "eval" / "metrics.json").write_text(json.dumps({"chosen_threshold": 0.42}))
    monkeypatch.setattr(recognizer, "ROOT", tmp_path)
    assert recognizer.calibrated_threshold() == 0.42
