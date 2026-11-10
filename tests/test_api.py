"""API-level checks of the consent flow (loads the real models, ~10 s)."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main, store

WAV = Path(__file__).resolve().parent.parent / "models" / "sherpa-onnx-whisper-tiny.en" / "test_wavs" / "0.wav"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "TENANTS_DIR", tmp_path)
    return TestClient(main.app)


def invite(client, tenant="alpine-dental"):
    r = client.post(f"/api/t/{tenant}/invites", json={"name": "Anna", "email": "anna@a.ch"})
    assert r.status_code == 200
    return r.json()["link"].split("token=")[1]


def enroll(client, token, consent=True, version=store.CONSENT_VERSION, tenant="alpine-dental"):
    with WAV.open("rb") as f:
        return client.post(f"/api/invite/{tenant}/{token}/enroll",
                           data={"consent": str(consent).lower(), "consent_version": version},
                           files={"audio": ("a.wav", f, "audio/wav")})


def test_unknown_tenant_is_rejected(client):
    assert client.get("/api/t/evil-corp/profiles").status_code == 404


def test_enrollment_without_consent_is_refused(client):
    token = invite(client)
    r = enroll(client, token, consent=False)
    assert r.status_code == 400 and "consent" in r.json()["detail"]
    assert client.get("/api/t/alpine-dental/profiles").json() == []


def test_consent_to_an_outdated_text_is_refused(client):
    token = invite(client)
    assert enroll(client, token, version="v0-old").status_code == 400


def test_invite_from_one_tenant_is_useless_in_another(client):
    token = invite(client, "alpine-dental")
    assert enroll(client, token, tenant="zurich-legal").status_code == 404


def test_too_short_sample_is_refused_and_nothing_stored(client):
    token = invite(client)
    r = enroll(client, token)  # the test clip is ~6 s, below the 8 s minimum
    assert r.status_code == 400 and "at least" in r.json()["detail"]
    assert client.get("/api/t/alpine-dental/profiles").json() == []
    events = [e["event"] for e in client.get("/api/t/alpine-dental/audit").json()]
    # Consent is logged before processing; the failure is logged too.
    assert "enrollment_failed" in events and "profile_created" not in events


def test_failed_enrollment_gives_the_invite_back(client):
    token = invite(client)
    assert enroll(client, token).status_code == 400  # too short
    # The same link still works for a second attempt.
    assert client.get(f"/api/invite/alpine-dental/{token}").status_code == 200


def test_self_delete_needs_the_manage_token_not_the_profile_id(client):
    v = store.TenantVault("alpine-dental")
    pid = v.add_profile("Anna", "anna@a.ch", np.ones(192, dtype=np.float32) / 192 ** 0.5, 20, "c")
    assert client.delete(f"/api/me/alpine-dental/{pid}").status_code == 404
    token = v.create_manage_token(pid)
    assert client.delete(f"/api/me/alpine-dental/{token}").status_code == 200
    assert client.get("/api/t/alpine-dental/profiles").json() == []


def test_playlist_upload_is_rejected(client):
    token = invite(client)
    playlist = b"#EXTM3U\n#EXTINF:10,\nhttp://127.0.0.1:9/secret.wav\n#EXT-X-ENDLIST\n"
    r = client.post(f"/api/invite/alpine-dental/{token}/enroll",
                    data={"consent": "true", "consent_version": store.CONSENT_VERSION},
                    files={"audio": ("a.m3u8", playlist, "audio/x-mpegurl")})
    assert r.status_code == 400 and "decode" in r.json()["detail"]
