from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import nexuml.storage.s3 as s3_module
from nexuml.storage.s3 import S3Client, configure_dali_s3_from_env


def _fake_boto3(captured: dict[str, Any]) -> SimpleNamespace:
    client = object()

    class Session:
        def client(self, service_name: str, **kwargs: Any) -> object:
            captured["service_name"] = service_name
            captured["kwargs"] = kwargs
            return client

    return SimpleNamespace(Session=Session, session=SimpleNamespace(Session=Session))


def test_s3_client_omits_verify_by_default(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}
    monkeypatch.delenv("NEXUML_S3_VERIFY_SSL", raising=False)
    monkeypatch.setattr(s3_module.importlib, "import_module", lambda _name: _fake_boto3(captured))

    S3Client()._get_client()

    assert captured == {"service_name": "s3", "kwargs": {}}


@pytest.mark.parametrize("verify", [False, True, "/etc/ssl/certs/custom-ca.pem"])
def test_s3_client_forwards_explicit_verify(monkeypatch: pytest.MonkeyPatch, verify: bool | str):
    captured: dict[str, Any] = {}
    monkeypatch.setenv("NEXUML_S3_VERIFY_SSL", "invalid")
    monkeypatch.setattr(s3_module.importlib, "import_module", lambda _name: _fake_boto3(captured))

    S3Client(verify=verify)._get_client()

    assert captured["kwargs"] == {"verify": verify}


def test_s3_client_explicit_verify_overrides_environment(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}
    monkeypatch.setenv("NEXUML_S3_VERIFY_SSL", "0")
    monkeypatch.setattr(s3_module.importlib, "import_module", lambda _name: _fake_boto3(captured))

    S3Client(verify=True)._get_client()

    assert captured["kwargs"] == {"verify": True}


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", False), ("1", True)],
)
def test_s3_client_reads_verify_environment(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
):
    captured: dict[str, Any] = {}
    monkeypatch.setenv("NEXUML_S3_VERIFY_SSL", value)
    monkeypatch.setattr(s3_module.importlib, "import_module", lambda _name: _fake_boto3(captured))

    S3Client()._get_client()

    assert captured["kwargs"] == {"verify": expected}


def test_s3_client_rejects_invalid_verify_environment_before_boto(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEXUML_S3_VERIFY_SSL", "yes")
    monkeypatch.setattr(
        s3_module.importlib,
        "import_module",
        lambda _name: pytest.fail("boto3 must not be loaded for an invalid setting"),
    )

    with pytest.raises(ValueError, match="NEXUML_S3_VERIFY_SSL.*'0' or '1'"):
        S3Client()._get_client()


def test_s3_client_injected_client_keeps_existing_lazy_semantics():
    injected = object()

    assert S3Client(client=injected)._get_client() is injected


@pytest.mark.parametrize(("shared", "expected"), [("0", "1"), ("1", "0")])
def test_dali_s3_flag_is_derived_from_shared_setting(
    monkeypatch: pytest.MonkeyPatch, shared: str, expected: str
):
    monkeypatch.setenv("NEXUML_S3_VERIFY_SSL", shared)
    monkeypatch.setenv("DALI_S3_NO_VERIFY_SSL", "0" if expected == "1" else "1")
    before = {
        name: value
        for name, value in s3_module.os.environ.items()
        if name != "DALI_S3_NO_VERIFY_SSL"
    }

    configure_dali_s3_from_env()

    after = {
        name: value
        for name, value in s3_module.os.environ.items()
        if name != "DALI_S3_NO_VERIFY_SSL"
    }
    assert after == before
    assert s3_module.os.environ["DALI_S3_NO_VERIFY_SSL"] == expected


@pytest.mark.parametrize("existing", ["0", "1"])
def test_dali_s3_flag_is_untouched_without_shared_setting(
    monkeypatch: pytest.MonkeyPatch, existing: str
):
    monkeypatch.delenv("NEXUML_S3_VERIFY_SSL", raising=False)
    monkeypatch.setenv("DALI_S3_NO_VERIFY_SSL", existing)

    configure_dali_s3_from_env()

    assert s3_module.os.environ["DALI_S3_NO_VERIFY_SSL"] == existing


def test_dali_s3_flag_rejects_invalid_shared_setting(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEXUML_S3_VERIFY_SSL", "invalid")
    monkeypatch.setenv("DALI_S3_NO_VERIFY_SSL", "0")

    with pytest.raises(ValueError, match="NEXUML_S3_VERIFY_SSL.*'0' or '1'"):
        configure_dali_s3_from_env()

    assert s3_module.os.environ["DALI_S3_NO_VERIFY_SSL"] == "0"
