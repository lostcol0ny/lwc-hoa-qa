import logging
from pathlib import Path

import pytest

from hoa_qa.web.app import create_app
from hoa_qa.web.settings import (
    DEFAULT_DOCUMENTS_URL,
    ConfigError,
    WebSettings,
    check_sdk_debug_logging,
)

PROD = {"VERCEL_ENV": "production"}


@pytest.mark.parametrize("var", ["TYPESAFE_LOG_LEVEL", "ANTHROPIC_LOG"])
def test_sdk_debug_logging_refused_in_production(var: str) -> None:
    with pytest.raises(ConfigError, match=var):
        check_sdk_debug_logging({**PROD, var: "DEBUG"})
    with pytest.raises(ConfigError):
        create_app(env={**PROD, var: "debug"})


@pytest.mark.parametrize("var", ["TYPESAFE_LOG_LEVEL", "ANTHROPIC_LOG"])
def test_sdk_debug_logging_warns_outside_production(
    var: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        check_sdk_debug_logging({"VERCEL_ENV": "preview", var: "debug"})
    assert var in caplog.text


def test_non_debug_sdk_logging_is_fine() -> None:
    check_sdk_debug_logging({**PROD, "TYPESAFE_LOG_LEVEL": "info", "ANTHROPIC_LOG": ""})


def test_defaults() -> None:
    s = WebSettings.from_env({})
    assert s.production is False
    assert s.on_vercel is False
    assert s.corpus_path == Path("corpus.json")
    assert s.documents_url == DEFAULT_DOCUMENTS_URL
    assert (s.rate_limit_per_hour, s.rate_limit_per_day) == (10, 50)
    assert s.monthly_budget_usd == 5.0


def test_env_overrides() -> None:
    s = WebSettings.from_env(
        {
            "VERCEL": "1",
            "CORPUS_PATH": "/data/c.json",
            "HOA_DOCUMENTS_URL": "https://docs.example.org/hoa",
            "MONTHLY_BUDGET_USD": "12.5",
            "RATE_LIMIT_PER_HOUR": "4",
            "RATE_LIMIT_PER_DAY": "20",
        }
    )
    assert s.on_vercel is True
    assert s.corpus_path == Path("/data/c.json")
    assert s.documents_url == "https://docs.example.org/hoa"
    assert s.monthly_budget_usd == 12.5
    assert (s.rate_limit_per_hour, s.rate_limit_per_day) == (4, 20)


@pytest.mark.parametrize("raw", ["", "abc", "-1", "inf", "nan"])
def test_invalid_budget_fails_closed_in_production(raw: str) -> None:
    prod = WebSettings.from_env({**PROD, "MONTHLY_BUDGET_USD": raw})
    dev = WebSettings.from_env({"MONTHLY_BUDGET_USD": raw})
    assert (prod.monthly_budget_usd, dev.monthly_budget_usd) == (0, 5.0)


def test_documents_url_must_be_https() -> None:
    with pytest.raises(ConfigError):
        WebSettings.from_env({"HOA_DOCUMENTS_URL": "http://insecure.example"})


@pytest.mark.parametrize("raw", ["0", "-3", "ten"])
def test_rate_limits_must_be_positive(raw: str) -> None:
    with pytest.raises(ConfigError):
        WebSettings.from_env({"RATE_LIMIT_PER_HOUR": raw})


def test_fake_asker_flag_disabled_in_production() -> None:
    assert WebSettings.from_env({"HOA_QA_FAKE_ASKER": "1"}).fake_asker is True
    assert WebSettings.from_env({**PROD, "HOA_QA_FAKE_ASKER": "1"}).fake_asker is False
