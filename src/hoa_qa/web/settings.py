"""Web-layer configuration, read once from environment variables."""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hoa_qa.budget import is_production
from hoa_qa.models import validate_https

logger = logging.getLogger("hoa_qa.web")

REPO_ROOT = Path(__file__).resolve().parents[3]
# Relative to the working directory: the repo root locally, and the project
# base on Vercel (where the package may be installed outside the repo tree).
DEFAULT_CORPUS_PATH = Path("corpus.json")
DEFAULT_DOCUMENTS_URL = "https://lakewoodcreekhoa.com/"
DEFAULT_MONTHLY_BUDGET_USD = 5.0
DEFAULT_RATE_LIMIT_PER_HOUR = 10
DEFAULT_RATE_LIMIT_PER_DAY = 50

DISCLAIMER = (
    "Unofficial tool, not legal advice; the governing documents and the Board "
    "are authoritative. Don't include personal information; questions are "
    "processed by third-party AI services."
)

# Both SDKs log full request bodies (neighbors' questions) at debug level.
SDK_DEBUG_ENV_VARS = ("TYPESAFE_LOG_LEVEL", "ANTHROPIC_LOG")


class ConfigError(RuntimeError):
    """The environment is unsafe or invalid; the app refuses to start."""


def check_sdk_debug_logging(env: Mapping[str, str]) -> None:
    """Refuse to start in production if an SDK would log request bodies."""
    enabled = [
        name for name in SDK_DEBUG_ENV_VARS if env.get(name, "").lower() == "debug"
    ]
    if not enabled:
        return
    message = (
        f"{', '.join(enabled)}=debug makes the AI SDKs log full request bodies, "
        "including users' questions"
    )
    if is_production(env):
        raise ConfigError(f"{message}; unset it before deploying to production")
    logger.warning("%s; never enable this in production", message)


@dataclass(frozen=True)
class WebSettings:
    production: bool
    on_vercel: bool
    corpus_path: Path
    documents_url: str
    monthly_budget_usd: float
    rate_limit_per_hour: int
    rate_limit_per_day: int
    fake_asker: bool

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "WebSettings":
        env = os.environ if env is None else env
        production = is_production(env)
        documents_url = env.get("HOA_DOCUMENTS_URL", "").strip() or (
            DEFAULT_DOCUMENTS_URL
        )
        try:
            validate_https(documents_url)
        except ValueError as exc:
            raise ConfigError("HOA_DOCUMENTS_URL must be an https:// URL") from exc
        return cls(
            production=production,
            on_vercel=env.get("VERCEL") == "1",
            corpus_path=Path(env.get("CORPUS_PATH") or DEFAULT_CORPUS_PATH),
            documents_url=documents_url,
            monthly_budget_usd=_budget(env, production),
            rate_limit_per_hour=_positive_int(
                env, "RATE_LIMIT_PER_HOUR", DEFAULT_RATE_LIMIT_PER_HOUR
            ),
            rate_limit_per_day=_positive_int(
                env, "RATE_LIMIT_PER_DAY", DEFAULT_RATE_LIMIT_PER_DAY
            ),
            # A dev convenience only; never honored in production.
            fake_asker=env.get("HOA_QA_FAKE_ASKER") == "1" and not production,
        )


def _budget(env: Mapping[str, str], production: bool) -> float:
    """Parse MONTHLY_BUDGET_USD. Missing or invalid in production means $0."""
    raw = env.get("MONTHLY_BUDGET_USD", "").strip()
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if value >= 0 and value != float("inf"):
        return value
    if production:
        logger.error("MONTHLY_BUDGET_USD missing or invalid; failing closed at $0")
        return 0.0
    return DEFAULT_MONTHLY_BUDGET_USD


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ConfigError(f"{name} must be a positive integer")
    return value
