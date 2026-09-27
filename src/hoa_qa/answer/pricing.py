"""Every per-token price the cost estimate uses, in USD per million tokens.

Sources, read 2026-09-27:
- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
  (base input / output rates; this app sends no cache or batch traffic)
- TypeSafe Jev 1.13: https://docs.typesafe.ai/models ($0.042 per 1M input
  tokens; output tokens are free)
Update this file, and only this file, when prices change.
"""

from dataclasses import dataclass

JEV_INPUT_USD_PER_MTOK = 0.042


@dataclass(frozen=True)
class ModelPrice:
    input_usd_per_mtok: float
    output_usd_per_mtok: float


ANSWER_MODEL_PRICES: dict[str, ModelPrice] = {
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
    "claude-sonnet-5": ModelPrice(2.00, 10.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-opus-5": ModelPrice(5.00, 25.00),
    "claude-opus-5-5": ModelPrice(4.00, 20.00),
}

# An unlisted ANSWER_MODEL is costed at the most expensive current tier
# (Fable 5.1) so the monthly budget errs toward stopping early, not late.
FALLBACK_PRICE = ModelPrice(10.00, 50.00)


def answer_price(model: str) -> ModelPrice:
    """Look up by exact ID, then by alias prefix (e.g. a dated snapshot ID)."""
    if model in ANSWER_MODEL_PRICES:
        return ANSWER_MODEL_PRICES[model]
    # Longest alias first, so claude-opus-5-5-* never matches claude-opus-5.
    for alias in sorted(ANSWER_MODEL_PRICES, key=len, reverse=True):
        if model.startswith(f"{alias}-"):
            return ANSWER_MODEL_PRICES[alias]
    return FALLBACK_PRICE


def jev_cost_usd(input_tokens: int) -> float:
    return input_tokens * JEV_INPUT_USD_PER_MTOK / 1_000_000


def answer_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = answer_price(model)
    return (
        input_tokens * price.input_usd_per_mtok
        + output_tokens * price.output_usd_per_mtok
    ) / 1_000_000
