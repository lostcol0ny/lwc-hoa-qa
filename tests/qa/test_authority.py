"""The code-level authority rule: support is not authority.

Reproduces the first live eval's launch blocker: the sweep ranked an informal
2022 blog (quoting the 2016 fines) above the 2023 Rules, and the model
answered "Under the current rules, the fine for a second violation is $50.00"
citing only the blog. The blog does say $50, so the support check passed.
"""

import asyncio

from qa_fakes import FakeJev, FakeProvider, claim, draft, make_asker

from hoa_qa.answer.prompt import AUTHORITY_FEEDBACK
from hoa_qa.ask import QASettings
from hoa_qa.models import Corpus, Outcome

QUESTION = "Under the current rules, what is the fine for a second violation?"

BLOG_AS_CURRENT = claim(
    "Under the current rules, the fine for a second violation is $50.00.",
    ("blog-2022-violations", "2nd offense - $50.00 fine"),
)
RULES_ANSWER = claim(
    "Under the 2023 Rules, a second violation is a $75 fine.",
    ("rules-2023-fines", "2nd violation: $75."),
)
BLOG_CONFLICT = claim(
    "An informal 2022 blog post lists the older $50 fine for a 2nd offense.",
    ("blog-2022-violations", "2nd offense - $50.00 fine"),
    kind="conflict",
    essential=False,
)


def ask(asker, question: str = QUESTION):
    return asyncio.run(asker(question))


def blog_first(jev: FakeJev) -> FakeJev:
    """Live-eval shape: the blog outscores the Rules; every claim 'supported'."""
    jev.relevance = {
        "blog-2022-violations": 0.97,
        "rules-2023-fines": 0.8,
        "rules-2016-fines": 0.6,
    }
    jev.support = 0.95
    return jev


def test_blog_only_answer_is_rejected_when_rules_were_provided(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([draft(BLOG_AS_CURRENT)] * 2)
    result = ask(make_asker(corpus, settings, blog_first(jev), provider))
    answer = result.answer
    # Never a wrong "current" fine: with the claim rejected twice, not_found.
    assert answer.outcome is Outcome.not_found
    assert "$50" not in answer.answer_text
    # The Rules passage was provided, and the retry explained the rule.
    assert 'chunk_id="rules-2023-fines"' in provider.prompts[0].user
    assert AUTHORITY_FEEDBACK in provider.prompts[1].user
    assert BLOG_AS_CURRENT.statement in provider.prompts[1].user
    # Rejected by code before any support check was spent on it.
    assert jev.calls_of("c") == []


def test_retry_answers_from_the_rules_and_reports_the_blog_as_conflict(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider(
        [draft(BLOG_AS_CURRENT), draft(RULES_ANSWER, BLOG_CONFLICT)]
    )
    answer = ask(make_asker(corpus, settings, blog_first(jev), provider)).answer
    assert answer.outcome is Outcome.answered
    assert answer.answer_text == RULES_ANSWER.statement
    assert "$50" not in answer.answer_text
    assert answer.conflicts_noted == (BLOG_CONFLICT.statement,)
    assert [c.chunk_id for c in answer.citations] == [
        "rules-2023-fines",
        "blog-2022-violations",
    ]


def test_superseded_only_answer_is_rejected(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    old = claim(
        "A second offense is a $50 fine.",
        ("rules-2016-fines", "2nd offense: $50."),
    )
    provider = FakeProvider([draft(old)] * 2)
    answer = ask(make_asker(corpus, settings, blog_first(jev), provider)).answer
    assert answer.outcome is Outcome.not_found


def test_mixed_citations_are_judged_on_the_official_source_only(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    """A $50 claim can't borrow support from the blog by also citing the Rules."""
    mixed = claim(
        "A second violation is a $50 fine.",
        ("rules-2023-fines", "2nd violation: $75."),
        ("blog-2022-violations", "2nd offense - $50.00 fine"),
    )
    rules_text = next(c.text_clean for c in corpus.chunks if c.id == "rules-2023-fines")
    judged: list[list[str]] = []

    def support(statement: str, ids) -> float:
        judged.append(list(ids))
        return 0.9 if "blog-2022-violations" in ids else 0.1

    jev = blog_first(jev)
    jev.support = support
    provider = FakeProvider([draft(mixed)] * 2)
    answer = ask(make_asker(corpus, settings, jev, provider)).answer
    assert judged and all(ids == ["rules-2023-fines"] for ids in judged)
    state, _ = next(c for c in jev.calls if "c0" in c[1])
    assert [p["text"] for p in state["claims"][0]["passages"]] == [rules_text]
    assert answer.outcome is Outcome.not_found


def test_kept_answer_shows_only_official_citations(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    both = claim(
        RULES_ANSWER.statement,
        ("rules-2023-fines", "2nd violation: $75."),
        ("blog-2022-violations", "2nd offense - $50.00 fine"),
    )
    provider = FakeProvider([draft(both)])
    answer = ask(make_asker(corpus, settings, blog_first(jev), provider)).answer
    assert answer.outcome is Outcome.answered
    assert [c.chunk_id for c in answer.citations] == ["rules-2023-fines"]


def test_website_is_official_next_to_governing_passages(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.relevance = {"website-2026-dues": 0.95, "declaration-8-assessment": 0.9}
    dues = claim(
        "The 2026 assessment is $452 a year.",
        ("website-2026-dues", "2026 Assessment Prices: $452 year"),
    )
    answer = ask(
        make_asker(corpus, settings, jev, FakeProvider([draft(dues)])),
        "How much are the dues?",
    ).answer
    assert answer.outcome is Outcome.answered
    assert answer.answer_text == dues.statement


def test_informal_answer_may_stand_alone_but_not_as_current(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    """Without an authoritative passage an informal source can answer, labeled."""
    jev.relevance = {"blog-2022-violations": 0.9}
    labeled = claim(
        "An informal 2022 blog post lists a $50 fine for a 2nd offense.",
        ("blog-2022-violations", "2nd offense - $50.00 fine"),
    )
    answer = ask(
        make_asker(corpus, settings, jev, FakeProvider([draft(labeled)]))
    ).answer
    assert answer.outcome is Outcome.answered

    provider = FakeProvider([draft(BLOG_AS_CURRENT)] * 2)
    answer = ask(make_asker(corpus, settings, jev, provider)).answer
    assert answer.outcome is Outcome.not_found
    assert AUTHORITY_FEEDBACK in provider.prompts[1].user


def test_conflict_claims_may_cite_only_informal_sources(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([draft(RULES_ANSWER, BLOG_CONFLICT)])
    answer = ask(make_asker(corpus, settings, blog_first(jev), provider)).answer
    assert answer.outcome is Outcome.answered
    assert answer.conflicts_noted == (BLOG_CONFLICT.statement,)
    assert len(provider.prompts) == 1
