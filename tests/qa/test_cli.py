"""CLI smoke tests with fake providers."""

import json

import pytest
from qa_fakes import FINES_DRAFT, FIXTURE, FakeJev, FakeProvider

from hoa_qa.ask import QAAsker, QASettings, build_asker
from hoa_qa.cli import main
from hoa_qa.models import Corpus


def fake_factory(corpus: Corpus, settings: QASettings) -> QAAsker:
    jev = FakeJev(
        relevance={"rules-2023-fines": 0.9},
        text_to_id={c.text_clean: c.id for c in corpus.chunks},
    )
    return build_asker(corpus, settings, jev=jev, provider=FakeProvider([FINES_DRAFT]))


def test_ask_prints_answer_citation_and_disclaimer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        ["ask", "What is the fine for a third violation?", "--corpus", str(FIXTURE)],
        asker_factory=fake_factory,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "$125" in out
    assert "Rules 2023, Fines, p. 15" in out
    assert "#page=15" in out
    assert "not legal advice" in out
    assert "[answered]" in out


def test_ask_json(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["ask", "fines?", "--corpus", str(FIXTURE), "--json"],
        asker_factory=fake_factory,
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["answer"]["outcome"] == "answered"
    assert payload["estimated_cost_usd"] > 0


def test_missing_corpus_is_a_clean_error(
    capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    code = main(
        ["ask", "q", "--corpus", str(tmp_path / "nope.json")],
        asker_factory=fake_factory,
    )
    assert code == 2
    assert "hoa-qa:" in capsys.readouterr().err


def test_missing_keys_is_a_clean_error(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    code = main(["ask", "q", "--corpus", str(FIXTURE)])
    assert code == 2
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "ask" in capsys.readouterr().out
