"""Fixtures for the QA tests, driven by the mini corpus."""

import pytest
from qa_fakes import FIXTURE, FakeJev

from hoa_qa.ask import QASettings
from hoa_qa.models import Corpus, load_corpus


@pytest.fixture
def corpus() -> Corpus:
    return load_corpus(FIXTURE)


@pytest.fixture
def jev(corpus: Corpus) -> FakeJev:
    return FakeJev(
        relevance={"rules-2023-fines": 0.9, "rules-2016-fines": 0.7},
        text_to_id={c.text_clean: c.id for c in corpus.chunks},
    )


@pytest.fixture
def settings() -> QASettings:
    return QASettings.from_env({})
