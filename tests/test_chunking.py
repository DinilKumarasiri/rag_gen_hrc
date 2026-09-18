import pytest

from rag.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=10, overlap=2) == []


def test_short_text_fits_in_one_chunk():
    text = "one two three"
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert chunks == ["one two three"]


def test_splits_into_expected_number_of_chunks():
    text = " ".join(f"word{i}" for i in range(20))
    chunks = chunk_text(text, chunk_size=10, overlap=0)
    assert len(chunks) == 2
    assert chunks[0] == " ".join(f"word{i}" for i in range(10))
    assert chunks[1] == " ".join(f"word{i}" for i in range(10, 20))


def test_overlap_repeats_words_between_consecutive_chunks():
    text = " ".join(f"word{i}" for i in range(10))
    chunks = chunk_text(text, chunk_size=5, overlap=2)
    # step = 3, so chunk starts are 0, 3, 6, 9
    assert chunks[0] == "word0 word1 word2 word3 word4"
    assert chunks[1] == "word3 word4 word5 word6 word7"
    # last two words of chunk 0 (word3, word4) reappear at the start of chunk 1
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_last_chunk_is_not_dropped_when_shorter_than_chunk_size():
    text = " ".join(f"word{i}" for i in range(23))
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert chunks[-1] == "word16 word17 word18 word19 word20 word21 word22"
    # every word appears in the reconstructed output
    assert "word22" in chunks[-1]


def test_non_positive_chunk_size_rejected():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0, overlap=0)


def test_negative_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, overlap=-1)


def test_overlap_not_smaller_than_chunk_size_rejected():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, overlap=10)
