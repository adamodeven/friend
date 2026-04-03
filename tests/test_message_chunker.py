from app.llm.message_chunker import MessageChunker


def test_chunker_splits_on_semantic_boundaries():
    chunker = MessageChunker()
    text = (
        "got your update. you're blocked on the website prerequisite. "
        "that makes sense. do the website pass first, then jump back to the assignment."
    )
    chunks = chunker.chunk(text, max_chunk_length=70, max_chunks=3)
    assert len(chunks) >= 2
    assert chunks[0].endswith(".")


def test_chunker_splits_run_on_text_on_connectors():
    chunker = MessageChunker()
    text = "youre making good progress now keep that same pace tonight and send me the next blocker"
    chunks = chunker.chunk(text, max_chunk_length=48, max_chunks=3)
    assert len(chunks) >= 2


def test_chunker_prefers_break_before_next_move_cue():
    chunker = MessageChunker()
    text = "locked in. finish the statics set tonight. next move: do problems 1 to 3 now and text me when you're through them."
    chunks = chunker.chunk(text, max_chunk_length=120, max_chunks=3, soft_chunk_length=58)
    assert len(chunks) == 2
    assert chunks[1].lower().startswith("next move:")


def test_normalize_messages_rechunks_long_single_bubble_at_soft_limit():
    chunker = MessageChunker()
    messages = [
        "you've got two things tonight. finish the portfolio edit first. then send me the blocker if the CAD pass still isn't moving."
    ]
    chunks = chunker.normalize_messages(messages, max_chunk_length=140, max_chunks=3, soft_chunk_length=62)
    assert len(chunks) >= 2
    assert any("then send me the blocker" in chunk.lower() for chunk in chunks)
