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

