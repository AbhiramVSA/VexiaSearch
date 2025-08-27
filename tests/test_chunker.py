from src.ingestion.pdf_parser import Block, blocks_to_chunks, dedupe_chunks

def test_blocks_to_chunks_basic():
    blocks = [Block(page=1, text='Heading:\nINTRO', heading_level=1), Block(page=1, text='Some content '*50)]
    chunks = blocks_to_chunks(blocks, max_tokens=200, overlap=20)
    assert len(chunks) >= 1
    assert chunks[0].page_start == 1

def test_dedupe():
    b = [Block(page=1, text='repeat content'), Block(page=2, text='repeat content')]
    chunks = blocks_to_chunks(b, max_tokens=50)
    deduped = dedupe_chunks(chunks, threshold=0.9)
    assert len(deduped) == 1
