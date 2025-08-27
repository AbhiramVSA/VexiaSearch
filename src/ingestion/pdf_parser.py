import hashlib
from dataclasses import dataclass
from typing import List, Optional, Dict
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:  # optional in test env
    HAS_FITZ = False
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import tiktoken
try:
    from datasketch import MinHash  # type: ignore
    HAS_MINHASH = True
except ImportError:  # optional dependency
    HAS_MINHASH = False
    class MinHash:  # minimal fallback (set-based, not probabilistic)
        def __init__(self, num_perm: int = 64):
            self._tokens = set()
        def update(self, b: bytes):
            self._tokens.add(b)
        def jaccard(self, other: 'MinHash') -> float:
            if not self._tokens and not other._tokens:
                return 1.0
            inter = len(self._tokens & other._tokens)
            union = len(self._tokens | other._tokens)
            return inter / union if union else 0.0

enc = tiktoken.get_encoding("cl100k_base")

@dataclass
class Block:
    page: int
    text: str
    block_type: str = "text"  # text | table_heuristic
    heading_level: Optional[int] = None

@dataclass
class Chunk:
    id: str
    text: str
    page_start: int
    page_end: int
    section_title: Optional[str]
    checksum: str
    content_type: str = "text"

# --- Extraction ---

def extract_blocks(path: str) -> List[Block]:
    blocks: List[Block] = []
    if HAS_FITZ:
        try:
            doc = fitz.open(path)
            for page_index, page in enumerate(doc):
                try:
                    page_dict = page.get_text("dict")
                    for b in page_dict.get("blocks", []):
                        if "lines" not in b:
                            continue
                        line_texts = []
                        for l in b.get("lines", []):
                            span_text = "".join(span.get("text", "") for span in l.get("spans", []))
                            if span_text.strip():
                                line_texts.append(span_text.strip())
                        if not line_texts:
                            continue
                        text_join = "\n".join(line_texts)
                        is_table = _heuristic_table(line_texts)
                        heading_level = _heuristic_heading(line_texts)
                        blocks.append(Block(page=page_index+1, text=text_join, block_type="table_heuristic" if is_table else "text", heading_level=heading_level))
                except Exception:
                    continue
        except Exception:
            pass
    if not blocks:  # fallback to pdfminer
        for page_no, page_layout in enumerate(extract_pages(path)):
            lines = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    t = element.get_text().strip()
                    if t:
                        lines.append(t)
            if lines:
                blocks.append(Block(page=page_no+1, text="\n".join(lines)))
    return blocks

# --- Heuristics ---

def _heuristic_table(lines: List[str]) -> bool:
    if len(lines) < 2:
        return False
    # presence of multiple delimiters or consistent column spacing
    delimiters = sum(1 for ln in lines if ("  " in ln or "|" in ln or "\t" in ln))
    return delimiters / len(lines) > 0.6

def _heuristic_heading(lines: List[str]) -> Optional[int]:
    first = lines[0]
    if len(first) < 120 and first.isupper():
        return 1
    if len(first) < 90 and first.endswith(":"):
        return 2
    return None

# --- Chunking ---

def blocks_to_chunks(blocks: List[Block], max_tokens: int = 900, overlap: int = 120) -> List[Chunk]:
    chunks: List[Chunk] = []
    buf: List[Block] = []
    buf_tokens = 0

    def flush(next_block_tokens: int = 0):
        nonlocal buf, buf_tokens
        if not buf:
            return
        text = "\n\n".join(b.text for b in buf)
        pages = [b.page for b in buf]
        section_title = next((b.text.split("\n")[0][:120] for b in buf if b.heading_level), None)
        checksum = hashlib.sha256(text.encode()).hexdigest()[:16]
        chunk = Chunk(
            id=checksum,
            text=text,
            page_start=min(pages),
            page_end=max(pages),
            section_title=section_title,
            checksum=checksum,
            content_type="text" if all(b.block_type=="text" for b in buf) else "table_heuristic"
        )
        chunks.append(chunk)
        if overlap > 0:
            # token overlap using tail tokens
            all_toks = enc.encode(text)
            tail = enc.decode(all_toks[-overlap:]) if len(all_toks) > overlap else text
            # keep pseudo block for overlap continuity
            buf = [Block(page=buf[-1].page, text=tail)]
            buf_tokens = len(enc.encode(tail))
        else:
            buf = []
            buf_tokens = 0

    for b in blocks:
        t = b.text.strip()
        if not t:
            continue
        toks = len(enc.encode(t))
        if buf_tokens + toks > max_tokens and buf:
            flush(next_block_tokens=toks)
        buf.append(b)
        buf_tokens += toks
    flush()
    return chunks

# --- Near-duplicate removal ---

def dedupe_chunks(chunks: List[Chunk], num_perm: int = 64, threshold: float = 0.85) -> List[Chunk]:
    """Near-duplicate removal; falls back to checksum uniqueness if MinHash unavailable."""
    if not HAS_MINHASH:
        out: Dict[str, Chunk] = {}
        for ch in chunks:
            out.setdefault(ch.checksum, ch)
        return list(out.values())
    seen: List[Chunk] = []
    sigs: List[MinHash] = []
    for ch in chunks:
        m = MinHash(num_perm=num_perm)
        for token in ch.text.split():
            m.update(token.encode())
        if any(m.jaccard(other) >= threshold for other in sigs):
            continue
        seen.append(ch)
        sigs.append(m)
    return seen

__all__ = ["Block", "Chunk", "extract_blocks", "blocks_to_chunks", "dedupe_chunks"]
