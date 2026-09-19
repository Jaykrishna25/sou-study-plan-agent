"""
Ingestion for the Meme & Internet Culture Historian.

Pipeline stages:
  document loading  -> the curated markdown reference set
  text splitting    -> ONE ENTRY PER CHUNK, not a fixed character split
  embeddings        -> HuggingFace BAAI/bge-small-en-v1.5
  vector store      -> Chroma
  retriever         -> as_retriever(search_kwargs={"k": 3})

Design decision worth defending: the splitter is a section splitter, not a
character splitter. Each entry is short and self-contained - origin, meaning,
variations - and a 500-character window would cut the origin away from the
meaning, so a query about where something came from would retrieve the half of
the entry that does not say. Splitting on the `##` heading keeps each entry
whole and makes retrieval an entry-level operation, which is what a lookup
question actually wants.

The "Also called" lines in the corpus exist for the same reason. Nobody searches
for "Distracted Boyfriend"; they search for "the guy looking back". Those
alternative names are embedded with the entry, which is what makes the fuzzy
queries in the brief work.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CORPUS = Path("data/memes.md")
COLLECTION = "meme_history"

# Only used for an entry long enough to warrant it; most never hit this.
MAX_CHUNK = 1400
CHUNK_OVERLAP = 150


@dataclass
class Entry:
    name: str
    body: str
    aliases: list[str]

    def as_text(self) -> str:
        return f"{self.name}\n{self.body}".strip()


_EMBEDDINGS = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDINGS


def load_entries(path: Path = CORPUS) -> list[Entry]:
    """Split the corpus on `## ` headings - one entry per chunk."""
    if not path.exists():
        raise FileNotFoundError(f"Reference corpus not found at {path}")

    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^## ", text, flags=re.M)[1:]      # drop the preamble

    entries: list[Entry] = []
    for part in parts:
        lines = part.strip().splitlines()
        if not lines:
            continue
        name = lines[0].strip()
        body = "\n".join(lines[1:]).strip()

        aliases: list[str] = []
        m = re.search(r"^Also called:\s*(.+)$", body, flags=re.M)
        if m:
            aliases = [a.strip(" .") for a in m.group(1).split(",") if a.strip()]

        entries.append(Entry(name=name, body=body, aliases=aliases))
    return entries


def build_index(entries: list[Entry]) -> tuple[Chroma, int]:
    """Embed one Document per entry. Long entries are split, short ones are not."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: list[Document] = []
    for e in entries:
        text = e.as_text()
        meta = {
            "name": e.name,
            "aliases": ", ".join(e.aliases),
            "source": "meme-corpus",
        }
        if len(text) <= MAX_CHUNK:
            docs.append(Document(page_content=text, metadata=meta))
        else:
            for piece in splitter.split_text(text):
                # Keep the name on every piece so a partial chunk is still
                # identifiable - otherwise the second half of a long entry
                # retrieves as anonymous text.
                docs.append(Document(page_content=f"{e.name}\n{piece}", metadata=meta))

    store = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=COLLECTION,
    )
    return store, len(docs)


def get_retriever(store: Chroma):
    return store.as_retriever(search_kwargs={"k": 3})
