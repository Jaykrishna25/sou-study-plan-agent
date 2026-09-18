"""
Ingestion pipeline: load -> split -> embed -> store -> retrieve.

Scoped to ONE uploaded statement per session. A portfolio is personal
data, so each upload gets its own in-memory Chroma collection rather
than being added to a shared corpus.
"""
from __future__ import annotations

import io
import os
import re
import tempfile
import uuid
from dataclasses import dataclass

import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Statements are dense and numeric. Small chunks with generous overlap keep a
# figure attached to the label above it instead of stranding it in its own chunk.
CHUNK_SIZE = 400
CHUNK_OVERLAP = 120


@dataclass
class Holding:
    symbol: str
    name: str
    sector: str
    quantity: float
    buy_price: float
    buy_date: str | None = None


_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Loaded once - the model takes a few seconds to initialise."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def read_upload(file_bytes: bytes, filename: str) -> tuple[str, list[Holding]]:
    """
    Returns (raw_text, holdings).

    Holdings are parsed deterministically so the analysis tool works on real
    numbers. The raw text is what gets embedded for follow-up questions.
    """
    lower = filename.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
        return _df_to_text(df), _df_to_holdings(df)
    if lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
        return _df_to_text(df), _df_to_holdings(df)
    if lower.endswith(".pdf"):
        text = _load_pdf(file_bytes)
        return text, _text_to_holdings(text)
    raise ValueError("Upload a CSV, Excel or PDF statement.")


def _load_pdf(file_bytes: bytes) -> str:
    """Load a PDF statement with LangChain's PyPDFLoader.

    PyPDFLoader reads from a path rather than bytes, so the upload is written to
    a temporary file that is removed immediately afterwards - an uploaded
    statement is personal data and should not outlive the request.

    It returns one Document per page with page metadata, which is why it is
    preferred here over calling pypdf directly: page provenance survives into
    the splitter, so a retrieved chunk can still say which page it came from.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(file_bytes)
        tmp.close()
        pages = PyPDFLoader(tmp.name).load()
        return "\n".join(p.page_content or "" for p in pages)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def _df_to_holdings(df: pd.DataFrame) -> list[Holding]:
    cols = {_norm(c): c for c in df.columns}

    def pick(*names, required=True):
        for n in names:
            if n in cols:
                return cols[n]
        if required:
            raise ValueError(
                "Could not find a column for " + names[0] +
                ". Found: " + ", ".join(df.columns)
            )
        return None

    c_sym = pick("symbol", "ticker", "scrip", "code")
    c_qty = pick("quantity", "qty", "units", "shares")
    c_buy = pick("buyprice", "purchaseprice", "avgprice", "averageprice", "cost")
    c_name = pick("name", "company", "security", required=False)
    c_sec = pick("sector", "industry", required=False)
    c_date = pick("buydate", "purchasedate", "date", required=False)

    out: list[Holding] = []
    for _, r in df.iterrows():
        try:
            qty = float(r[c_qty])
            price = float(r[c_buy])
        except (TypeError, ValueError):
            continue          # skip totals rows and blank lines
        if qty <= 0:
            continue
        out.append(Holding(
            symbol=str(r[c_sym]).strip().upper(),
            name=str(r[c_name]).strip() if c_name else str(r[c_sym]).strip(),
            sector=str(r[c_sec]).strip() if c_sec else "Unclassified",
            quantity=qty,
            buy_price=price,
            buy_date=str(r[c_date]).strip() if c_date else None,
        ))
    return out


_PDF_ROW = re.compile(
    r"^([A-Z][A-Z0-9.\-]{1,14})\s+(.+?)\s+([\d,]+(?:\.\d+)?)\s+([\d,]+(?:\.\d+)?)\s*$"
)


def _text_to_holdings(text: str) -> list[Holding]:
    """Best-effort parse of a tabular PDF statement: SYMBOL NAME QTY PRICE."""
    out: list[Holding] = []
    for line in text.splitlines():
        m = _PDF_ROW.match(line.strip())
        if not m:
            continue
        sym, name, qty, price = m.groups()
        try:
            out.append(Holding(
                symbol=sym.upper(),
                name=name.strip(),
                sector="Unclassified",
                quantity=float(qty.replace(",", "")),
                buy_price=float(price.replace(",", "")),
            ))
        except ValueError:
            continue
    return out


def _df_to_text(df: pd.DataFrame) -> str:
    """One line per holding reads better for retrieval than a raw CSV dump."""
    lines = ["PORTFOLIO STATEMENT", ""]
    for _, r in df.iterrows():
        lines.append(" | ".join(f"{c}: {r[c]}" for c in df.columns))
    return "\n".join(lines)


def build_index(raw_text: str, holdings: list[Holding]):
    """
    One Chroma collection per upload. Each holding also becomes its own
    document so a question about a single stock retrieves that stock's row
    rather than a slice of the middle of the table.
    """
    docs = [Document(page_content=raw_text, metadata={"source": "statement", "kind": "full"})]
    for h in holdings:
        docs.append(Document(
            page_content=(
                f"Holding: {h.name} ({h.symbol})\n"
                f"Sector: {h.sector}\n"
                f"Quantity: {h.quantity}\n"
                f"Purchase price: INR {h.buy_price} per share\n"
                f"Purchase date: {h.buy_date or 'not stated'}\n"
                f"Invested amount: INR {round(h.quantity * h.buy_price, 2)}"
            ),
            metadata={"source": "statement", "kind": "holding", "symbol": h.symbol},
        ))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " | ", ". ", " "],
    )
    chunks = splitter.split_documents(docs)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="portfolio_" + uuid.uuid4().hex[:8],
    )
    return store, len(chunks)


def get_retriever(store):
    return store.as_retriever(search_kwargs={"k": 3})
