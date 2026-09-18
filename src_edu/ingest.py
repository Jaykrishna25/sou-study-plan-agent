"""
Ingestion for the Transcript & Study Plan Agent.

Pipeline stages this file covers:
  document loading  -> PyPDFLoader / pandas for tabular transcripts
  text splitting    -> RecursiveCharacterTextSplitter, plus per-course documents
  embeddings        -> HuggingFace BAAI/bge-small-en-v1.5
  vector store      -> Chroma, one collection per uploaded transcript
  retriever         -> as_retriever(search_kwargs={"k": 3})

Design decision worth defending: each course is indexed as its own Document in
addition to the whole transcript. The track brief asks for retrieval that works
for both a narrow question ("what grade did I get in databases") and a broad one
("what have I not covered yet"). A fixed character split cuts through the middle
of a course row and puts a subject name in one chunk and its marks in the next,
which answers the narrow question wrongly. One document per course keeps a
subject, its credits, its marks and its grade together, and the full-transcript
chunks still serve the broad questions.
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
CHUNK_SIZE = 500
CHUNK_OVERLAP = 120

# Below this percentage a subject is treated as a weak area.
WEAK_THRESHOLD = 60.0
# Below this it is urgent rather than merely weak.
CRITICAL_THRESHOLD = 45.0

_GRADE_POINTS = {"O": 10, "A+": 9, "A": 9, "B+": 8, "B": 7, "C": 6, "D": 5, "E": 4, "F": 0}


@dataclass
class Course:
    semester: int
    code: str
    subject: str
    credits: float
    total: float          # percentage or marks out of 100
    grade: str

    @property
    def weak(self) -> bool:
        return self.total < WEAK_THRESHOLD

    @property
    def critical(self) -> bool:
        return self.total < CRITICAL_THRESHOLD

    def as_text(self) -> str:
        band = "critical" if self.critical else ("weak" if self.weak else "satisfactory")
        return (
            f"Semester {self.semester} - {self.code} {self.subject}. "
            f"Credits {self.credits:g}. Score {self.total:g} out of 100. "
            f"Grade {self.grade}. Performance band: {band}."
        )


_EMBEDDINGS = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once per process - it is slow to construct."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDINGS


# ---------------------------------------------------------------- loading

def read_upload(file_bytes: bytes, filename: str) -> tuple[str, list[Course]]:
    """Return (raw_text, courses) for a transcript in CSV, Excel, PDF or text."""
    lower = filename.lower()

    if lower.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
        return _df_to_text(df), _df_to_courses(df)

    if lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
        return _df_to_text(df), _df_to_courses(df)

    if lower.endswith(".pdf"):
        text = _load_pdf(file_bytes)
        return text, _text_to_courses(text)

    text = file_bytes.decode("utf-8", errors="replace")
    return text, _text_to_courses(text)


def _load_pdf(file_bytes: bytes) -> str:
    """Load a PDF transcript with LangChain's PyPDFLoader.

    PyPDFLoader reads from a path, so the upload is written to a temporary file
    and deleted immediately - a transcript is personal data and should not
    outlive the request.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(file_bytes)
        tmp.close()
        return "\n".join(p.page_content or "" for p in PyPDFLoader(tmp.name).load())
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def _num(v, default=0.0) -> float:
    try:
        f = float(str(v).strip().replace("%", ""))
        return f if f == f else default      # NaN check
    except (TypeError, ValueError):
        return default


def _df_to_courses(df: pd.DataFrame) -> list[Course]:
    """Map a tabular transcript to Course rows, matching headers loosely.

    Real transcripts vary: "Subject" or "Course Title", "Total" or "Marks" or
    "Percentage". Demanding exact headers would fail on the first real file.
    """
    cols = {_norm(c): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        for n in names:                       # substring fallback
            for key, original in cols.items():
                if n in key:
                    return original
        return None

    c_sem = pick("semester", "sem", "term")
    c_code = pick("code", "coursecode", "subjectcode")
    c_sub = pick("subject", "coursetitle", "course", "title", "paper")
    c_cred = pick("credits", "credit")
    c_total = pick("total", "marks", "percentage", "score", "obtained")
    c_grade = pick("grade")
    c_int = pick("internal", "internalmarks", "ia")
    c_ext = pick("external", "externalmarks", "ese")

    if c_sub is None:
        return []

    out: list[Course] = []
    for _, row in df.iterrows():
        subject = str(row[c_sub]).strip()
        if not subject or subject.lower() == "nan":
            continue

        if c_total is not None:
            total = _num(row[c_total])
        elif c_int is not None and c_ext is not None:
            total = _num(row[c_int]) + _num(row[c_ext])
        else:
            total = 0.0

        grade = str(row[c_grade]).strip().upper() if c_grade is not None else ""
        # A transcript with grades but no marks is still usable.
        if total <= 0 and grade in _GRADE_POINTS:
            total = _GRADE_POINTS[grade] * 10.0

        out.append(Course(
            semester=int(_num(row[c_sem], 0)) if c_sem is not None else 0,
            code=str(row[c_code]).strip() if c_code is not None else "",
            subject=subject,
            credits=_num(row[c_cred], 3.0) if c_cred is not None else 3.0,
            total=total,
            grade=grade or _grade_from_total(total),
        ))

    out.sort(key=lambda c: (c.semester, c.code))
    return out


def _grade_from_total(total: float) -> str:
    for cut, g in [(90, "O"), (80, "A"), (70, "B+"), (60, "B"), (50, "C"), (45, "D"), (0, "E")]:
        if total >= cut:
            return g
    return "E"


_ROW = re.compile(
    r"^\s*(?P<code>[A-Z]{2,4}\s?\d{3})\s+(?P<subject>.+?)\s+(?P<total>\d{1,3})\s*(?P<grade>[A-EO]\+?)?\s*$"
)


def _text_to_courses(text: str) -> list[Course]:
    """Best-effort parse of a transcript that arrived as free text or PDF."""
    courses: list[Course] = []
    semester = 0
    for line in text.splitlines():
        sem = re.search(r"semester\s*[:\-]?\s*(\d)", line, re.I)
        if sem:
            semester = int(sem.group(1))
        m = _ROW.match(line.strip())
        if not m:
            continue
        total = _num(m.group("total"))
        courses.append(Course(
            semester=semester,
            code=m.group("code").replace(" ", ""),
            subject=m.group("subject").strip(),
            credits=3.0,
            total=total,
            grade=(m.group("grade") or _grade_from_total(total)).upper(),
        ))
    return courses


def _df_to_text(df: pd.DataFrame) -> str:
    lines = ["ACADEMIC TRANSCRIPT", ""]
    lines.append(" | ".join(str(c) for c in df.columns))
    for _, row in df.iterrows():
        lines.append(" | ".join(str(row[c]) for c in df.columns))
    return "\n".join(lines)


# ---------------------------------------------------------------- indexing

def build_index(raw_text: str, courses: list[Course]) -> tuple[Chroma, int]:
    """Embed the transcript into a Chroma collection scoped to this upload."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " | ", ". ", " ", ""],
    )

    docs: list[Document] = [
        Document(page_content=chunk, metadata={"source": "transcript", "kind": "section"})
        for chunk in splitter.split_text(raw_text)
    ]

    # One document per course - see the module docstring for why.
    for c in courses:
        docs.append(Document(
            page_content=c.as_text(),
            metadata={
                "source": "transcript", "kind": "course",
                "code": c.code, "subject": c.subject,
                "semester": c.semester, "weak": c.weak,
            },
        ))

    store = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name="transcript_" + uuid.uuid4().hex[:8],
    )
    return store, len(docs)


def get_retriever(store: Chroma):
    return store.as_retriever(search_kwargs={"k": 3})
