"""
Class WhatsApp group ingestion, for the Transcript & Study Plan Agent.

Why this exists
---------------
At most Indian universities the real announcement channel is not the portal.
It is the class WhatsApp group: exam dates, submission deadlines, room changes
and "bring your practical file tomorrow" all arrive there and are never written
down anywhere else. A student who scrolls back three weeks to find one message
is doing retrieval by hand. That is exactly the job this agent should take.

Pipeline stages this file covers:
  document loading  -> LangChain's WhatsAppChatLoader
  redaction         -> phone numbers removed before anything is embedded
  text splitting    -> one Document per day, plus one per announcement
  embeddings        -> shared with the transcript index (BAAI/bge-small-en-v1.5)
  vector store      -> Chroma, a separate collection per upload
  retriever         -> as_retriever(search_kwargs={"k": 4})

The privacy problem, and what is done about it
----------------------------------------------
A WhatsApp export is not one person's data. It contains every message every
member of that group ever sent, and in most exports the senders who are not in
the uploader's contacts appear as raw phone numbers. A student uploading their
class group is handing over sixty other people's data, and none of those sixty
were asked.

That cannot be waved away, so three things are done here and none of them are
optional:

  1. Phone numbers are removed at load time - both as sender names, which
     become stable pseudonyms ("Member 3"), and anywhere inside message text.
     Nothing downstream ever sees a number, so nothing downstream can leak one.
  2. Attachments are not read. "<Media omitted>" is all an export contains
     anyway, but the intent matters: this reads text, not photographs.
  3. The file is written to a temporary path only because WhatsAppChatLoader
     reads from a path rather than from bytes, and it is deleted in a finally
     block. The chat is never persisted.

Display names that are not phone numbers are kept, because "what did sir say
about the viva" is the question people actually ask, and stripping names makes
the feature useless. That is a deliberate trade and it should be stated out
loud rather than hidden.
"""
from __future__ import annotations

import os
import re
import tempfile
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_community.document_loaders import WhatsAppChatLoader
from langchain_chroma import Chroma

from src_edu.ingest import get_embeddings


# --------------------------------------------------------------- data types

@dataclass
class ChatMessage:
    date: str
    sender: str
    text: str

    def as_text(self) -> str:
        return f"{self.sender} on {self.date}: {self.text}"


@dataclass
class ChatLoadResult:
    messages: list[ChatMessage]
    announcements: list[ChatMessage]
    senders: list[str]
    days: int
    #: How many sender names were phone numbers and became pseudonyms.
    numbers_redacted: int = 0
    #: How many numbers were removed from inside message bodies.
    inline_numbers_redacted: int = 0
    #: System lines, media placeholders and joins that were dropped.
    dropped: int = 0
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------- redaction

# Deliberately broad. A false positive costs a mangled message; a false
# negative publishes somebody's phone number into a vector store.
_PHONE = re.compile(
    r"(?:\+\d{1,3}[\s\-]?)?(?:\(?\d{2,5}\)?[\s\-]?){1,3}\d{4,}"
)

# Lines WhatsApp writes itself, or that carry no content worth retrieving.
_SYSTEM = re.compile(
    r"(messages and calls are end-to-end encrypted"
    r"|<media omitted>|image omitted|video omitted|audio omitted|sticker omitted"
    r"|document omitted|this message was deleted|you deleted this message"
    r"|joined using this group's invite link"
    r"|was added|was removed|left$|changed the subject"
    r"|changed this group's icon|changed the group description"
    r"|created group|security code changed|missed voice call|missed video call)",
    re.I,
)


def _looks_like_phone(name: str) -> bool:
    """True when a sender name is really a phone number rather than a name."""
    stripped = re.sub(r"[\s\-\(\)\+]", "", name)
    return stripped.isdigit() and len(stripped) >= 7


def _redact_inline(text: str) -> tuple[str, int]:
    """Remove phone numbers from inside a message body.

    Guards against the obvious false positive: a bare four-digit year, a room
    number or a time is not a phone number, and mangling "exam at 1030 in 204"
    would make the retrieved text wrong.
    """
    count = 0

    def sub(m: re.Match) -> str:
        nonlocal count
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) < 9:          # too short to be a phone number
            return m.group(0)
        count += 1
        return "[number removed]"

    return _PHONE.sub(sub, text), count


# --------------------------------------------------------------- loading

# WhatsAppChatLoader emits a single Document whose body is a run of
# "Sender on date: message" lines. Parsing that back is the price of using the
# library loader rather than hand-rolling a parser - and the library loader is
# worth the price, because it already handles the several date formats WhatsApp
# exports in across locales.
_LOADED_LINE = re.compile(r"^(?P<sender>.+?) on (?P<date>.+?): (?P<text>.*)$", re.S)

_ANNOUNCEMENT = re.compile(
    r"\b(exam|exams|examination|viva|practical|lab|submission|submit|deadline|due"
    r"|assignment|project|presentation|seminar|test|quiz|unit test|internal"
    r"|syllabus|timetable|time table|schedule|reschedul|postpon|prepon|cancel"
    r"|holiday|leave|attendance|room|venue|hall|result|marks|fee|last date"
    r"|tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday)\b",
    re.I,
)


def load_class_chat(file_bytes: bytes, filename: str = "chat.txt") -> ChatLoadResult:
    """Load an exported WhatsApp group chat, redacted and ready to index."""
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    try:
        tmp.write(file_bytes)
        tmp.close()
        docs = WhatsAppChatLoader(tmp.name).load()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    body = "\n\n".join(d.page_content or "" for d in docs)

    messages: list[ChatMessage] = []
    pseudonyms: "OrderedDict[str, str]" = OrderedDict()
    dropped = 0
    inline_redacted = 0

    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        m = _LOADED_LINE.match(block)
        if not m:
            dropped += 1
            continue

        text = m.group("text").strip()
        if not text or _SYSTEM.search(text):
            dropped += 1
            continue

        sender = m.group("sender").strip()
        if _looks_like_phone(sender):
            if sender not in pseudonyms:
                pseudonyms[sender] = f"Member {len(pseudonyms) + 1}"
            sender = pseudonyms[sender]

        text, n = _redact_inline(text)
        inline_redacted += n

        messages.append(ChatMessage(date=m.group("date").strip(), sender=sender, text=text))

    announcements = [msg for msg in messages if _ANNOUNCEMENT.search(msg.text)]
    senders = list(dict.fromkeys(msg.sender for msg in messages))
    days = len({msg.date for msg in messages})

    notes: list[str] = []
    if pseudonyms:
        notes.append(
            f"{len(pseudonyms)} sender(s) appeared as phone numbers and were replaced "
            f"with pseudonyms. No number was indexed."
        )
    if inline_redacted:
        notes.append(f"{inline_redacted} phone number(s) were removed from message text.")
    if dropped:
        notes.append(f"{dropped} system line(s), media placeholder(s) and join notices were skipped.")

    return ChatLoadResult(
        messages=messages,
        announcements=announcements,
        senders=senders,
        days=days,
        numbers_redacted=len(pseudonyms),
        inline_numbers_redacted=inline_redacted,
        dropped=dropped,
        notes=notes,
    )


# --------------------------------------------------------------- indexing

def build_chat_index(result: ChatLoadResult) -> tuple[Chroma, int]:
    """Embed a loaded chat into its own Chroma collection.

    Two granularities, for the same reason the transcript index keeps one
    Document per course: a fixed character split cuts a conversation in the
    middle and separates "the exam is on" from the date that answers it.

      - One Document per day, so "what happened last week" has something to
        retrieve and the surrounding conversation stays with the message.
      - One Document per announcement, so "when is the DBMS viva" matches a
        single decisive line rather than a day of chatter it is buried in.
    """
    docs: list[Document] = []

    by_day: "OrderedDict[str, list[ChatMessage]]" = OrderedDict()
    for msg in result.messages:
        by_day.setdefault(msg.date, []).append(msg)

    for date, msgs in by_day.items():
        docs.append(Document(
            page_content=f"Class group messages on {date}:\n"
                         + "\n".join(f"{m.sender}: {m.text}" for m in msgs),
            metadata={"source": "class_chat", "kind": "day", "date": date},
        ))

    for msg in result.announcements:
        docs.append(Document(
            page_content=msg.as_text(),
            metadata={"source": "class_chat", "kind": "announcement",
                      "date": msg.date, "sender": msg.sender},
        ))

    if not docs:
        raise ValueError(
            "No readable messages were found. Export the chat from WhatsApp with "
            "'Without media' and upload the .txt file it produces."
        )

    store = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name="classchat_" + uuid.uuid4().hex[:8],
    )
    return store, len(docs)


def get_chat_retriever(store: Chroma):
    return store.as_retriever(search_kwargs={"k": 4})
