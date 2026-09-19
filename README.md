# Transcript & Study Plan Agent

**Track 3 — Education.** A student uploads their own transcript. Before they ask
anything, the agent flags weak areas and builds a prioritised study plan, then
stays available for follow-ups and matches live internships to what the
transcript actually shows they studied.

Built by Navlani Jaykrishna Satishkumar (SOU2023CSE69), Silver Oak University.

> **This Streamlit app is the submission.** The same author also built
> [SOU AI HelpDesk Pro](https://github.com/Jaykrishna25/SOU-AI-HelpDesk-Pro), a
> live help desk portal for the university
> ([deployed here](https://sou-ai-help-desk-pro-frontend.vercel.app)), which
> carries the same study-plan logic plus a fee assistant built on the same rule:
> the model never calculates, tools do, and a figure that cannot be sourced is
> not produced. That portal is Next.js rather than Python and so is **not** an
> entry to this hackathon — it is mentioned only as context for where the
> approach came from.

---

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt

ollama pull qwen3:8b            # or qwen3:4b on a machine short of RAM
streamlit run app_edu.py
```

Then click **Use sample transcript**, or upload your own CSV, Excel, PDF or text
transcript. Everything runs locally except the internship search, which calls a
public jobs feed and needs no API key.

To use a smaller model: `set OLLAMA_MODEL=qwen3:4b` before launching.

To run against a hosted model instead of locally — useful on a machine that
cannot hold an 8B model, or on a slow connection:

```bash
pip install langchain-google-genai
set LLM_PROVIDER=gemini
set GOOGLE_API_KEY=your-free-key    # aistudio.google.com/apikey
```

The pipeline is unchanged either way; only the chat model swaps. Unset
`LLM_PROVIDER` to go back to Ollama, which is the default.

---

## The required pipeline, and where each stage lives

| Stage | Implementation | File |
|---|---|---|
| LLM setup | `ChatOllama`, default `qwen3:8b` (switchable to Gemini) | `src_edu/agent.py` |
| Document loading | `PyPDFLoader` for PDF, pandas for CSV/Excel | `src_edu/ingest.py` |
| Document loading | `WhatsAppChatLoader` for the class group | `src_edu/classchat.py` |
| Text splitting | `RecursiveCharacterTextSplitter` + one Document per course | `src_edu/ingest.py` |
| Text splitting | One Document per day, plus one per announcement | `src_edu/classchat.py` |
| Embeddings | `HuggingFaceEmbeddings`, `BAAI/bge-small-en-v1.5` | `src_edu/ingest.py` |
| Vector store | `Chroma`, one collection per uploaded transcript | `src_edu/ingest.py` |
| Retriever | `as_retriever(search_kwargs={"k": 3})` | `src_edu/ingest.py` |
| Custom tools | `generate_study_plan`, `search_internships`, `search_class_chat` | `src_edu/tools.py` |
| Tool calling | `@tool` with units stated in every docstring | `src_edu/tools.py` |
| Agent | `create_agent`, ReAct style | `src_edu/agent.py` |
| Deployment | Streamlit, built around the upload | `app_edu.py` |

---

## The class WhatsApp group

At most Indian universities the real announcement channel is not the portal. It
is the class WhatsApp group: exam dates, submission deadlines, room changes and
"bring your practical file tomorrow" arrive there and are written down nowhere
else. A student scrolling back three weeks to find one message is doing
retrieval by hand — which is the job this agent should be taking.

Upload an exported chat and `search_class_chat` is added to the agent's
toolset. Ask *"when is the DBMS internal and which room?"* and it answers from
the group, citing who said it and when — including the message on 18 August
that moved the test from room 204 to room 108, which is exactly the kind of
correction that gets missed.

**The privacy problem is real and is handled, not waved away.** A WhatsApp
export is not one person's data; it is every message every member of that group
ever sent, and senders who are not in the uploader's contacts appear as raw
phone numbers. Three things follow, none optional:

1. **Phone numbers are removed before anything is embedded** — as sender names,
   which become stable pseudonyms (`Member 1`), and inside message bodies. The
   length threshold is deliberate: `room 204` and `10:00 AM` survive, a
   ten-digit number does not. Mangling the exam time would be worse than not
   building the feature.
2. **Attachments are not read.** An export contains only `<Media omitted>`
   anyway, but the intent matters — this reads text, not photographs.
3. **The file is never persisted.** It is written to a temporary path only
   because `WhatsAppChatLoader` reads from a path rather than from bytes, and
   deleted in a `finally` block.

Display names that are not phone numbers are kept, because *"what did sir say
about the viva"* is the question people actually ask and stripping names makes
the feature useless. That is a deliberate trade, stated here rather than
hidden.

`sample/class_group.txt` is a fabricated export — invented names, invented
dates — so the feature can be demonstrated without uploading a real group.

---

## The tools

### 1. `generate_study_plan` — runs automatically on ingestion

Pure Python. No model involvement. For every subject below the threshold it
computes urgency, suggested weekly hours, and the reason it was flagged.

Priority is worst score first, with credits breaking ties — a weak four-credit
subject costs more than a weak two-credit one. Hours scale with the gap to the
threshold plus credit weight, so the plan spends more time where more is at
stake rather than splitting hours evenly.

Two bands: below 60 is **weak**, below 45 is **at risk of failing**. The
threshold is a slider, so a student aiming for a distinction can raise it to 80
and see a different plan.

### 2. `search_internships` — keywords come from the transcript

Takes **no search term from the user.** Keywords are extracted from the
student's strongest subjects through a course-to-skill map, then queried against
a live public listings feed. Each result shows which keyword matched it, so the
student can see why it appeared.

Strongest-first is deliberate: a student is most employable in what they did
well, and the weak subjects are already covered by the study plan.

---

## Three design decisions

### One Document per course, not just character chunks

The brief asks for retrieval that serves both a narrow question ("what grade did
I get in databases") and a broad one ("what have I not covered yet"). A fixed
character split cuts through a course row and leaves the subject name in one
chunk and its marks in the next — so the narrow question retrieves a chunk that
looks right and answers wrong.

Each course is therefore indexed as its own Document, carrying subject, credits,
score, grade and performance band together, alongside the whole-transcript chunks
that serve the broad questions.

### The model is forbidden to calculate

Every figure comes from `compute_plan()`. The system prompt's first rule is *"You
never calculate. The tools calculate."* The model turns computed figures into
sentences; that is the job it is reliable at.

The app shows a **"raw study plan tool output"** expander, so any answer can be
checked against exactly what the model was handed.

### The plan runs on ingestion, not on request

`opening_summary()` calls the study-plan tool **directly** and asks the model only
to reword the result. The tool genuinely runs the moment ingestion finishes — it
does not depend on a small local model choosing to call it — so the plan on
screen is always correct even when the model is unavailable.

---

## What it refuses to do

- **Predict a grade.** No forecast, no pass probability. Nothing produces one.
- **Invent a figure.** If a tool returns nothing, the answer says so.
- **Pretend a job feed outage is a verdict on the student.** When listings fail
  to load it says the feed could not be reached, explicitly.
- **Guess at an unreadable transcript.** Rows that cannot be parsed are skipped,
  and no subject is fabricated to fill the gap.

---

## Sample data

`sample/transcript.csv` — 15 subjects across semesters 3 to 5 of a B.Tech CSE
programme, with a realistic spread: three subjects below 45, several strong ones,
and one borderline. Invented for the demo; no real student record is used.

---

## Known limits

- PDF transcripts are parsed with a regex against `CODE SUBJECT MARKS GRADE`
  rows; layouts far from that fall back to whatever text extraction returns.
- The course-to-skill map is hand-built and covers common CSE subjects.
- The listings feed is remote-work oriented, so results skew that way.
- `qwen3:0.6b` is too small for reliable tool calling — measured, not assumed.
  It emitted a raw tool-call string as its answer. Default is `qwen3:4b`.
