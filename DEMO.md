# Three-minute demo

App running, sample transcript **not** yet loaded. Do not load it live — the
first embedding-model load takes 30–90 seconds. Start the app, load the sample
once, then restart the browser tab so the empty state is showing but the model is
warm.

---

## 0:00 — The problem (20s)

> "A transcript tells a student their marks. It does not tell them what to do
> about them. Working that out means ranking subjects by how far below the line
> they are, weighting by credits, and turning that into hours. That's arithmetic,
> and students shouldn't have to do it to find out what to revise."

## 0:20 — Upload drives it (30s)

Click **Use sample transcript**. Let the status box step through.

> "I haven't asked anything. The plan already ran."

Point at: 15 subjects, average 63, four weak, three at risk of failing.

## 0:50 — Required query one: retrieval (30s)

Ask: **"What grade did I get in databases?"**

> "That's retrieval. The transcript is indexed one course per document, so this
> query returns the DBMS row — subject, marks and grade together."

Open **Raw study plan tool output**.

> "That's exactly what the model was handed. You can check any answer against
> its own working."

## 1:20 — Required query two: a tool call (40s)

Ask: **"Find internships that match my transcript."**

> "I gave it no search term. The keywords come from my strongest subjects —
> software engineering, web technologies, cloud — through a course-to-skill map,
> then a live listings feed. Each result shows which keyword matched it."

## Optional 30s — the class group (use if you have time, or if asked "what else?")

Upload `sample/class_group.txt` in the sidebar. Ask:

> when is the DBMS internal and which room?

It answers **room 108**, and says the room was changed from 204 on 18 August.

> "At this university the real announcement channel isn't the portal, it's the
> class WhatsApp group. Exam dates, deadlines, room changes — they're there and
> nowhere else, and finding one three weeks later means scrolling by hand.
>
> Notice what it actually did. Two messages mention this test. The first says
> room 204, a later one moves it to 108. It took the later one and told me it
> had changed — which is the difference between retrieval and a search box.
>
> And the privacy: an export isn't one person's data, it's sixty people's.
> Phone numbers are stripped before anything is embedded — senders become
> 'Member 1' — attachments aren't read, and the file is never saved. The room
> number survives; a ten-digit number doesn't."

Open **The exact messages this came from** to show the source lines.

*(This sample file is fabricated — invented names and dates. Never demo with a
real group chat.)*

## 2:00 — The design decision (45s)

Move the sidebar slider from 60 to 80.

> "The deliberate decision: the model never calculates. Everything you see is
> Python — priority is worst score first with credits breaking ties, and hours
> scale with the gap to the line, so it spends more time where more is at stake.
> The model only turns those numbers into sentences.
>
> That matters here because a student acts on this. A plan that's confidently
> wrong about which subject is most urgent sends them to revise the wrong thing.
> And notice it never predicts a grade — nothing in the system produces one, so
> it doesn't pretend to."

## 2:45 — Close (15s)

> "Runs entirely locally apart from the job feed. The transcript is embedded into
> a collection scoped to that one upload and nothing persists between sessions,
> because a transcript is personal data."

---

## Questions to expect

**"Isn't uploading a class group chat a privacy problem?"**
Yes, and it is handled rather than waved away. Phone numbers are removed before
anything is embedded, both as sender names and inside message bodies; the length
threshold keeps "room 204" and drops a ten-digit number. Attachments aren't
read. The file goes to a temporary path only because `WhatsAppChatLoader` reads
from a path rather than bytes, and it is deleted in a `finally` block. Display
names are kept, deliberately, because "what did sir say" is the question people
actually ask — that trade is stated in the README rather than hidden.

**"Why one document per course?"**
A fixed character split separates a subject name from its marks. The narrow
question then retrieves something that looks right and answers wrong.

**"What if the transcript is a PDF?"**
`PyPDFLoader`, then a regex for `CODE SUBJECT MARKS GRADE` rows. Layouts far from
that degrade to plain text extraction — a known limit, in the README.

**"Could it predict my final grade?"**
No, and it's built to refuse. Nothing in the system produces a forecast, so
offering one would mean inventing it.

**"Why not a bigger model?"**
It runs locally on a student laptop. `qwen3:0.6b` was measurably unreliable at
tool calling, so the default is `qwen3:4b`, overridable by environment variable.

**"What happens if the job feed is down?"**
It says the feed could not be reached, and says explicitly that this is a feed
problem rather than a statement about the student's prospects.

---

## Before you start

```powershell
cd C:\dev\hackathon
.\.venv\Scripts\Activate.ps1
streamlit run app_edu.py
```

Default model is `qwen3:8b` via Ollama — pull it before demo day, it is 5.2 GB.
If the machine is short of memory, `$env:OLLAMA_MODEL="qwen3:4b"` still works.

## If something breaks

- **Model 404** → the model isn't pulled. `ollama pull qwen3:4b`.
- **No listings** → say the feed is remote-work oriented and occasionally slow;
  the study plan is unaffected.
- **Slow first answer** → it's making a tool round trip; say what it's doing.

Never debug live. Move on and mention it afterwards.
