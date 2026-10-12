# Valorant Aim Coach v2 — AI Coach + Aim Lab

A personalized, RAG-grounded Valorant coach that runs entirely on your own
machine. Ask it a question by voice, it retrieves the relevant coaching notes,
tailors the answer to your rank, agent and playstyle, and speaks the answer
back.

Built for an Intel i7 + RTX 4050 Laptop (6 GB VRAM). Nothing leaves the machine.

---

## What's new in v2

Version 1 was a single-file aim toolkit (`app/web/aim.html` now). Version 2
puts it inside a real AI coach:

- **AI strategy coach** — ask anything about maps, agents, economy, post-plant
  or ranking up; answers come from a local LLM grounded in a coaching
  knowledge base, in a casual teammate-on-comms voice.
- **Aim Lab tab** — every v1 tool (sens converter, crosshair builder, target
  trainer, sens game, sensitivity finder, 3D aim range) opens inside the
  coach, pre-filled with your saved DPI, sens and rank.
- **One profile for both** — rank, role, playstyle, agents, maps, strengths,
  weaknesses and mouse setup (DPI, sens, live eDPI verdict) are picked from
  clickable lists. A Sensitivity Finder result saves straight into it, and
  the AI coaches you with it.
- **Aim knowledge** — eDPI ranges, overshoot/undershoot fixes, crosshair
  placement and a daily routine, so aim questions get grounded answers.
- **Full agent and map roster**, including Waylay, Tejo, Veto and Summit.

Run it with the Quick start below, then open <http://127.0.0.1:8000/>
(or <http://127.0.0.1:8000/#aim> to land on the Aim Lab).

---

## What works today

| Component | Status |
|---|---|
| FastAPI backend, streaming + buffered chat | working |
| Ollama / Llama 3.2 3B local inference | working, 75 tok/s |
| Hybrid RAG (Chroma + BM25 + profile re-ranking) | working, 100% recall@3 |
| Knowledge base, 37 documents / 211 chunks | working |
| Aim Lab (v1 aim tools, profile-synced) | working |
| User profiles (SQLite + SQLAlchemy + Alembic) | working |
| Valorant-only domain gate | working |
| Browser UI at `/` | working |
| Text CLI client | working |
| Voice client (Whisper + Piper) | code complete, needs Piper download |

---

## Quick start

```bash
# 1. install
uv venv --python 3.11
uv pip install -e ".[dev]"

# 2. model (Ollama must be running)
ollama pull llama3.2:3b-instruct-q4_K_M

# 3. config
cp .env.example .env

# 4. database
uv run alembic upgrade head

# 5. knowledge base -> vector store
uv run python -m app.rag.ingest --reset

# 6. run
uv run uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000/> in a browser. That serves a small chat UI
with a profile panel on the left, streamed replies, and an optional view of
which notes were retrieved.

Prefer a terminal? Same server, same endpoints:

```bash
uv run python clients/cli_chat.py --setup    # tell it your rank and agents
uv run python clients/cli_chat.py            # chat
```

Health at <http://127.0.0.1:8000/health>, API docs at
<http://127.0.0.1:8000/docs>.

---

## Architecture

```
   mic ──► VAD ──► Whisper ──┐                    (client process)
                             │
 speaker ◄── Piper ◄── sentence splitter ◄──┐
                                            │  text over HTTP
                                            │
┌───────────────────────────────────────────▼──────────────────────┐
│ FastAPI                                                          │
│   POST /chat          buffered reply + sources + timings         │
│   POST /chat/stream   token stream (this is what voice uses)      │
│   GET/PUT /profile/{user}                                        │
│   GET /health   POST /kb/reindex                                 │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│ CoachService  (app/core/coach.py)                                │
│   1 domain gate    reject non-Valorant before spending a token    │
│   2 load profile   rank, agents, role, playstyle, weaknesses      │
│   3 query context  extract map / side / site / agents / rank band │
│   4 retrieve       hybrid search, profile-aware re-ranking        │
│   5 build prompt   system + profile + notes + history + question  │
│   6 generate       streamed from Ollama                           │
│   7 persist        transcript for short-term memory               │
└──┬─────────────┬──────────────────┬────────────────┬─────────────┘
   │             │                  │                │
 SQLite      Chroma + BM25      Ollama            Whisper/Piper
 profiles    202 chunks         Llama 3.2 3B      (in the client)
```

**Speech runs in the client, not the server.** Audio never crosses the
network, the API stays a simple text service, and the sentence-to-speech loop
sits next to the audio device, which is what keeps latency low.

### The latency budget

A voice assistant feels broken above about 1.2 s from "you stop talking" to
"it starts talking". Measured on this machine:

| Stage | Time |
|---|---|
| Whisper `small.en` int8 on CPU | ~300 ms |
| Hybrid retrieval (202 chunks) | ~70 ms |
| Ollama time-to-first-token | ~34 ms |
| First sentence complete | ~400 ms |
| Piper synthesis of that sentence | ~150 ms |

The design rule that makes this work: **never wait for the full LLM response**.
`app/speech/streaming.py` splits the token stream on sentence boundaries so
sentence one is spoken while sentence two is still being generated.

---

## Model choice on 6 GB VRAM

Windows takes roughly 0.8 GB of the 6 GB, leaving about 5.0 GB usable. Both
models below were measured on this machine, not estimated:

| Model | Loaded | GPU | Answer time | Verdict |
|---|---|---|---|---|
| `llama3.2:3b-instruct-q4_K_M` | 2.6 GB | **100%** | 1.8 s | **recommended** — 75 tok/s, TTFT 34 ms |
| `llama3.1:8b-instruct-q4_K_M` | 5.6 GB | 78% | 4.2-5.5 s | does not fit; spills to CPU |

**The 8B does not fit, even at a 2048 context.** Forcing `num_ctx 2048` via a
Modelfile still left it at 22% CPU / 78% GPU, and answers took 4-5 s — which
breaks the voice loop. Its answers are noticeably better reasoned and more
consistently formatted than the 3B's, so the quality gap is real; it just
cannot be bought on this card at Q4.

If you want to chase that quality, the next thing to try is a smaller
quantization of the 8B (`q3_K_M`, roughly 4.0 GB), which should fit fully.
Measure it before trusting it — Q3 degrades quality unevenly:

```bash
ollama pull llama3.1:8b-instruct-q3_K_M
uv run python scripts/bench_llm.py --model llama3.1:8b-instruct-q3_K_M
uv run python scripts/compare_models.py --b llama3.1:8b-instruct-q3_K_M
ollama ps    # must read 100% GPU
```

### Context length gotcha

`LLM_NUM_CTX` in `.env` is **informational only**. Ollama's OpenAI-compatible
endpoint silently discards a per-request `options` block, so sending `num_ctx`
looks like it works while changing nothing — `ollama ps` kept reporting the
default. Context length is fixed when the model loads. To change it, create a
variant:

```powershell
'FROM llama3.2:3b-instruct-q4_K_M'  | Out-File -Encoding ascii Modelfile
'PARAMETER num_ctx 8192'            | Out-File -Encoding ascii -Append Modelfile
ollama create valcoach:ctx8192 -f Modelfile
```

then point `LLM_MODEL` at `valcoach:ctx8192`.

### Keep the model warm

Avoids a ~2 s cold load on the first question of each session:

```powershell
setx OLLAMA_KEEP_ALIVE 60m
```

Or just run `scripts/setup_ollama.ps1`, which does the pull, the keep-alive and
the GPU check in one pass.

---

## Project layout

```
app/
  config.py            every tunable knob, sourced from .env
  main.py              FastAPI app + warm-up on startup
  api/
    deps.py            dependency wiring (swap implementations here)
    routes/            chat.py, profile.py, admin.py, web.py
  core/
    coach.py           the orchestrator, the 7 steps above
    prompt.py          system prompt + profile/context rendering
    domain_gate.py     Valorant-only enforcement
    filters.py         question -> retrieval hints
    vocab.py           maps, agents, roles, ranks, game terms
  llm/
    base.py            LLMClient protocol
    ollama_client.py   OpenAI-compatible streaming client
  rag/
    chunker.py         markdown + frontmatter -> chunks
    embedder.py        fastembed / bge-small, CPU only
    chroma_store.py    Chroma implementation of VectorStore
    retriever.py       hybrid search + fusion + re-ranking
    ingest.py          CLI: data/kb -> vector store
  db/
    models.py          User, Profile, Message
    repositories.py    ProfileRepo, ConversationRepo
  web/
    index.html         the browser UI (single file, no build step)
  speech/
    stt.py             faster-whisper
    tts.py             Piper subprocess
    vad.py             energy-based endpointing
    streaming.py       token stream -> speakable sentences
data/kb/               THE KNOWLEDGE BASE - edit these to improve answers
clients/               cli_chat.py, cli_voice.py
scripts/               bench_llm.py, eval_rag.py, smoke_test.py, compare_models.py
tests/                 unit tests + tests/eval/questions.yaml
```

---

## The knowledge base is the product

Answer quality is a content problem far more than a model problem. The model
is instructed to use only what retrieval gives it, so **if a fact is not in
`data/kb/`, the coach will not say it.**

Each document is markdown with YAML frontmatter. The frontmatter is what makes
personalization work — it is what retrieval filters and boosts on:

```markdown
---
id: ascent-attack-strategies
title: Ascent Attack Strategies
map: ascent            # hard filter: an Ascent question never gets Bind notes
side: attack           # hard filter: attack questions never get defense notes
site: A
topic: [execute, entry]
agents: [jett, omen, sova]   # soft boost toward the player's main agents
roles: [duelist]
rank_band: [low, mid, high]  # soft boost toward the player's rank
playstyle: [aggressive]      # soft boost toward how they play
---

## A site execute

Smoke Heaven and Hell, plus a smoke or wall on CT...
```

Split content on `## ` headings — one heading is one self-contained coaching
idea, and that is exactly the unit that gets retrieved.

After editing anything under `data/kb/`:

```bash
uv run python -m app.rag.ingest --reset
uv run python scripts/eval_rag.py        # confirm you did not make it worse
```

---

## Why these choices

**Hybrid retrieval, not pure embeddings.** This domain is almost entirely
proper nouns — "Heaven", "Catwalk", "A Main", "Chamber", "Killjoy". Dense
vectors blur exactly those distinctions; a Chamber question happily retrieves
Killjoy notes because both are sentinels. BM25 anchors on the literal token,
the embedder handles paraphrase, and reciprocal-rank fusion combines them
without the two score scales needing to be comparable.

**Embeddings on CPU.** bge-small via ONNX embeds a query in ~15 ms on the i7
and uses zero VRAM, which is what lets the LLM stay fully GPU-resident.

**No agent framework.** The MVP is a deterministic pipeline — retrieve, build
prompt, generate — with no branching decision for an agent loop to make. A
framework would add abstraction and hide the prompt, which is the thing most
worth iterating on. Pydantic AI becomes worthwhile in Phase 7, alongside real
tools and a larger model.

**A code-level domain gate.** A 3B model does not reliably hold a refusal
instruction, and rejecting off-topic input before the LLM call saves a full
generation. The gate is deliberately biased toward recall: wrongly refusing a
real coaching question is far worse than answering a borderline one, and the
system prompt is the second line of defence.

**Everything behind an interface.** `LLMClient`, `VectorStore`. Swapping
Ollama for a hosted API, or Chroma for pgvector, is an edit in
`app/api/deps.py` and one implementation file.

---

## Development workflow

```bash
uv run pytest                            # 36 unit tests, no server needed
uv run python scripts/smoke_test.py      # full pipeline in-process, no server
uv run python scripts/eval_rag.py        # retrieval quality vs golden set
uv run python scripts/bench_llm.py       # TTFT and tokens/sec
uv run ruff check app clients scripts    # lint
```

`smoke_test.py` is the fastest loop when changing prompts or retrieval — it
exercises the whole pipeline without starting a server.

### Current measurements

```
retrieval  : recall@1 87%,  recall@3 100%   (31 golden questions)
llm        : TTFT 34 ms,    75.3 tok/s      (Llama 3.2 3B, 1000-word context)
end-to-end : first speakable sentence 0.88 s after the request
tests      : 43 passed, ruff clean
```

Adding Whisper (~300 ms) and Piper (~150 ms) to that 0.88 s puts the full
spoken loop at roughly 1.3 s, which is inside the 1.5 s target.

### Known limitation

The 3B follows the answer format inconsistently — it sometimes returns two
bullets instead of the requested three or four, and occasionally folds the
"Watch out" line into a bullet. Content is well grounded (retrieval quality is
doing the heavy lifting), but formatting varies run to run. This is a model
capability limit, not a prompt bug; see the model comparison above.

---

## Enabling voice

Speech dependencies are a separate extra so the text MVP stays light:

```bash
uv pip install -e ".[voice]"
```

**Whisper** downloads automatically on first use.

**Piper** must be fetched manually:

1. Download `piper_windows_amd64.zip` from the Piper releases page, unzip to
   `data/voices/piper/`
2. Download a voice — `en_US-ryan-high.onnx` **and** `en_US-ryan-high.onnx.json`
   — into `data/voices/`
3. Confirm `PIPER_EXE` and `PIPER_VOICE` in `.env` point at them

```bash
uv run python clients/cli_voice.py           # hands-free, VAD endpointing
uv run python clients/cli_voice.py --push    # push-to-talk
uv run python clients/cli_voice.py --no-tts  # transcribe only, no speech out
```

The voice client runs without Piper — it just prints instead of speaking.

---

## Configuration

All settings live in `.env`. The ones worth knowing:

| Key | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `llama3.2:3b-instruct-q4_K_M` | swap to compare models |
| `LLM_NUM_CTX` | `4096` | drop to `2048` for the 8B |
| `LLM_TEMPERATURE` | `0.25` | low, to limit invented callouts |
| `LLM_MAX_TOKENS` | `400` | answers are spoken; long is unusable |
| `RETRIEVAL_TOP_K` | `8` | chunks fed to the prompt |
| `HYBRID_ALPHA` | `0.55` | 1.0 = pure vector, 0.0 = pure BM25 |
| `DOMAIN_GATE_ENABLED` | `true` | set false to debug refusals |
| `DATABASE_URL` | `sqlite:///./data/app.db` | swap for a PostgreSQL URL |
| `STT_MODEL` | `small.en` | `base.en` is faster, less accurate |

---

## Roadmap

Done: environment, text coach, profiles, RAG, evaluation harness, voice
components.

Next:

1. **Model bake-off** — `compare_models.py` against 3B and 8B, pick with data.
2. **Grow the knowledge base** — the remaining maps and agents. This is the
   highest-value work by a wide margin.
3. **Finish the voice loop** — install Piper, measure real end-to-end latency.
4. **Conversation memory** — summarize older turns instead of dropping them.
5. **Web UI** — browser mic + WebSocket, replacing the CLI client.
6. **PostgreSQL + pgvector**, multi-user auth, match-history import.

---

## Troubleshooting

**`llm.unreachable` on startup** — Ollama is not running. Start it with
`ollama serve`, confirm with `ollama list`.

**Answers are slow (under 20 tok/s)** — the model is partly on CPU. Check
`ollama ps` says `100% GPU`. Lower `LLM_NUM_CTX`, or use a smaller model.

**`kb.empty` warning** — run `uv run python -m app.rag.ingest --reset`.

**A real question gets refused** — add the missing vocabulary to
`GAME_TERMS` in `app/core/vocab.py`, then add the question to
`tests/test_domain_gate.py` so it stays fixed.

**Answers ignore the notes** — check what was actually retrieved with
`clients/cli_chat.py --sources`. If the right note is missing, it is a
retrieval problem, and no amount of prompt tuning will fix it.
