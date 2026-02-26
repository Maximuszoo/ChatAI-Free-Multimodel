# ChatAI-Free-Multimodel (CAFM)

A CLI tool that orchestrates a **multi-agent debate** using local LLMs via [Ollama](https://ollama.com). Multiple AI instances discuss a topic across several rounds and converge on a synthesised consensus answer.

I built this for fun — I wanted to test how much AI agents can challenge each other to produce better, more refined answers. The adversarial dynamic (especially with the Skeptic agent) forces models to justify their reasoning rather than just agreeing.

> **Tip:** **Mixing different models** (e.g. `llama3.2` + `qwen2.5` + `mistral`) produces richer debates — each model has different reasoning styles, knowledge biases and blind spots, so the disagreements are more genuine and the final synthesis more robust.

---

## Example

```
python main.py
```

```
╭──────────────────────────────────────╮
│                                      │
│   ██████╗ █████╗ ███████╗███╗   ███╗ │
│  ██╔════╝██╔══██╗██╔════╝████╗ ████║ │
│  ██║     ███████║█████╗  ██╔████╔██║ │
│  ██║     ██╔══██║██╔══╝  ██║╚██╔╝██║ │
│  ╚██████╗██║  ██║██║     ██║ ╚═╝ ██║ │
│   ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝     ╚═╝ │
│   ChatAI-Free-Multimodel  v1.0       │
╰──────────────────────────────────────╯

                  Current Configuration
┌──────────────────┬────────────────────────────────────┐
│ Instances        │ 3                                  │
│ Rounds           │ 1                                  │
│   Agent 1 Model  │ llama3.2                           │
│   Agent 2 Model  │ qwen2.5                            │
│   Agent 3 Model  │ mistral             ⚡ SKEPTIC     │
│ Skeptic Agent    │ ON                                 │
│ Context Limit    │ 32768 tokens                        │
│ Context Strategy │ sliding_window                     │
│ Stream Output    │ True                               │
│ Save Logs        │ True                               │
└──────────────────┴────────────────────────────────────┘

✓ All models available.
System Ready.

Enter your query: If 5 shirts take 5 hours to dry, how long do 30 shirts take?

── Conclave of Experts — Debate Begins ──
  Agent 1 : llama3.2
  Agent 2 : qwen2.5
  Agent 3 ⚡ SKEPTIC : mistral
  Rounds  : 1  |  Skeptic: ON

── Round 1 / 1 ──

╭─── Agent 1 — llama3.2 — Round 1 ───╮
  5 hours. Drying is parallel, not serial.

╭─── Agent 2 — qwen2.5 — Round 1 ────╮
  All shirts dry simultaneously: 5 hours.

╭─── ⚡ SKEPTIC — mistral — Round 1 ──╮
  Assumes infinite space and identical conditions.
  What if shirts overlap or airflow is blocked?

── Final Synthesis ──
╭─── ✦ Agent 1 — llama3.2 — Final Synthesis ───╮
  Under standard conditions (enough space, same sun
  exposure), 30 shirts still take 5 hours. The Skeptic
  raises a valid caveat: overcrowding could extend drying
  time, but the canonical answer to this classic puzzle
  is 5 hours.
```

---

## How it works

| Phase | What happens |
|---|---|
| **Round 1** | Every agent gives an independent answer. The Skeptic challenges the question itself — its assumptions, ambiguities, and hidden traps. |
| **Rounds 2…N** | Models read the full transcript and must refute, clarify, or expand on each other. The Skeptic keeps attacking weak points. |
| **Final Synthesis** | One agent integrates all perspectives into the best possible consensus answer. |

Models detect the **language of your query** (Spanish or English) and respond in that language automatically.

---

## Installation

**Requirements:** Python 3.11+, [Ollama](https://ollama.com)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull models (use different ones for better diversity)
ollama pull llama3.2
ollama pull qwen2.5
ollama pull mistral

# 3. Clone and set up
git clone <repo-url> ChatAI-Free-Multimodel
cd ChatAI-Free-Multimodel
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. Run
python main.py
```

---

## Settings menu

Type `/settings` at the prompt to configure:

| # | Option |
|---|--------|
| 1 | Number of models |
| 2 | Number of rounds |
| 3 | Assign a model to each agent (pick by number from local list) |
| 4 | Context token limit |
| 5 | Context strategy: `sliding_window` or `summary` |
| 6 | Toggle streaming output |
| 7 | Toggle session logging |
| 8 | Toggle Skeptic agent ON / OFF |

All changes are saved to `config.json` immediately.

---

## Project structure

```
ChatAI-Free-Multimodel/
├── main.py
├── config.json
├── requirements.txt
├── logs/                    ← JSON session transcripts (auto-created)
└── cafm/
    ├── config_manager.py    ← Config load/save with deep merge
    ├── ollama_client.py     ← Ollama streaming + sync wrapper
    ├── context_manager.py   ← Token budget, sliding window, summary
    ├── debate_engine.py     ← Round orchestration, language detection
    └── cli.py               ← Terminal UI, menus, main loop
```

---

## License

This project is licensed under The Unlicense. See the [LICENSE](LICENSE) file for details.
