# 🧠 Attack on Reflexion — Adversarial Robustness of Self-Reflecting Multi-Agent Systems

> **MTP Research Project** — Investigating the robustness of Reflexion-augmented multi-agent LLM systems against adversarial attacks.

## 📖 Overview

This project implements the **Reflexion Algorithm** (Shinn et al., 2023) on top of a **CrewAI-based multi-agent trip planner** and studies its robustness against adversarial manipulation. The system uses multiple collaborating LLM agents — each with a distinct role — that plan a travel itinerary, self-evaluate using a hybrid evaluator, and iteratively improve via self-reflection stored in a persistent vector memory.

The long-term goal is to **attack** this self-improvement loop: can adversarial inputs poison the reflection memory, mislead the evaluator, or degrade the agent pipeline — and can the system defend itself?

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Trip Planner Crew                      │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ City Selector │──│ Local Expert │──│ Travel         │  │
│  │    Agent      │  │    Agent     │  │ Concierge      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬─────────┘  │
│         │                 │                  │            │
│    Search Tool       Search Tool        Search Tool      │
│    Browser Tool      Browser Tool       Browser Tool     │
│                                         Calculator       │
└──────────────────────┬───────────────────────────────────┘
                       │ Output
                       ▼
              ┌────────────────┐
              │   Evaluator    │
              │ (Heuristic +   │
              │  LLM Judge)    │
              └───────┬────────┘
                      │ Eval Result
                      ▼
              ┌────────────────┐
              │   Reflexion    │
              │   Memory (M_sr)│        ◄── ChromaDB
              │   Self-Reflect │            (Persistent)
              └───────┬────────┘
                      │ Lessons
                      ▼
               Next Trial (retry with reflections injected)
```

### Agents

| Agent | Role | Tools |
|-------|------|-------|
| **City Selection Expert** | Analyze weather, costs, and events to select the best destination | `SearchTools`, `BrowserTools` |
| **Local Expert** | Provide in-depth local knowledge, hidden gems, cultural insights | `SearchTools`, `BrowserTools` |
| **Travel Concierge** | Build a detailed day-by-day itinerary with budget and packing list | `SearchTools`, `BrowserTools`, `CalculatorTools` |

### Tools

- **SearchTools** (`tools/search_tools.py`) — LLM-powered internet-style knowledge retrieval
- **BrowserTools** (`tools/browser_tools.py`) — Website scraping and summarization using `unstructured`
- **CalculatorTools** (`tools/calculator_tools.py`) — Safe mathematical expression evaluation via AST parsing

---

## 🔄 Reflexion Loop (Algorithm 1)

Based on [Shinn et al., 2023 — *"Reflexion: Language Agents with Verbal Reinforcement Learning"*](https://arxiv.org/abs/2303.11366):

1. **Trial 0**: Run the multi-agent crew to produce an itinerary
2. **Evaluate**: Score the output using a hybrid evaluator (heuristic checks + LLM judge)
3. **Reflect**: If the trial failed, a self-reflection model (`M_sr`) analyzes root causes and generates actionable lessons
4. **Store**: Reflections are persisted in **ChromaDB** (vector similarity) for retrieval in future trials
5. **Retry**: Inject past reflections as context into the next trial's agent prompts
6. **Repeat** until the output passes all checks or max retries are exhausted

### Reflexion Memory (`reflexion_memory.py`)

- **Embedding**: `nomic-embed-text` via Ollama with retry logic and exponential backoff
- **Storage**: ChromaDB persistent client with similarity-based retrieval
- **Session Memory**: In-memory buffer of current-session reflections for multi-trial loops
- **LLM Backbone**: `qwen2.5:7b` via Ollama (fully local, no API keys required)

---

## 📊 Evaluation System (`evaluator.py`)

A two-layer evaluator combining deterministic checks with LLM-based judgment:

### Heuristic Checks (Deterministic)
| Check | Description |
|-------|-------------|
| `budget_ok` | Extracts budget constraints from input and verifies output totals don't exceed them |
| `days_ok` | Parses expected trip length and counts day markers in output (±1 tolerance) |
| `no_duplicates` | Detects repeated venues across different days using NLP-based venue extraction |

### LLM Judge (Semantic)
| Check | Description |
|-------|-------------|
| `constraints_ok` | Verifies all stated constraints (dietary, mobility, dates, origin) are honored |
| `feasible` | Checks for realistic scheduling (no impossible same-day cross-country travel) |
| `no_hallucination` | Validates that named venues/restaurants/hotels are plausible for the stated city |

The judge uses **llama3.1:8b** (intentionally different from the agent model to avoid self-grading bias).

---

## 🧪 Benchmarking

### Reflexion-Only Benchmark (`benchmark_reflexion_only.py`)

Runs each test case through the Reflexion loop with up to 3 retry trials:

```bash
python benchmark_reflexion_only.py
```

**Outputs:**
- `benchmark_reflexion_only.csv` — Per-trial results
- `benchmark_reflexion_only.json` — Detailed JSON results
- `traces_reflexion_only/` — Full traces (crew output, eval result, reflections) for each trial

### Memory Comparison Benchmark (`benchmark_memory_comparison.py`)

Compares **memory-enabled** vs **memory-disabled** configurations across test cases:

```bash
python benchmark_memory_comparison.py
```

**Metrics tracked:**
- Pass@1 rate (eventual pass across any trial)
- Per-check pass rates
- Accuracy progression across trials
- Execution time statistics

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10–3.11
- **Ollama** running locally with the following models pulled:
  ```bash
  ollama pull qwen2.5:7b
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
  ```

### Installation

```bash
# Clone the repo
git clone https://github.com/Sethayan/Attack_on_Reflextion.git
cd Attack_on_Reflextion

# Install dependencies
pip install -e .
# or with uv
uv sync

# Copy environment file
cp .env.example .env
```

### Running the Trip Planner

```bash
python main.py
```

This runs the full pipeline: agents → evaluator → reflexion → output.

### Running Benchmarks

```bash
# Reflexion-only benchmark
python benchmark_reflexion_only.py

# Memory comparison benchmark
python benchmark_memory_comparison.py
```

---

## 📁 Project Structure

```
.
├── main.py                         # Entry point — runs the trip planner crew
├── trip_agents.py                  # Agent definitions (City Selector, Local Expert, Concierge)
├── trip_tasks.py                   # Task prompts with reflexion context injection
├── reflexion_memory.py             # Reflexion Memory system (ChromaDB + Ollama embeddings)
├── evaluator.py                    # Hybrid evaluator (heuristic + LLM judge)
├── benchmark_reflexion_only.py     # Benchmark: Reflexion Algorithm 1 trials
├── benchmark_memory_comparison.py  # Benchmark: Memory-enabled vs disabled
├── validate_judge.py               # Validation script for the LLM judge
├── tools/
│   ├── search_tools.py             # LLM-powered search tool
│   ├── browser_tools.py            # Website scraping & summarization tool
│   └── calculator_tools.py         # Safe math calculator tool
├── pyproject.toml                  # Project dependencies
└── uv.lock                        # Dependency lock file
```

---

## 🎯 Future Work — Adversarial Attacks on Reflexion

The core research goal is to **attack** the Reflexion loop and study failure modes:

### Phase 1: Memory Poisoning Attacks
- **Reflection Injection**: Craft adversarial reflections that, once stored in ChromaDB, mislead future trials into producing worse outputs
- **Embedding Space Attacks**: Manipulate embeddings so that irrelevant or harmful reflections are retrieved as "most relevant"
- **Temporal Poisoning**: Gradually degrade memory quality over successive runs

### Phase 2: Evaluator Adversarial Attacks
- **Judge Manipulation**: Craft outputs that fool the LLM judge into giving false positives (passing bad itineraries)
- **Heuristic Evasion**: Generate outputs that pass heuristic checks (correct day count, budget format) while being semantically nonsensical
- **Evaluator Disagreement**: Exploit gaps between heuristic and LLM-based evaluation

### Phase 3: Agent Pipeline Attacks
- **Prompt Injection via Tools**: Inject adversarial content through the browser/search tools that derails agent behavior
- **Inter-Agent Poisoning**: Manipulate the output of one agent to corrupt downstream agents
- **Context Window Overflow**: Flood the reflexion context with noise to dilute useful lessons

### Phase 4: Defense Mechanisms
- **Reflection Verification**: Validate stored reflections before injection
- **Memory Sanitization**: Detect and filter poisoned reflections
- **Robust Evaluation**: Ensemble evaluators with disagreement detection
- **Adversarial Training**: Harden agents against known attack vectors

---

## 📚 References

- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366). *NeurIPS 2023*.
- [CrewAI Framework](https://github.com/crewAIInc/crewAI) — Multi-agent orchestration
- [Ollama](https://ollama.ai/) — Local LLM inference
- [ChromaDB](https://www.trychroma.com/) — Vector database for reflection storage

---

## 📄 License

This project is released under the MIT License.
