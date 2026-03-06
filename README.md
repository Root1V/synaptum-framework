# Synaptum

**A minimal, bus-driven agent framework for building LLM-powered single-agents and multi-agent systems in Python.**

Synaptum sits between your LLM SDK and your agent platform. It gives you the minimal primitives — agents, a message bus, a runtime, and prompt management — to build reliable, testable agent applications without opinion on your infrastructure.

> **Version:** 0.4.0 · **Python:** ≥ 3.13 · **License:** MIT

---

## Why Synaptum?

| Problem | Synaptum's answer |
|---|---|
| LLM frameworks force you into their cloud/SDK | Bring your own LLM client via `LLMClient` ABC |
| Agents are hard to test in isolation | Pure async handlers, no hidden threads or global state |
| Prompt strings scattered across code | Versioned `PromptTemplate` + `PromptProvider` system |
| Scaling to multi-agent requires a rewrite | Message bus decouples every agent from the start |
| Vendor lock-in in observability | Pluggable; integrates with Langfuse, OpenTelemetry, etc. |

---

## Architecture

### C4 Level 1 — System Context

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             External World                               │
│                                                                          │
│   ┌──────────────┐                                                       │
│   │   Developer  │                                                       │
│   │  / Platform  │                                                       │
│   │  Application │                                                       │
│   └──────┬───────┘                                                       │
│          │ uses                                                          │
│          ▼                                                               │
│   ┌──────────────────────────────────────────┐                           │
│   │              S Y N A P T U M             │                           │
│   │                                          │                           │
│   │  Minimal bus-driven agent framework      │                           │
│   │  for LLM-powered agent systems           │                           │
│   └──────┬───────────────────────────────────┘                           │
│          │                           │                                   │
│          │ reads prompts from        │ calls                             │
│          ▼                           ▼                                   │
│   ┌──────────────┐          ┌─────────────────────┐                      │
│   │  Prompt Store│          │   Axonium SDK       │                      │
│   │              │          │                     │                      │
│   │  YAML / DB / │          │  LLM adapter layer  │                      │
│   │  Remote API  │          └──────────┬──────────┘                      │
│   └──────────────┘                     │ invokes                         │
│                                        ▼                                 │
│                            ┌─────────────────────┐                       │
│                            │    LLM Server       │                       │
│                            │                     │                       │
│                            │  llama.cpp / vLLM / │                       │
│                            │  OpenAI-compatible  │                       │
│                            └─────────────────────┘                       │
└──────────────────────────────────────────────────────────────────────────┘
```

### C4 Level 2 — Container Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Synaptum Framework                               │
│                                                                             │
│  ┌──────────────────────┐      ┌───────────────────────────────────────┐    │
│  │     AgentRuntime     │      │           Message Bus                 │    │
│  │                      │      │                                       │    │
│  │  - register(agent)   │─────▶│  publish(msg) ──▶ queue               │    │
│  │  - start(run_id)     │      │  subscribe(name, handler)             │    │
│  │  - run_until_idle()  │◀─────│  deliver(msg) ──▶ handlers            │    │
│  │  - prompts provider  │      │                                       │    │
│  └──────────┬───────────┘      └───────────────────────────────────────┘    │
│             │ injects                          ▲                            │
│             │ PromptProvider                   │ messages                   │
│             ▼                                  │                            │
│  ┌──────────────────────┐      ┌───────────────┴───────────────────────┐    │
│  │   Prompt System      │      │              Agents                   │    │
│  │                      │      │                                       │    │
│  │  PromptRegistry      │      │  MessageAgent   LLMAgent              │    │
│  │  FilePromptProvider  │      │  ┌────────────┐ ┌──────────────────┐  │    │
│  │  InMemoryProvider    │      │  │ name       │ │ name             │  │    │
│  │                      │      │  │ _ref       │ │ _ref (inherited) │  │    │
│  │  PromptTemplate      │      │  │ on_message │ │ think()          │  │    │
│  │  - content           │      │  └─────┬──────┘ │ output_model     │  │    │
│  │  - version           │      │        │        └────────┬─────────┘  │    │
│  │  - variables         │      │        └────────────┬────┘            │    │
│  └──────────────────────┘      │               CompositeAgent          │    │
│                                │       (ConsensusAgent, HITLAgent,     │    │
│  ┌──────────────────────┐      │        SagaAgent — auto-register      │    │
│  │    LLM Layer         │◀─────┤        their sub-agent topology)      │    │
│  │                      │      └───────────────────────────────────────┘    │
│  │  LLMClient (ABC)     │                                                   │
│  │  LlamaClient         │      ┌───────────────────────────────────────┐    │
│  │  (axonium SDK)       │      │   Patterns                            │    │
│  └──────────────────────┘      │                                       │    │
│                                │  MapReduceAgent   PlanAndExecuteAgent │    │
│                                │  SwarmAgent       ReflectionAgent     │    │
│                                │  ConsensusAgent   HITLAgent           │    │
│                                │  SagaAgent        RouterPattern       │    │
│                                │  SupervisorPattern  GraphBuilder      │    │
│                                └───────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### Agent
An agent is the fundamental unit. It has a `name` (used for routing), an internal `id` (UUID), and reacts to messages asynchronously.

```python
class Agent(ABC):
    name: str   # routing address — "calculator", "planner"
    id: str     # internal UUID — immutable, system-generated

    async def on_message(self, message: Message, context: AgentContext): ...
```

### Message Bus
All communication is message-driven. Agents never call each other directly — they publish messages to the bus and the runtime delivers them.

```python
await runtime.publish(Message(
    sender="planner",
    recipient="executor",
    type="user.input",
    payload={"text": "Summarize this document"},
    reply_to="planner",
))
```

### Prompt Management
Prompts are first-class citizens, versioned and decoupled from agent code.

```python
# Configure once at application level
runtime = AgentRuntime(bus, prompts=FilePromptProvider("prompts/agents.yaml"))

# Agents declare by name — the runtime resolves the template
agent = SimpleAgent("summarizer", prompt_name="summarizer.system")
```

**Prompt file (`prompts/agents.yaml`):**
```yaml
summarizer.system:
  content: "You are a document summarization agent. Be concise."
  version: "1.0"
  description: "System prompt for the summarizer agent."
```

---

## Built On

Synaptum's LLM layer is powered by **[Axonium SDK](https://github.com/Root1V/axonium-sdk.git)** — a lightweight Python SDK that provides a unified, observable adapter interface for local LLM inference servers (llama.cpp, vLLM, and any OpenAI-compatible endpoint). Axonium handles HTTP transport, authentication, observability instrumentation, and async I/O, so Synaptum can remain focused on agent orchestration.

> Synaptum's `LLMClient` abstraction is intentionally decoupled from Axonium. You can replace `LlamaClient` with any custom implementation — see [Extending Synaptum](#extending-synaptum).

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.13 | Uses `asyncio`, `dataclasses`, `typing` |
| [uv](https://docs.astral.sh/uv/) | any | Recommended package manager |
| LLM Server | — | llama.cpp / vLLM / any OpenAI-compatible endpoint |

### Option A — Install as a dependency (recommended)

```bash
uv add git+https://github.com/Root1V/synaptum-framework.git
```

Or with pip:

```bash
pip install git+https://github.com/Root1V/synaptum-framework.git
```

### Option B — Clone for local development

```bash
git clone https://github.com/Root1V/synaptum-framework.git
cd synaptum-framework
uv sync
```

### Environment setup

Create a `.env` file in your project root (see [`.env.example`](.env.example)):

```dotenv
LLAMA_BASE_URL=http://localhost:8080   # your LLM server URL
LLAMA_API_KEY=your-api-key-here        # leave empty if not required
LLAMA_MODEL=your-model-name            # e.g. llama-3.2-3b-instruct
```

Then load it at the top of your application:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Verify

```bash
uv run python -c "import synaptum; print('Synaptum OK')"
```

---

## Quick Start

### LLM agent with prompt from file

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()

from synaptum import AgentRuntime, SimpleAgent
from synaptum.prompts import FilePromptProvider
from synaptum.messaging.in_memory_bus import InMemoryMessageBus

async def client_handler(agent, msg, ctx):
    if msg.type == "agent.output":
        print("Result:", msg.payload["answer"])

async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus, prompts=FilePromptProvider("prompts/agents.yaml"))

    rt.register(SimpleAgent("analyst", prompt_name="analyst.system"))
    rt.register(SimpleAgent("client", handler=client_handler))

    await rt.start(run_id="run-1")

    await rt.publish(Message(
        sender="client", recipient="analyst",
        type="request", payload={"text": "Summarize the AI market in 2025"},
        reply_to="client",
    ))

    await rt.run_until_idle()
    await rt.stop()

asyncio.run(main())
```

> More examples covering messaging, routing, supervisor/worker, pipelines, pub/sub and more are available in the [`examples/`](examples/) directory.

---

## Multi-Agent Patterns

Synaptum includes ready-to-use coordination patterns in `synaptum.patterns`:

| # | Pattern | API | Use case |
|---|---|---|---|
| 06 | Request / Reply | `SimpleAgent` | Single agent query/response |
| 07 | Supervisor Queue | `SimpleAgent` + handlers | Worker pool with dynamic task dispatching |
| 08 | Pipeline | `SimpleAgent` chain | Fixed linear processing steps |
| 09 | Pub/Sub | `SimpleAgent` + bus | Broadcast events to multiple subscribers |
| 10 | Debate / Critique | `SimpleAgent` × 2 | Two agents argue iteratively toward a conclusion |
| 11 | Router Dispatcher | `RouterPattern` | LLM decides which specialist handles the task |
| 12 | Blackboard | `SimpleAgent` + shared state | Agents read/write a common knowledge store |
| 13 | Graph — sequential | `GraphBuilder` | State machine with typed state and conditional transitions |
| 14 | Graph — parallel | `parallel()` + `GraphBuilder` | Fork/join: independent agents run concurrently |
| 15 | Map-Reduce | `MapReduceAgent` | Same agent runs over N independent chunks simultaneously; reducer aggregates |
| 16 | Plan-and-Execute | `PlanAndExecuteAgent` | Planner generates a step-by-step plan; specialists execute each step; replanner may revise mid-run |
| 17 | Swarm / Handoff | `SwarmAgent` | Peer agents autonomously hand off control to each other; no central dispatcher |
| 18 | Reflection Loop | `ReflectionAgent` | Generator produces output; critic scores it; generator iterates until quality threshold |
| 19 | Consensus / Voting | `ConsensusAgent` | N panelists vote independently on the same input; judge synthesises a final decision |
| 20 | Human-in-the-Loop | `HITLAgent` | Automated screening pauses for a mandatory human decision gate before executing |
| 21 | Saga | `SagaAgent` | Ordered steps with per-step compensators; failure triggers LIFO rollback |

See the [`examples/`](examples/) directory for runnable code for each pattern.

### Pattern decision guide

```
Single agent?
└─ Request/Reply (06)

Fixed sequence of agents?
├─ Linear chain           → Pipeline (08)
├─ Typed state machine    → Graph sequential (13)
└─ State + parallel fork  → Graph parallel (14)

Dynamic dispatch?
├─ LLM picks a specialist → Router (11)
├─ LLM assigns subtasks   → Supervisor Queue (07)
└─ Same agent, N chunks   → Map-Reduce (15)

Plan upfront, then execute?
└─ Plan-and-Execute (16)     ← plan can be revised mid-run

Agents decide the flow themselves?
└─ Swarm / Handoff (17)      ← no central authority

Iterative quality improvement?
└─ Reflection Loop (18)      ← critic drives revisions

Multiple independent opinions?
└─ Consensus / Voting (19)   ← fan-out + judge synthesises

Human must review before proceeding?
└─ Human-in-the-Loop (20)    ← execution pauses at decision gate

Distributed transaction with rollback?
└─ Saga (21)                 ← LIFO compensating transactions
```

---

## Project Structure

```
synaptum/
├── core/
│   ├── agent.py          # Agent ABC — name, id, on_message
│   ├── runtime.py        # AgentRuntime — register, start, run_until_idle
│   ├── message.py        # Message dataclass
│   └── context.py        # AgentContext — run_id, metadata, agent_names()
├── agents/
│   ├── simple_agent.py   # SimpleAgent — LLM + prompt + optional handler (legacy)
│   ├── message_agent.py  # MessageAgent — bus messaging, no LLM; base for all agents
│   ├── llm_agent.py      # LLMAgent — MessageAgent + think() + structured output
│   ├── composite_agent.py# CompositeAgent — owns a sub-agent topology; auto-registers children
│   ├── graph_agent.py    # GraphAgent — drives a compiled Graph (internal, created by GraphBuilder)
│   └── llm_tool_agent.py # LLMToolAgent — tool-use loop
├── prompts/
│   ├── template.py       # PromptTemplate — content, version, variables
│   ├── provider.py       # PromptProvider ABC
│   ├── in_memory.py      # InMemoryPromptProvider
│   ├── file_provider.py  # FilePromptProvider (YAML / JSON)
│   └── registry.py       # PromptRegistry — chains multiple providers
├── messaging/
│   ├── bus.py            # MessageBus ABC
│   └── in_memory_bus.py  # InMemoryMessageBus
├── llm/
│   ├── client.py         # LLMClient ABC + LLMResponse
│   └── llama_client.py   # LlamaClient (axonium SDK)
├── patterns/
│   ├── router.py         # RouterPattern
│   ├── supervisor.py     # SupervisorPattern
│   ├── graph_builder.py  # GraphBuilder, Graph, ParallelNode, parallel(), END
│   ├── map_reduce.py     # MapReduceAgent — fan-out/fan-in over N independent chunks
│   ├── plan_execute.py   # PlanAndExecuteAgent — upfront plan + adaptive replan
│   ├── swarm.py          # SwarmAgent — peer handoff choreography
│   ├── reflection.py     # ReflectionAgent — generate / critique / revise loop
│   ├── consensus.py      # ConsensusAgent — fan-out panelists + judge synthesis
│   ├── hitl.py           # HITLAgent — screener → human gate → executor
│   └── saga.py           # SagaAgent — ordered steps + LIFO compensating transactions
├── tools/                # Tool registry for LLMToolAgent
└── memory/               # Pluggable memory store
```

### Agent class hierarchy

```
Agent (ABC)
└── MessageAgent          bus messaging + _ref; no LLM
    ├── LLMAgent          adds think() + structured output via Pydantic
    └── CompositeAgent    owns a sub-agent topology; auto-registers children
        ├── ConsensusAgent
        ├── HITLAgent
        └── SagaAgent

SimpleAgent               legacy; combines LLM + optional handler (pre-0.4)
```

### When to use each agent base

| Base class | Use when |
|---|---|
| `MessageAgent` | Deterministic routing, client stubs, human gates, aggregators — anything that sends/receives but never calls an LLM |
| `LLMAgent` | Any agent whose primary job is to call the language model and produce structured or free-text output |
| `CompositeAgent` | Your agent wires together a topology of sub-agents registered automatically by the runtime (e.g. `ConsensusAgent`, `HITLAgent`, `SagaAgent`) |
| `SimpleAgent` | Legacy — kept for backward compatibility with examples 06–14 |

---

## Extending Synaptum

### Custom LLM client

```python
from synaptum.llm.client import LLMClient, LLMResponse

class OpenAIClient(LLMClient):
    async def chat(self, messages, **kwargs) -> LLMResponse:
        response = await openai_client.chat.completions.create(
            model="gpt-4o", messages=messages, **kwargs
        )
        return LLMResponse(content=response.choices[0].message.content)
```

### Custom prompt provider (remote / database)

```python
from synaptum.prompts import PromptProvider, PromptTemplate

class DBPromptProvider(PromptProvider):
    def get(self, name: str) -> PromptTemplate:
        row = db.query("SELECT content, version FROM prompts WHERE name = ?", name)
        return PromptTemplate(content=row.content, version=row.version)

    def exists(self, name: str) -> bool:
        return db.exists("SELECT 1 FROM prompts WHERE name = ?", name)
```

---

## Changelog

### [0.4.0] — 2026-03-06

- **`MessageAgent`** — new base class for all bus-capable, non-LLM agents; provides `self._ref` (`AgentRef`) and `_bind_runtime` automatically; replaces boilerplate in every pattern class
- **`LLMAgent(MessageAgent)`** — new base for all LLM agents; inherits bus wiring from `MessageAgent`, adds `think(user_message) → str | BaseModel`; replaces direct `SimpleAgent` usage in patterns 16–20
- **`CompositeAgent(MessageAgent)`** — new base for agents that own a sub-agent topology; `sub_agents()` → runtime auto-registers all children on `register(composite)`; used by `ConsensusAgent`, `HITLAgent`, `SagaAgent`
- **`MapReduceAgent`** — new pattern (example 15); splits input into N independent chunks, maps the same agent concurrently over all chunks via `asyncio.gather()`, reduces all results with an aggregator agent; single `runtime.register()` call also registers mapper and reducer
- **`PlanAndExecuteAgent(MessageAgent)`** — refactored to pure async message choreography (example 16); planner, replanner, finalizer, and executor agents are external; all LLM calls go through the bus (`_PE_PLAN / _PE_EXECUTE / _PE_REPLAN / _PE_FINALIZE` internal message types); `_pending_msgs` correlates replies to in-flight phases; per-step timing via `time.perf_counter()`; all prompt text in YAML (`plan/exec/replan/final_user_prompt` keys)
- **`SwarmAgent(MessageAgent)`** — new pattern (example 17); peer agents autonomously hand off control via `HandoffDecision` (Pydantic model with `handoff_to`); no external orchestrator; `_SWARM_TURN` internal message type; participants registered independently; all structural prompt text in YAML
- **`ReflectionAgent(MessageAgent)`** — new pattern (example 18); `Critique` model with `score / passed / dimension_scores / revision_instructions`; passes best output to caller even if budget exhausted without passing; per-iteration timing; YAML-driven generate and critique user prompts
- **`ConsensusAgent(CompositeAgent)`** — new pattern (example 19); `_PanelistAgent(LLMAgent)` fan-out → `_AggregatorAgent(MessageAgent)` fan-in → `_JudgeAgent(LLMAgent)` synthesis; `PanelistVerdict` Pydantic model; single `runtime.register(consensus)` registers all internal agents
- **`HITLAgent(CompositeAgent)`** — new pattern (example 20); `_HITLScreenerAgent` → `_HITLGateAgent` (human pause, no LLM) → `_HITLExecutorAgent`; `HumanReviewRequest / HumanReviewResponse` Pydantic models; `ReviewHandler` type alias for the async callable; `_APPROVAL_DECISIONS` set for case-insensitive approval matching
- **`SagaAgent(CompositeAgent)`** — new pattern (example 21); `_SagaForwardAgent / _SagaCompensatorAgent / _SagaOutcomeAgent` internal nodes; `SagaStep` Pydantic model (name, description, forward_prompt, compensate_prompt); `StepResult / StepAuditEntry / SagaOutcome` models; full LIFO rollback on failure; all saga state carried immutably in `__saga__` key of message payload — no shared mutable state anywhere
- **YAML-first rule** — from pattern 16 onward, all structural prompt text lives in YAML; Python only serialises data variables (`fmt_dict`, `fmt_list`, `fmt_records` from `synaptum.utils.formatting`)
- **Independent registration rule** — from pattern 16 onward, all participant agents are registered independently via `runtime.register()` (no implicit cascade); `CompositeAgent` sub-agents are the only exception (auto-registered once)

### [0.3.0] — 2026-03-01

- **`GraphBuilder` declarative API** — replaces the old procedural `GraphPattern`; follows the LangGraph `StateGraph` pattern: nodes are `SimpleAgent` instances, edges declare topology, `build()` returns a `GraphAgent` that drives the entire graph
- **Typed run state** — `GraphBuilder(name, state=MyTypedDict)` accepts a `TypedDict` that flows through every node; each stage writes its output under a normalised key (hyphens → underscores); conditional router lambdas receive the **full accumulated state dict**
- **`parallel()` fork/join** — `parallel(name, agent_a, agent_b, agent_c)` creates a `ParallelNode` that executes all children concurrently via `asyncio.gather()` and merges their results back into the shared state; the graph sees it as a single node
- **Single registration** — `runtime.register(processor)` is enough; the `GraphAgent` automatically cascade-registers all child nodes (including `ParallelNode` children)
- **`GraphAgent`** — internal execution engine created by `build()`; no message-bus round-trips between internal stages; only two messages exist on the bus per run (submit + result)
- **`_resolve_name` / `_resolve_target`** — moved from module-level to private `@staticmethod` inside `GraphBuilder`
- **Example 13** — mortgage state machine fully rewritten with new `GraphBuilder` API and `MortgageState(TypedDict)`
- **Example 14** — new personal loan assessment demonstrating `parallel()` with three concurrent risk-check agents (`credit-check`, `employment-verify`, `fraud-scan`)

### [0.2.0] — 2026-02-28

- **Prompt Management System** — new `synaptum.prompts` module with `PromptTemplate`, `PromptProvider` ABC, `InMemoryPromptProvider`, `FilePromptProvider` (YAML/JSON, lazy + cached) and `PromptRegistry` (priority chain)
- **Runtime-level prompt injection** — `AgentRuntime` now accepts `prompts: PromptProvider`; agents declare `prompt_name` and the runtime resolves the template at registration time
- **Smart LLM init** — `SimpleAgent` only instantiates `LlamaClient` when a prompt is present; passive agents have no LLM overhead
- **Rename `agent_id` → `name`** — cleaner API aligned with AutoGen, CrewAI, and LangGraph conventions; `agent.id` remains the internal UUID
- **Axonium SDK ≥ 0.6.0** — replaced `run_in_executor` workaround with native `await adapter.async_chat()`

### [0.1.0] — 2026-02-27 *(initial release)*

- Core primitives: `Agent` ABC, `Message`, `AgentRuntime`, `AgentContext`
- `SimpleAgent` and `LLMToolAgent` implementations
- `InMemoryMessageBus` with async publish/subscribe/deliver
- `LlamaClient` wrapping Axonium SDK
- Patterns: `RouterPattern`, `SupervisorPattern`, `GraphPattern`
- Pluggable `MemoryStore` and `ToolRegistry`
- Example suite covering echo, tool-use, router, supervisor/worker, graph, pipeline, pub/sub, blackboard and debate/critique patterns

---

## 🙌 Acknowledgements

If you find Synaptum useful in your work, consider:

- ⭐ **Starring the repository** — it helps others discover the project
- 📢 **Sharing it with your team** — especially if you're building LLM-powered systems
- 🤝 **Contributing improvements or reporting issues** — any feedback is welcome

For citations or references in technical documentation:

| Field | Value |
|---|---|
| **Project** | Synaptum |
| **Repository** | https://github.com/Root1V/synaptum-framework |
| **Author** | Emeric Espiritu Santiago |
| **Contact** | emericespiritusantiago@gmail.com |
| **License** | MIT |

Synaptum is built on top of **[Axonium SDK](https://github.com/Root1V/axonium-sdk.git)** — if you use the LLM layer directly, consider acknowledging that project as well.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

1. **Fork** the repository
2. **Create a branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/my-new-feature
   # or
   git checkout -b bugfix/fix-some-bug
   ```
3. **Make your changes** — ensure code is well-commented and follows the project style
4. **Run the examples** to verify nothing is broken:
   ```bash
   uv run python examples/patterns/06_request_reply.py
   ```
5. **Submit a pull request** with a clear description of your changes

You can also open an [issue](https://github.com/Root1V/synaptum-framework/issues) to:

- Report a bug
- Suggest an enhancement
- Propose a new feature or architecture improvement

---

## License

MIT © 2026 Synaptum Contributors
