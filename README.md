# Synaptum

**A minimal, bus-driven agent framework for building LLM-powered single-agents and multi-agent systems in Python.**

Synaptum sits between your LLM SDK and your agent platform. It gives you the minimal primitives — agents, a message bus, a runtime, and prompt management — to build reliable, testable agent applications without opinion on your infrastructure.

> **Version:** 0.3.0 · **Python:** ≥ 3.13 · **License:** MIT

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
│  │  PromptRegistry      │      │  SimpleAgent      LLMToolAgent        │    │
│  │  FilePromptProvider  │      │  ┌────────────┐  ┌────────────────┐   │    │
│  │  InMemoryProvider    │      │  │ name       │  │ name           │   │    │
│  │                      │      │  │ prompt     │  │ system_prompt  │   │    │
│  │  PromptTemplate      │      │  │ handler()  │  │ tools          │   │    │
│  │  - content           │      │  │ on_message │  │ on_message     │   │    │
│  │  - version           │      │  └─────┬──────┘  └────────┬───────┘   │    │
│  │  - variables         │      │        │                  │           │    │
│  └──────────────────────┘      └────────┼──────────────────┼───────────┘    │
│                                         │                  │                │
│  ┌──────────────────────┐               │                  │                │
│  │    LLM Layer         │◀──────────────┘                  │                │
│  │                      │                                  │                │
│  │  LLMClient (ABC)     │                                  │                │
│  │  LlamaClient         │      ┌───────────────────────────┘                │
│  │  (axonium SDK)       │      │                                            │
│  └──────────────────────┘      ▼                                            │
│                         ┌──────────────────────┐                            │
│                         │   Patterns           │                            │
│                         │                      │                            │
│                         │  RouterPattern       │                            │
│                         │  SupervisorPattern   │                            │
│                         │  GraphPattern        │                            │
│                         └──────────────────────┘                            │
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

| Pattern | API | Use case |
|---|---|---|
| Request / Reply | `SimpleAgent` | Single agent query/response |
| Router | `RouterPattern` | LLM decides which specialist handles the task |
| Supervisor / Worker | `SupervisorPattern` | LLM plans tasks and distributes to workers |
| Graph — sequential | `GraphBuilder` | State machine with typed state and conditional transitions |
| Graph — parallel | `parallel()` + `GraphBuilder` | Fork/join: independent agents run concurrently via `asyncio.gather()` |
| Supervisor Queue | `SimpleAgent` + handlers | Worker pool with dynamic task dispatching |
| Pipeline | `SimpleAgent` chain | Fixed linear processing steps |
| Pub/Sub | `SimpleAgent` + bus | Broadcast events to multiple subscribers |
| Blackboard | `SimpleAgent` + shared state | Agents read/write a common knowledge store |

See the [`examples/`](examples/) directory for runnable code for each pattern.

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
│   ├── simple_agent.py   # SimpleAgent — LLM + prompt + custom handler
│   ├── graph_agent.py    # GraphAgent — drives a compiled Graph (internal)
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
│   └── graph_builder.py  # GraphBuilder, Graph, ParallelNode, parallel(), END
├── tools/                # Tool registry for LLMToolAgent
└── memory/               # Pluggable memory store
```

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
