# Synaptum

**A minimal, bus-driven agent framework for building LLM-powered single-agents and multi-agent systems in Python.**

Synaptum sits between your LLM SDK and your agent platform. It gives you the minimal primitives — agents, a message bus, a runtime, and prompt management — to build reliable, testable agent applications without opinion on your infrastructure.

> **Version:** 0.2.0 · **Python:** ≥ 3.13 · **License:** MIT

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
│          │ uses                                                           │
│          ▼                                                                │
│   ┌──────────────────────────────────────────┐                           │
│   │              S Y N A P T U M             │                           │
│   │                                          │                           │
│   │  Minimal bus-driven agent framework      │                           │
│   │  for LLM-powered agent systems           │                           │
│   └──────┬───────────────────────────────────┘                           │
│          │                           │                                   │
│          │ reads prompts from        │ calls                             │
│          ▼                           ▼                                   │
│   ┌──────────────┐          ┌─────────────────────┐                     │
│   │  Prompt Store│          │   Axonium SDK        │                     │
│   │              │          │                      │                     │
│   │  YAML / DB / │          │  LLM adapter layer   │                     │
│   │  Remote API  │          └──────────┬───────────┘                     │
│   └──────────────┘                     │ invokes                         │
│                                        ▼                                 │
│                            ┌─────────────────────┐                      │
│                            │    LLM Server        │                      │
│                            │                      │                      │
│                            │  llama.cpp / vLLM /  │                      │
│                            │  OpenAI-compatible   │                      │
│                            └─────────────────────┘                      │
└──────────────────────────────────────────────────────────────────────────┘
```

### C4 Level 2 — Container Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Synaptum Framework                               │
│                                                                             │
│  ┌──────────────────────┐      ┌───────────────────────────────────────┐   │
│  │     AgentRuntime     │      │           Message Bus                 │   │
│  │                      │      │                                       │   │
│  │  - register(agent)   │─────▶│  publish(msg) ──▶ queue              │   │
│  │  - start(run_id)     │      │  subscribe(name, handler)            │   │
│  │  - run_until_idle()  │◀─────│  deliver(msg) ──▶ handlers           │   │
│  │  - prompts provider  │      │                                       │   │
│  └──────────┬───────────┘      └───────────────────────────────────────┘   │
│             │ injects                          ▲                            │
│             │ PromptProvider                   │ messages                   │
│             ▼                                  │                            │
│  ┌──────────────────────┐      ┌───────────────┴───────────────────────┐   │
│  │   Prompt System      │      │              Agents                   │   │
│  │                      │      │                                       │   │
│  │  PromptRegistry      │      │  SimpleAgent      LLMToolAgent        │   │
│  │  FilePromptProvider  │      │  ┌────────────┐  ┌────────────────┐  │   │
│  │  InMemoryProvider    │      │  │ name       │  │ name           │  │   │
│  │                      │      │  │ prompt     │  │ system_prompt  │  │   │
│  │  PromptTemplate      │      │  │ handler()  │  │ tools          │  │   │
│  │  - content           │      │  │ on_message │  │ on_message     │  │   │
│  │  - version           │      │  └─────┬──────┘  └───────┬────────┘  │   │
│  │  - variables         │      │        │                  │           │   │
│  └──────────────────────┘      └────────┼──────────────────┼───────────┘   │
│                                         │                  │               │
│  ┌──────────────────────┐               │                  │               │
│  │    LLM Layer         │◀──────────────┘                  │               │
│  │                      │                                  │               │
│  │  LLMClient (ABC)     │                                  │               │
│  │  LlamaClient         │      ┌───────────────────────────┘               │
│  │  (axonium SDK)       │      │                                           │
│  └──────────────────────┘      ▼                                           │
│                         ┌──────────────────────┐                           │
│                         │   Patterns           │                           │
│                         │                      │                           │
│                         │  RouterPattern       │                           │
│                         │  SupervisorPattern   │                           │
│                         │  GraphPattern        │                           │
│                         └──────────────────────┘                           │
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

## Quick Start

### Installation

```bash
git clone https://github.com/Root1V/synaptum-framework.git
cd synaptum-framework
uv sync
```

### Minimal agent (no LLM)

```python
import asyncio
from synaptum import AgentRuntime, SimpleAgent, Message
from synaptum.messaging.in_memory_bus import InMemoryMessageBus

async def echo_handler(agent, msg, ctx):
    if msg.type == "ping":
        await agent._ref.send(to=msg.reply_to, type="pong", payload=msg.payload)

async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus)
    rt.register(SimpleAgent("echo", handler=echo_handler))
    await rt.start(run_id="demo")
    ...

asyncio.run(main())
```

### LLM agent with prompt from file

```python
from synaptum import AgentRuntime, SimpleAgent
from synaptum.prompts import FilePromptProvider
from synaptum.messaging.in_memory_bus import InMemoryMessageBus

async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus, prompts=FilePromptProvider("prompts/agents.yaml"))

    rt.register(SimpleAgent("analyst", prompt_name="analyst.system"))
    rt.register(SimpleAgent("client", handler=client_handler))

    await rt.start(run_id="run-1")
    ...
```

---

## Multi-Agent Patterns

Synaptum includes ready-to-use coordination patterns in `synaptum.patterns`:

| Pattern | Class | Use case |
|---|---|---|
| Request / Reply | `SimpleAgent` | Single agent query/response |
| Router | `RouterPattern` | LLM decides which specialist handles the task |
| Supervisor / Worker | `SupervisorPattern` | LLM plans tasks and distributes to workers |
| Graph | `GraphPattern` | Sequential pipeline with conditional transitions |
| Supervisor Queue | `SimpleAgent` + handlers | Worker pool with dynamic task dispatching |

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
│   └── graph.py          # GraphPattern
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

## Running the Examples

```bash
# Single agent — request/reply with LLM
uv run python examples/patterns/06_request_reply.py

# Supervisor/Worker queue — dynamic task dispatching
uv run python examples/patterns/07_supervisor_worker_queue.py
```

---

## License

MIT © 2026 Synaptum Contributors
