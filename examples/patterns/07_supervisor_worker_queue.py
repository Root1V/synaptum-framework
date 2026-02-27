import asyncio
from collections import deque

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus

# estado del supervisor: por cada run guarda caller, cola de tareas, workers libres y resultados
_sup_state: dict = {}

async def supervisor_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type == "start":
        # descubrir workers dinámicamente desde el contexto
        worker_ids = ctx.agent_ids(prefix="worker")
        tasks = deque(msg.payload["tasks"])
        run_key = msg.id
        _sup_state[run_key] = {
            "caller": msg.reply_to or msg.sender,
            "task_queue": tasks,
            "idle_workers": set(worker_ids),
            "pending": len(tasks),
            "results": [],
        }
        # asignar una tarea a cada worker libre hasta agotar cola o workers
        await _dispatch(agent, run_key)

    elif msg.type == "result":
        run_key = msg.payload.get("run_key")
        if run_key not in _sup_state:
            return
        state = _sup_state[run_key]
        state["results"].append({"worker": msg.payload["worker"], "result": msg.payload["result"]})
        state["pending"] -= 1
        worker_id = msg.payload["worker"]
        state["idle_workers"].add(worker_id)  # worker vuelve al pool

        if state["pending"] == 0:
            # todas las tareas completadas → responder al caller
            await agent._ref.send(
                to=state["caller"],
                type="reply",
                payload={"results": state["results"]},
            )
            del _sup_state[run_key]
        else:
            # hay tareas pendientes en cola → asignar al worker que acaba de quedar libre
            await _dispatch(agent, run_key)


async def _dispatch(agent: SimpleAgent, run_key: str):
    """Asigna tareas de la cola a workers libres."""
    state = _sup_state[run_key]
    while state["task_queue"] and state["idle_workers"]:
        task = state["task_queue"].popleft()
        worker_id = state["idle_workers"].pop()
        await agent._ref.send(
            to=worker_id,
            type="task",
            payload={"task": task, "run_key": run_key},
        )


async def worker_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type != "task":
        return
    task = msg.payload["task"]
    run_key = msg.payload["run_key"]
    result = f"processed: {task}"
    await agent._ref.send(
        to="supervisor",
        type="result",
        payload={"worker": agent.agent_id, "result": result, "run_key": run_key},
    )


async def client_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type == "reply":
        for r in msg.payload["results"]:
            print(f"  {r['worker']}: {r['result']}")


async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus)

    supervisor = SimpleAgent("supervisor", handler=supervisor_handler)
    # 3 workers para 8 tareas → distribución dinámica
    workers = [SimpleAgent(f"worker-{i}", handler=worker_handler) for i in range(3)]
    client = SimpleAgent("client", handler=client_handler)

    rt.register(supervisor)
    for worker in workers:
        rt.register(worker)
    rt.register(client)

    await rt.start(run_id="run-supervisor-worker")

    await client._ref.send(
        to="supervisor",
        payload={"tasks": [f"task-{i}" for i in range(8)]},
        type="start",
        reply_to="client",
    )

    await rt.run_until_idle()
    await rt.stop()

asyncio.run(main())
