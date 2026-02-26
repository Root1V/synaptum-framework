import asyncio

from agentium.core.message import Message
from agentium.core.agent import Agent
from agentium.core.runtime import AgentRuntime
from agentium.messaging.in_memory_bus import InMemoryMessageBus
from agentium.core.context import AgentContext


class EchoAgent(Agent):
    async def on_message(self, message: Message, context: AgentContext) -> None:
        if message.type == "user.input":
            txt = message.payload.get("text") if isinstance(message.payload, dict) else str(message.payload)
            reply_to = message.metadata.get("reply_to") or message.sender
            await self.runtime.publish(Message(
                sender=self.agent_id,
                recipient=reply_to,
                type="agent.output",
                payload={"text": f"echo: {txt}"},
                metadata={"in_reply_to": message.id},
            ))


async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus)
    echo = EchoAgent("echo")
    client = EchoAgent("client")  # cliente también es agente para recibir output

    rt.register(echo)
    rt.register(client)

    async def client_on_message(msg: Message, ctx: AgentContext):
        if msg.type == "agent.output":
            print("CLIENT GOT:", msg.payload["text"])

    # monkeypatch handler simple (para demo)
    client.on_message = client_on_message  # type: ignore

    await rt.start(run_id="run-echo")

    await rt.publish(Message(
        sender="client",
        recipient="echo",
        type="user.input",
        payload={"text": "hola"},
        metadata={"reply_to": "client"},
    ))

    await rt.run_until_idle()
    await rt.stop()

asyncio.run(main())
