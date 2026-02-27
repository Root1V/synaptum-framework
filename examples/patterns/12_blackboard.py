import asyncio

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.memory.in_memory import InMemoryMemoryStore
from synaptum.messaging.in_memory_bus import InMemoryMessageBus

    
async def client_handler(agent: SimpleAgent, msg, ctx):
    if msg.type == "agent:output":
        print("CLIENT GOT result:", msg.payload.get("text"))


async def main():
    bus = InMemoryMessageBus()
    rt = AgentRuntime(bus)
    
    blackboard = InMemoryMemoryStore() # <---- la pizarra negra compartida
    
    async def writer_handler(agent: SimpleAgent, msg, ctx):
        if msg.type != "case.new":
            return
        
        case_id = msg.payload["case_id"]
        text = msg.payload["text"]
        
        # Escribe la primera pieza en la pizarra negra
        await blackboard.append(f"case:{case_id}", {"source": "writer", "note": text})
        
        await agent._ref.send(
            to="risk",
            payload={"case_id": case_id},
            type="case.updated",
            reply_to=msg.reply_to
        )
    
    
    async def risk_handler(agent: SimpleAgent, msg: Message, ctx):
        if msg.type != "case.updated":
            return
        
        case_id = msg.payload["case_id"]
        # Lee toda la información del caso desde la pizarra negra
        entries = await blackboard.read(f"case:{case_id}")
                
        # Heuristica simple si hay palabra fraude hay riesgo alto
        joined = " ".join ( [str(e) for e in entries]).lower()
        score = 0.9 if "fraude" in joined else 0.2
        
        await blackboard.append(f"case:{case_id}", {"source": "risk", "risk_score": score})
        
        # Notifica al reviewer
        await agent._ref.send(
            to="reviewer",
            payload={"case_id": case_id},
            type="case.ready",
            reply_to=msg.reply_to
        )
            
    async def reviewer_handler(agent: SimpleAgent, msg: Message, ctx):
        if msg.type != "case.ready":
            return
        
        case_id = msg.payload["case_id"]
        
        # Lee toda la información del caso desde la pizarra negra
        entries = await blackboard.read(f"case:{case_id}")
        risk_score = next((e["risk_score"] for e in entries if e.get("risk_score") is not None), 0)
        
        final_text = f"Case {case_id} - Blackboard entries: \n " + "\n".join([str(e) for e in entries]) + f"\nFinal decision based on risk score {risk_score}"
        
        await agent._ref.send(
            to=msg.reply_to,
            payload={"text": final_text},
            type="agent:output",
        )
        
    writer = SimpleAgent("writer", handler=writer_handler)  
    risk = SimpleAgent("risk", handler=risk_handler)
    reviewer = SimpleAgent("reviewer", handler=reviewer_handler)
    
    client = SimpleAgent("client", handler=client_handler)

    rt.register(writer)
    rt.register(risk)
    rt.register(reviewer)
    rt.register(client)
    
    await rt.start(run_id="run-blackboard")

    await client._ref.send(
        to="writer",
        payload={"case_id": "123", "text": "Se detectó posible fraude en la transacción."},
        type="case.new",
        reply_to="client",
    )

    await rt.run_until_idle()
    await rt.stop()

asyncio.run(main())
