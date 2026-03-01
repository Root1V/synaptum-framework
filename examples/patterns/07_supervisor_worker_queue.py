"""
Pattern: Supervisor / Fan-out — Loan Application Analysis (Banking)
====================================================================
A loan application arrives at the supervisor. It fans out the analysis
to all four specialist workers simultaneously (in parallel):

  credit-analyst      → evaluates creditworthiness
  risk-assessor       → assesses financial risk
  compliance-officer  → checks AML / KYC / policy compliance
  fraud-detector      → identifies fraud signals

Each worker uses its own system prompt (banking.yaml) to produce a focused
analysis. The supervisor collects all four reports and sends a consolidated
reply to the client.

Note: a queue is not used here because each task maps to exactly one
specialist worker. Fan-out is the correct pattern when tasks are
heterogeneous and workers are not interchangeable.
"""

import asyncio

from dotenv import load_dotenv
load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.prompts.file_provider import FilePromptProvider


async def supervisor_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type == "start":
        # discover all registered specialists dynamically — no hardcoded list needed
        specialist_names = ctx.agent_names(prefix="specialist-")
        run_key = msg.id
        agent.state[run_key] = {
            "caller":  msg.reply_to or msg.sender,
            "pending": len(specialist_names),
            "reports": [],
        }
        # fan-out: dispatch all specialists simultaneously
        for name in specialist_names:
            await agent._ref.send(
                to=name,
                type="analyze",
                payload={"application": msg.payload["application"], "run_key": run_key},
            )

    elif msg.type == "report":
        run_key = msg.payload.get("run_key")
        if run_key not in agent.state:
            return
        state = agent.state[run_key]
        state["reports"].append({
            "specialist": msg.payload["specialist"],
            "analysis":   msg.payload["analysis"],
        })
        state["pending"] -= 1

        if state["pending"] == 0:
            # all specialists done → consolidate and reply to client
            summary = "\n\n".join(
                f"[{r['specialist'].upper()}]\n{r['analysis']}"
                for r in state["reports"]
            )
            await agent._ref.send(
                to=state["caller"],
                type="decision",
                payload={"summary": summary},
            )
            del agent.state[run_key]


async def worker_handler(agent: SimpleAgent, msg: Message, ctx):
    """LLM-powered specialist: receives loan data, returns focused analysis."""
    if msg.type != "analyze":
        return

    application = msg.payload["application"]
    run_key = msg.payload["run_key"]

    app_text = "\n".join(f"  {k}: {v}" for k, v in application.items())
    user_message = f"Loan application received:\n{app_text}\n\nProvide your specialist analysis."
    print(f"\n[{agent.name.upper()}] received application, running analysis...")
    
    analysis = await agent.think(user_message)

    await agent._ref.send(
        to="supervisor",
        type="report",
        payload={
            "specialist": agent.name,
            "analysis":   analysis,
            "run_key":    run_key,
        }
    )


async def client_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type == "decision":
        print("\n── LOAN ANALYSIS REPORTS ──────────────────────────────────────")
        print(msg.payload["summary"])
        print("───────────────────────────────────────────────────────────────\n")


async def main():
    bus = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/loan.yaml")
    runtime = AgentRuntime(bus, prompts=prompt_provider)

    supervisor = SimpleAgent("supervisor", handler=supervisor_handler)
    specialist_workers = [
        SimpleAgent("specialist-credit-analyst",     prompt_name="bank.worker.credit_analyst.system",     handler=worker_handler),
        SimpleAgent("specialist-risk-assessor",      prompt_name="bank.worker.risk_assessor.system",      handler=worker_handler),
        SimpleAgent("specialist-compliance-officer", prompt_name="bank.worker.compliance_officer.system", handler=worker_handler),
        SimpleAgent("specialist-fraud-detector",     prompt_name="bank.worker.fraud_detector.system",     handler=worker_handler),
    ]
    client = SimpleAgent("client", handler=client_handler)

    runtime.register(supervisor)
    for worker in specialist_workers:
        runtime.register(worker)
    runtime.register(client)

    await runtime.start(run_id="run-loan-analysis")

    # sample loan application
    loan_application = {
        "applicant":          "Liesel Espiritu",
        "age":                34,
        "annual_income_usd":  72000,
        "requested_amount":   25000,
        "loan_purpose":       "home renovation",
        "credit_score":       710,
        "debt_to_income":     "28%",
        "employment_status":  "full-time, 6 years",
        "collateral":         "none",
        "existing_loans":     1,
    }

    await client._ref.send(
        to="supervisor",
        type="start",
        payload={"application": loan_application},
        reply_to="client",
    )

    await runtime.run_until_idle()
    await runtime.stop()


asyncio.run(main())
