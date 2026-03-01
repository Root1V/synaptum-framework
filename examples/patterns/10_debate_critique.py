"""
Pattern: Debate / Critique — Corporate Credit Committee (Banking)
=================================================================
A corporate credit line application is submitted to a moderator who fans it out
to two opposing analysts. Each argues their position independently, then the
moderator critiques both arguments and delivers the committee's final decision.

Flow:
  client  ──[credit.application]──▶  moderator
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                       analyst-optimist    analyst-conservative
                       (argues FOR)        (argues AGAINST)
                              │                     │
                              └──────────┬──────────┘
                                         ▼
                                      moderator  (synthesizes + critiques)
                                         │
                              ◀──[committee.decision]──  client

Key property: the moderator uses its own LLM to critique and synthesize,
not just concatenate — making the final output richer than either opinion alone.
"""

import asyncio

from dotenv import load_dotenv
load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.prompts import FilePromptProvider

async def moderator_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type == "credit.application":
        run_key = msg.id
        analyst_names = ctx.agent_names(prefix="analyst-")
        agent.state[run_key] = {
            "caller":      msg.reply_to or msg.sender,
            "application": msg.payload["application"],
            "opinions":    {},
            "pending":     len(analyst_names),
        }
        # fan-out both analysts simultaneously — discovered dynamically from context
        for analyst in analyst_names:
            await agent._ref.send(
                to=analyst,
                type="credit.review",
                payload={"application": msg.payload["application"], "run_key": run_key},
            )

    elif msg.type == "analyst.opinion":
        run_key = msg.payload.get("run_key")
        if run_key not in agent.state:
            return
        state = agent.state[run_key]
        state["opinions"][msg.sender] = msg.payload["opinion"]
        state["pending"] -= 1

        if state["pending"] == 0:
            # both opinions received — moderator critiques and synthesizes
            app_text = "\n".join(f"  {k}: {v}" for k, v in state["application"].items())
            synthesis_prompt = (
                f"Credit line application:\n{app_text}\n\n"
                + "\n\n".join(
                    f"[{name.upper()}]\n{opinion}"
                    for name, opinion in state["opinions"].items()
                )
                + "\n\nCritique both positions and deliver the committee's final decision."
            )
            decision = await agent.think(synthesis_prompt)

            await agent._ref.send(
                to=state["caller"],
                type="committee.decision",
                payload={"decision": decision},
            )
            del agent.state[run_key]


async def analyst_handler(agent: SimpleAgent, msg: Message, ctx):
    """Form and submit a position on the credit application."""
    if msg.type != "credit.review":
        return

    app = msg.payload["application"]
    app_text = "\n".join(f"  {k}: {v}" for k, v in app.items())
    opinion = await agent.think(
        f"Corporate credit line application:\n{app_text}\n\nPresent your position."
    )
    print(f"\n[{agent.name.upper()}]\n{opinion}")

    await agent._ref.send(
        to="moderator",
        type="analyst.opinion",
        payload={"opinion": opinion, "run_key": msg.payload["run_key"]},
    )


async def client_handler(agent: SimpleAgent, msg: Message, ctx):
    if msg.type == "committee.decision":
        print("\n── CREDIT COMMITTEE FINAL DECISION ─────────────────────────────")
        print(msg.payload["decision"])
        print("─────────────────────────────────────────────────────────────────\n")


async def main():
    bus = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/debate.yaml")
    runtime = AgentRuntime(bus, prompts=prompt_provider)

    moderator          = SimpleAgent("moderator",           prompt_name="bank.debate.moderator.system",    handler=moderator_handler)
    analyst_optimist   = SimpleAgent("analyst-optimist",    prompt_name="bank.debate.optimist.system",     handler=analyst_handler)
    analyst_conservative = SimpleAgent("analyst-conservative", prompt_name="bank.debate.conservative.system", handler=analyst_handler)
    client             = SimpleAgent("client",              handler=client_handler)

    runtime.register(moderator)
    runtime.register(analyst_optimist)
    runtime.register(analyst_conservative)
    runtime.register(client)

    await runtime.start(run_id="run-credit-committee")

    # corporate credit line application
    await client._ref.send(
        to="moderator",
        type="credit.application",
        payload={
            "application": {
                "company":              "TechNova Labs S.A.",
                "industry":             "B2B SaaS — HR automation",
                "requested_credit_line": "USD 2,000,000",
                "purpose":              "product expansion and sales team hiring",
                "annual_revenue":       "USD 4,200,000",
                "revenue_growth_yoy":   "38%",
                "ebitda_margin":        "11%",
                "current_debt":         "USD 1,800,000",
                "debt_to_equity":       "1.4x",
                "collateral":           "accounts receivable + IP",
                "operating_years":      4,
                "key_risk":             "35% revenue concentration in one client",
            }
        },
        reply_to="client",
    )

    await runtime.run_until_idle()
    await runtime.stop()


asyncio.run(main())

