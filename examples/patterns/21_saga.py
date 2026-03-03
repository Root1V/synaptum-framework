"""
Pattern: Saga / Compensating Transactions — Cross-Border Wire Transfer
======================================================================
Demonstrates the ``SagaAgent`` pattern using **true message choreography**.

There is no central orchestrator.  Each step is an autonomous agent that
reacts to a message, performs its action via LLM, and fires the next
message.  The SagaAgent only launches the chain — after that it steps
back completely.

Architecture (choreography, not orchestration)
-----------------------------------------------

  SagaAgent("wire-saga")
    registers:
      wire-saga._fwd.debit-source
      wire-saga._fwd.fx-conversion
      wire-saga._fwd.swift-transmission
      wire-saga._fwd.credit-destination
      wire-saga._cmp.fx-conversion       ← compensator for step 2
      wire-saga._cmp.debit-source        ← compensator for step 1
      wire-saga._outcome                 ← terminal node

  Message flow (all via bus — no Python loops at runtime):

  client ──[wire.saga.started]──▶ wire-saga
                  │
                  └──[__saga.fwd__]──▶ _fwd.debit-source         ✓ JE-XXXXX
                                             │
                                        [__saga.fwd__]──▶ _fwd.fx-conversion      ✓ FXD-XXXXX
                                                               │
                                                          [__saga.fwd__]──▶ _fwd.swift-transmission  ✗ OFFLINE
                                                                                 │
                                                                            [__saga.cmp__]──▶ _cmp.fx-conversion    ↩ REV-FXD
                                                                                                     │
                                                                                                [__saga.cmp__]──▶ _cmp.debit-source  ↩ REV-JE
                                                                                                                       │
                                                                                                                  [__saga.cmp__]──▶ _outcome ──▶ client

Saga state
----------
All intermediate results travel immutably inside the message payload
under the ``__saga__`` key.  No shared mutable state.  No central loop.

Saga vs other patterns
-----------------------
  Plan-and-Execute → dynamic steps; no rollback
  Swarm            → handoff control; no compensation concept
  HITL             → single human gate
  Reflection       → iterative quality loop; no side-effects to reverse
  Saga             → FIXED ordered steps + compensators;
                     failure at step K triggers LIFO rollback of K-1…0;
                     **entire flow is driven by messages, not Python loops**
"""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.context import AgentContext
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.patterns.saga import SagaAgent, SagaOutcome, SagaStep
from synaptum.prompts import FilePromptProvider


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    bus             = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/saga.yaml")
    runtime         = AgentRuntime(bus, prompts=prompt_provider)

    results: list = []

    async def client_handler(agent, message: Message, context: AgentContext) -> None:
        if message.type == "wire.saga.completed":
            results.append(message.payload)

    # ── Saga definition ────────────────────────────────────────────────────
    # Each SagaStep declares the LLM prompt for the forward action and its
    # compensating action.  No agent objects to create manually — SagaAgent
    # builds and registers all internal step/compensator/outcome agents
    # automatically during runtime.register().
    wire_saga = SagaAgent(
        "wire-saga",
        steps = [
            SagaStep(
                name              = "debit-source",
                description       = "Debit USD from payer account and record GL journal entry",
                forward_prompt    = "bank.saga.debit_source.system",
                compensate_prompt = "bank.saga.credit_source.system",
            ),
            SagaStep(
                name              = "fx-conversion",
                description       = "Convert USD to EUR at today's spot rate via FX Trading Desk",
                forward_prompt    = "bank.saga.fx_conversion.system",
                compensate_prompt = "bank.saga.fx_reversal.system",
            ),
            SagaStep(
                name              = "swift-transmission",
                description       = "Transmit SWIFT MT103 to JPMorgan Madrid as correspondent bank",
                forward_prompt    = "bank.saga.swift_transmission.system",
                compensate_prompt = "bank.saga.swift_cancellation.system",
            ),
            SagaStep(
                name              = "credit-destination",
                description       = "Credit EUR to beneficiary account at Banco Santander Madrid",
                forward_prompt    = "bank.saga.credit_destination.system",
                compensate_prompt = "bank.saga.debit_destination.system",
            ),
        ],
        trigger_type = "wire.saga.started",
        result_type  = "wire.saga.completed",
        verbose      = True,
    )

    client = SimpleAgent("client", handler=client_handler)

    runtime.register(wire_saga)   # registers wire-saga + all 9 internal agents
    runtime.register(client)

    await runtime.start(run_id="run-wire-saga-2026-031")

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  EXAMPLE 21 — Saga / Compensating Transactions                  ║")
    print("║  Cross-border wire — Constructora Andina S.A.S.                 ║")
    print("║  USD 451,163 → EUR 420,000 | Madrid, Spain                      ║")
    print("║  Scenario: SWIFT fails (correspondent bank OFFLINE)              ║")
    print("║  Expected: Steps 1+2 committed, step 3 fails, rollback 2→1      ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    await client._ref.send(
        to      = "wire-saga",
        type    = "wire.saga.started",
        payload = {
            # Transaction details
            "transfer_id":             "TXF-2026-031-0042",
            "customer_name":           "Constructora Andina S.A.S.",
            "customer_account":        "041-38271054-6",
            "customer_since":          "2014",
            "amount_usd":              451_163.00,
            "target_currency":         "EUR",
            "target_amount_eur":       420_000.00,
            "indicative_rate_usd_eur": 0.9309,
            # Beneficiary
            "beneficiary_name":        "Construcciones Ibéricas S.L.",
            "beneficiary_account":     "ES76-0049-1805-9127-1016-4325",
            "beneficiary_bank":        "Banco Santander",
            "beneficiary_bic":         "BSCHESMMXXX",
            "beneficiary_country":     "Spain",
            # Correspondent routing
            "correspondent_bank":      "JPMorgan Chase Madrid Branch",
            "correspondent_bic":       "CHASESM2XXX",
            # Transfer metadata
            "purpose":                 "Construction materials and prefab structures — Invoice ES-2026-0441",
            "invoice_reference":       "ES-2026-0441",
            "value_date":              "2026-03-05",
            # ── Demo failure injection ──
            # Setting this flag forces the SWIFT step to return success=False
            "beneficiary_bank_status": "OFFLINE",
        },
        reply_to = "client",
    )

    await runtime.run_until_idle()
    await runtime.stop()

    # ── Display outcome ────────────────────────────────────────────────────
    if not results:
        print("No result received.")
        return

    outcome = SagaOutcome.model_validate(results[0])

    status_icon = {
        "COMMITTED":       "✅",
        "ROLLED_BACK":     "↩",
        "PARTIAL_ROLLBACK": "⚠️",
    }.get(outcome.status, "●")

    BOLD  = "\033[1m"
    RESET = "\033[0m"
    GREEN = "\033[32m"
    RED   = "\033[31m"
    CYAN  = "\033[36m"
    W = 68

    print(f"\n{'═' * W}")
    print(f"  SAGA OUTCOME — {outcome.status}  {status_icon}")
    print(f"{'═' * W}")

    if outcome.failed_step:
        print(f"  Failed step   : {BOLD}{outcome.failed_step}{RESET}")
        print(f"  Failure reason: {outcome.failure_reason[:110]}")

    print(f"  Steps completed before failure : {outcome.steps_completed}")
    if outcome.compensations_applied:
        print(f"  Compensations applied          : {', '.join(outcome.compensations_applied)}")

    print(f"\n  {BOLD}SUMMARY{RESET}")
    for line in outcome.summary.split(". "):
        if line.strip():
            print(f"  {line.strip()}.")

    print(f"\n  {BOLD}AUDIT LOG{RESET}")
    for entry in outcome.audit_log:
        icon = {
            "COMPLETED":           f"{GREEN}✓{RESET}",
            "FAILED":              f"{RED}✗{RESET}",
            "COMPENSATED":         f"\033[33m↩{RESET}",
            "COMPENSATION_FAILED": f"{RED}⚠{RESET}",
        }.get(entry.status, "●")
        ref = f"  [ref: {entry.reference_id}]" if entry.reference_id else ""
        print(f"  {icon}  {entry.step_name:<26}  {entry.status:<22}{ref}")
        if entry.failure_reason:
            print(f"       ↳ {entry.failure_reason[:90]}")

    print(f"{'═' * W}\n")


if __name__ == "__main__":
    asyncio.run(main())
