"""
Pattern: Saga / Compensating Transactions — Cross-Border Wire Transfer
======================================================================
Demonstrates the ``SagaAgent`` pattern: a fixed sequence of steps each paired
with a compensating action.  If any step fails, all previously completed steps
are reversed in LIFO order — restoring the system to its pre-saga state.

Saga vs other patterns:

  Plan-and-Execute → planner generates steps dynamically; no rollback
  Swarm            → agents hand off control; no compensation concept
  HITL             → single human gate; not a multi-step transaction
  Reflection       → iterative quality loop; no side-effects to reverse
  Saga             → FIXED ordered steps; each paired with a compensator;
                     failure at step K triggers rollback of K-1…0 in reverse

Use-case — Cross-border wire transfer:
  A Colombian company wants to send EUR funds to a supplier in Spain.
  The transfer involves 4 sequential steps with external side effects:

    Step 1 → debit-source:        Debit USD from payer account (GL entry)
    Step 2 → fx-conversion:       Convert USD → EUR at spot rate (FX deal)
    Step 3 → swift-transmission:  Send SWIFT MT103 to correspondent bank
    Step 4 → credit-destination:  Credit EUR to beneficiary account

  Demo scenario: the beneficiary's correspondent bank is OFFLINE.
  SWIFT transmission (step 3) fails → saga rolls back:
    ↩ step 2: FX reversal  (sell back EUR, recover USD)
    ↩ step 1: Credit-back  (reverse GL debit, restore balance)

  Steps 1 and 2 are successfully compensated; step 4 never ran.

Execution flow:

  client._ref.send("wire-saga", type="wire.saga.started")
         │
         ▼  ✓ Step 1: debit-source       (JE-2026-XXXXXX)
         ▼  ✓ Step 2: fx-conversion      (FXD-2026-XXXXXX)
         ▼  ✗ Step 3: swift-transmission (OFFLINE — RJCT)
         │
         ├── ROLLBACK triggered
         ▼  ↩ Step 2 compensated: fx-reversal
         ▼  ↩ Step 1 compensated: credit-source
         │
         ▼  SagaOutcome(status=ROLLED_BACK) → client
"""

import asyncio

from pathlib import Path as _Path
from dotenv import load_dotenv

load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.context import AgentContext
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.patterns.saga import (
    SagaAgent,
    SagaOutcome,
    SagaStep,
    StepResult,
)
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

    # ── Step agents (each paired: executor + compensator) ─────────────────
    debit_source = SimpleAgent(
        "debit-source",
        prompt_name  = "bank.saga.debit_source.system",
        output_model = StepResult,
    )
    credit_source = SimpleAgent(          # compensator for debit-source
        "credit-source",
        prompt_name  = "bank.saga.credit_source.system",
        output_model = StepResult,
    )

    fx_conversion = SimpleAgent(
        "fx-conversion",
        prompt_name  = "bank.saga.fx_conversion.system",
        output_model = StepResult,
    )
    fx_reversal = SimpleAgent(            # compensator for fx-conversion
        "fx-reversal",
        prompt_name  = "bank.saga.fx_reversal.system",
        output_model = StepResult,
    )

    swift_transmission = SimpleAgent(
        "swift-transmission",
        prompt_name  = "bank.saga.swift_transmission.system",
        output_model = StepResult,
    )
    swift_cancellation = SimpleAgent(     # compensator for swift-transmission
        "swift-cancellation",             # (not invoked when MT103 was never sent)
        prompt_name  = "bank.saga.swift_cancellation.system",
        output_model = StepResult,
    )

    credit_destination = SimpleAgent(
        "credit-destination",
        prompt_name  = "bank.saga.credit_destination.system",
        output_model = StepResult,
    )
    debit_destination = SimpleAgent(      # compensator for credit-destination
        "debit-destination",              # (not invoked — step 4 never ran)
        prompt_name  = "bank.saga.debit_destination.system",
        output_model = StepResult,
    )

    # ── Saga definition ────────────────────────────────────────────────────
    wire_saga = SagaAgent(
        "wire-saga",
        steps = [
            SagaStep(
                name        = "debit-source",
                description = "Debit USD from payer account and record GL journal entry",
                executor    = debit_source,
                compensator = credit_source,
            ),
            SagaStep(
                name        = "fx-conversion",
                description = "Convert USD to EUR at today's spot rate via FX Trading Desk",
                executor    = fx_conversion,
                compensator = fx_reversal,
            ),
            SagaStep(
                name        = "swift-transmission",
                description = "Transmit SWIFT MT103 to JPMorgan Madrid as correspondent bank",
                executor    = swift_transmission,
                compensator = swift_cancellation,
            ),
            SagaStep(
                name        = "credit-destination",
                description = "Credit EUR to beneficiary account at Banco Santander Madrid",
                executor    = credit_destination,
                compensator = debit_destination,
            ),
        ],
        submit_type = "wire.saga.started",
        result_type = "wire.saga.completed",
        verbose     = True,
    )

    client = SimpleAgent("client", handler=client_handler)

    runtime.register(wire_saga)
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
