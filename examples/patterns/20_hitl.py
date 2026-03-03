"""
Pattern: Human-in-the-Loop (HITL) — Wire Transfer Enhanced Due Diligence
=========================================================================
Demonstrates the ``HITLAgent`` pattern: automated screening runs first, then
execution PAUSES for a mandatory human decision, and finally an executor agent
produces the final output incorporating the human's response.

HITL vs other patterns:

  Reflection        → automated critic scores output; no human involved
  Consensus         → multiple LLM panelists vote; no human veto
  Plan-and-Execute  → planner + executors; no mandatory pause
  HITL              → execution PAUSES at a human decision gate; the reviewer
                      can approve, reject, or modify; the LLM executor then
                      incorporates that decision into the final output

Use-case — Outgoing international wire with Enhanced Due Diligence (EDD):
  Bank policy requires any outgoing SWIFT wire transfer above $500 000 USD to
  a first-time beneficiary in a FATF-monitored jurisdiction to be reviewed
  and signed off by a human Compliance Officer before release.

  Automated pipeline:
    1. AML Screener    — checks beneficiary, jurisdiction, amount, customer
                         profile; returns structured risk assessment
    2. ⏸ HUMAN PAUSE  — Compliance Officer reviews findings and decides:
                         APPROVE / APPROVE_WITH_CONDITIONS / REJECT
    3. Wire Operations — if approved, generates the SWIFT MT103 release
                         instruction memo; if rejected, issues formal denial

Transfer under review:
  Customer    : Petroquímica del Caribe S.A.S.  (client since 2017)
  Account     : 060-47823910-9  (USD commercial account)
  Amount      : USD 875,000.00
  Beneficiary : Refinería Panameña Corp.
  Benef. Bank : Banco Nacional de Panamá  (SWIFT: BNPDPAPAXXXX)
  Purpose     : Purchase of industrial cracking unit components
                Invoice IND-2026-0847 dated 2026-02-20
  Correspondent: JPMorgan Chase, New York  (USD clearing)

Execution flow:

  client ──[wire.edd.submitted]──▶ wire-edd-review (HITLAgent)
                  │
           [__hitl.screen__]──▶ wire-edd-review._screener  ← AML screening LLM
                                        │
                                 [__hitl.review__]──▶ wire-edd-review._gate  ← PAUSE: human decides
                                                              │
                                                      [__hitl.execute__]──▶ wire-edd-review._executor  ← MT103 / denial LLM
                                                                                    │
                                                                         [wire.edd.processed]──▶ client
"""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()

from synaptum.agents.simple_agent import SimpleAgent
from synaptum.core.context import AgentContext
from synaptum.core.message import Message
from synaptum.core.runtime import AgentRuntime
from synaptum.messaging.in_memory_bus import InMemoryMessageBus
from synaptum.patterns.hitl import (
    HITLAgent,
    HumanReviewRequest,
    HumanReviewResponse,
)
from synaptum.prompts import FilePromptProvider


# ── Output schema ──────────────────────────────────────────────────────────────

class WireReleaseInstruction(BaseModel):
    """Final processing instruction produced after the human decision."""
    status: str = Field(
        description="APPROVED | APPROVED_WITH_CONDITIONS | REJECTED"
    )
    swift_ref: Optional[str] = Field(
        default=None,
        description="SWIFT reference (COLYYMMDDnnnn) if approved, else null.",
    )
    decision_basis: str = Field(
        description="One-paragraph explanation of the decision.",
    )
    pre_release_checklist: List[str] = Field(
        default_factory=list,
        description="Items that must be verified before funds are released.",
    )
    compliance_memo: str = Field(
        description="Formal compliance memo for the transfer file.",
    )
    customer_notification: str = Field(
        description="Text to be sent to the customer.",
    )
    next_review_date: Optional[str] = Field(
        default=None,
        description="ISO date for required follow-up review, if applicable.",
    )


# ── Mock review handler ────────────────────────────────────────────────────────
# In production this would open a web form, send a Slack message,
# or block on a terminal prompt via asyncio run_in_executor.
# Here we simulate a Compliance Officer reviewing and deciding.

async def mock_compliance_review(request: HumanReviewRequest) -> HumanReviewResponse:
    """
    Simulates a human Compliance Officer reviewing the AML screening findings
    and returning a structured decision.

    Replace this with an interactive handler:

        loop = asyncio.get_event_loop()
        decision = await loop.run_in_executor(
            None, input, "Decision [APPROVE/APPROVE_WITH_CONDITIONS/REJECT]: "
        )
    """
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    CYAN   = "\033[36m"
    GREEN  = "\033[32m"

    risk_colour = {
        "LOW": GREEN, "MEDIUM": YELLOW, "HIGH": RED, "CRITICAL": RED
    }.get(request.risk_level.upper(), YELLOW)

    print(f"\n{'═' * 68}")
    print(f"{BOLD}{CYAN}  ⏸  HUMAN REVIEW REQUIRED — COMPLIANCE DESK{RESET}")
    print(f"{'═' * 68}")
    print(f"  Task ID   : {BOLD}{request.task_id}{RESET}")
    print(f"  Risk Level: {risk_colour}{BOLD}{request.risk_level}{RESET}")
    print(f"\n{BOLD}  CASE SUMMARY{RESET}")
    print(f"  {request.summary}")
    print(f"\n{BOLD}  AUTOMATED SCREENING FINDINGS{RESET}")
    for line in request.automated_findings.split("\n"):
        print(f"  {line}")
    print(f"\n{BOLD}  QUESTION FOR REVIEWER{RESET}")
    print(f"  {request.question}")
    print(f"  Options: {' | '.join(request.options)}")
    print(f"{'─' * 68}")

    print(f"\n  {YELLOW}[Compliance Officer reviewing\u2026 (simulated 2 s)]{RESET}")
    await asyncio.sleep(2)

    decision  = "APPROVE_WITH_CONDITIONS"
    comments  = (
        "Petroquímica del Caribe is a known, long-standing customer with clean "
        "compliance history.  The beneficiary is a first-time counterparty in a "
        "FATF-monitored jurisdiction, which triggers our EDD policy.  I am approving "
        "subject to documentation requirements being met before release.  "
        "Operations must not release funds until all items in the pre-release "
        "checklist are verified and signed off by a second compliance officer."
    )
    conditions = [
        "Obtain original commercial invoice IND-2026-0847 (signed, stamped) before release",
        "Collect sworn declaration of economic purpose signed by the customer's legal representative",
        "Verify BNPDPAPAXXXX against SWIFT BIC registry; confirm no sanctions alerts on correspondent bank",
        "Apply mandatory 24-hour release hold from time of documentation receipt",
        "Flag for UIAF monitoring: submit ROS if beneficiary verification fails",
    ]

    print(f"\n  {GREEN}{BOLD}→ Compliance Officer decision: {decision}{RESET}")
    print(f"  Comments : {comments[:90]}…")
    print(f"  Conditions imposed: {len(conditions)} item(s)")
    print(f"{'═' * 68}\n")

    return HumanReviewResponse(
        decision=decision,
        comments=comments,
        conditions=conditions,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    bus           = InMemoryMessageBus()
    prompt_provider = FilePromptProvider("examples/prompts/hitl.yaml")
    runtime       = AgentRuntime(bus, prompts=prompt_provider)

    results: list = []

    async def client_handler(agent, message: Message, context: AgentContext) -> None:
        if message.type == "wire.edd.processed":
            results.append(message.payload)

    # ── HITL definition ────────────────────────────────────────────────
    # HITLAgent builds and registers all 3 internal agents automatically:
    #   wire-edd-review._screener  — AML screening LLM
    #   wire-edd-review._gate      — human pause (no LLM)
    #   wire-edd-review._executor  — final output LLM
    hitl = HITLAgent(
        "wire-edd-review",
        screener_prompt = "bank.hitl.aml_screener.system",
        executor_prompt = "bank.hitl.wire_operations.system",
        executor_model  = WireReleaseInstruction,
        review_handler  = mock_compliance_review,
        trigger_type    = "wire.edd.submitted",
        result_type     = "wire.edd.processed",
        verbose         = True,
    )

    client = SimpleAgent("client", handler=client_handler)

    runtime.register(hitl)
    runtime.register(client)

    await runtime.start(run_id="run-wire-edd-2026-031")

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║  EXAMPLE 20 — Human-in-the-Loop (HITL)                          ║")
    print("║  Wire Transfer EDD — Petroquímica del Caribe S.A.S.             ║")
    print("║  Amount: USD 875,000 | Beneficiary: Panama | First-time         ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    await client._ref.send(
        to      = "wire-edd-review",
        type    = "wire.edd.submitted",
        payload = {
            "request_id":           "WXF-2026-031-0089",
            "customer_name":        "Petroquímica del Caribe S.A.S.",
            "customer_account":     "060-47823910-9",
            "customer_since":       "2017",
            "amount_usd":           875_000.00,
            "beneficiary_name":     "Refinería Panameña Corp.",
            "beneficiary_account":  "PA43-0200-0001-2345-6789-0123-456",
            "beneficiary_bank":     "Banco Nacional de Panamá",
            "beneficiary_swift":    "BNPDPAPAXXXX",
            "beneficiary_country":  "Panama",
            "purpose": (
                "Purchase of industrial cracking unit components — "
                "Invoice IND-2026-0847 dated 2026-02-20.  "
                "Equipment destined for the Cartagena refinery expansion project."
            ),
            "invoice_reference":    "IND-2026-0847",
            "correspondent_bank":   "JPMorgan Chase New York (USD clearing)",
            "requested_value_date": "2026-03-05",
            "additional_notes": (
                "Customer confirms this is the first transaction with this supplier.  "
                "Potential for follow-on purchases of ~$2M over the next 18 months."
            ),
        },
        reply_to = "client",
    )

    await runtime.run_until_idle()
    await runtime.stop()

    if not results:
        print("No result received.")
        return

    raw = results[0]
    result = WireReleaseInstruction.model_validate(raw.get("result", raw))

    status_icon = {
        "APPROVED":                 "✅",
        "APPROVED_WITH_CONDITIONS": "⚠️",
        "REJECTED":                 "❌",
    }.get(result.status, "●")

    print(f"\n{'═' * 68}")
    print(f"  WIRE OPERATIONS — FINAL PROCESSING INSTRUCTION")
    print(f"{'═' * 68}")
    print(f"  Status       : {status_icon}  {result.status}")
    if result.swift_ref:
        print(f"  SWIFT Ref    : {result.swift_ref}")
    if result.next_review_date:
        print(f"  Review date  : {result.next_review_date}")

    print(f"\n  DECISION BASIS\n  {result.decision_basis}\n")

    if result.pre_release_checklist:
        print(f"  PRE-RELEASE CHECKLIST ({len(result.pre_release_checklist)} items)")
        for i, item in enumerate(result.pre_release_checklist, 1):
            print(f"  {i:2d}. {item}")

    print(f"\n  COMPLIANCE MEMO\n  {result.compliance_memo}\n")
    print(f"  CUSTOMER NOTIFICATION\n  {result.customer_notification}")
    print(f"{'═' * 68}\n")


if __name__ == "__main__":
    asyncio.run(main())
