"""Synthetic instruction dataset for enterprise support / IT-ops Q&A.

The dataset is 100% synthetic: tickets are composed from hand-written
templates, entity pools (applications, error codes, departments) and
procedural answer templates. No real tickets, users, or companies appear.

Each example teaches a strict enterprise response format:

    Ticket: <free-text ticket>
    Response:
    Category: <one of CATEGORIES>
    Priority: <P1..P4>
    Action: <short imperative runbook-style action>

Because the mapping ticket -> (category, priority, action) is rule-based,
held-out combinations (unseen application x symptom pairings) test genuine
generalisation rather than memorisation of full strings.

Splits are deterministic (seeded) and stratified by category.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = [
    "Access & Identity",
    "Network",
    "Hardware",
    "Software Defect",
    "Data & Reporting",
    "Change Request",
]

# symptom -> (category, priority, action template)
# The action template may reference {app}, {code}, {dept}.
SYMPTOMS: dict[str, tuple[str, str, str]] = {
    "cannot log in even after resetting the password": (
        "Access & Identity",
        "P2",
        "Unlock the account in the identity provider, force a password reset, and verify MFA enrollment for the user.",
    ),
    "account is locked after too many failed attempts": (
        "Access & Identity",
        "P3",
        "Unlock the account in the identity provider and coach the user on the lockout policy.",
    ),
    "new starter needs access to the shared mailbox": (
        "Access & Identity",
        "P4",
        "Raise an access request for {dept} approval, then grant mailbox delegation in the mail admin console.",
    ),
    "vpn drops every few minutes when working from home": (
        "Network",
        "P2",
        "Reinstall the VPN client, switch the tunnel to the backup gateway, and monitor the session for drops.",
    ),
    "office wifi is unreachable on the third floor": (
        "Network",
        "P1",
        "Dispatch network operations to inspect the third-floor access points and controller uplink.",
    ),
    "latency to the file server is above two seconds": (
        "Network",
        "P3",
        "Run a path trace to the file server, check switch utilisation, and open a capacity ticket if congestion is confirmed.",
    ),
    "laptop battery swells and the case will not close": (
        "Hardware",
        "P1",
        "Quarantine the device, stop using it immediately, and arrange a same-day replacement under the safety procedure.",
    ),
    "monitor flickers when connected through the dock": (
        "Hardware",
        "P3",
        "Swap the dock and cable with known-good spares and update the dock firmware.",
    ),
    "printer on level two jams on every second page": (
        "Hardware",
        "P3",
        "Clear the paper path, replace the pickup rollers, and log a maintenance call for the printer.",
    ),
    "application crashes with error {code} on save": (
        "Software Defect",
        "P2",
        "Collect the crash dump for {app}, attach error {code} to the defect backlog, and advise the save-draft workaround.",
    ),
    "dashboard totals do not match the source system": (
        "Data & Reporting",
        "P2",
        "Re-run the overnight extract for {app}, reconcile the totals, and notify {dept} of the corrected figures.",
    ),
    "monthly report is missing rows from last week": (
        "Data & Reporting",
        "P3",
        "Backfill the missing extract window, re-issue the report, and add a row-count check to the pipeline.",
    ),
    "export to spreadsheet produces an empty file": (
        "Data & Reporting",
        "P3",
        "Check the export job logs for {app}, re-run the export with the date filter removed, and verify the output row count.",
    ),
    "please add a new approval step to the purchase workflow": (
        "Change Request",
        "P4",
        "Log a change request for {dept} review, estimate the workflow change in {app}, and schedule it for the next release train.",
    ),
    "we need single sign-on enabled for the new tool": (
        "Change Request",
        "P3",
        "Register {app} in the identity provider, configure SAML with {dept} as owner, and pilot with the requester group.",
    ),
    "requesting an upgrade of the team workspace to the premium tier": (
        "Change Request",
        "P4",
        "Confirm the cost centre with {dept}, raise the licence change request, and apply the tier upgrade at the next billing cycle.",
    ),
}

APPS = [
    "CRM Portal",
    "HR Self Service",
    "Finance Hub",
    "Ticket Desk",
    "Inventory Manager",
    "Expense App",
    "Learning Portal",
    "Asset Tracker",
]

DEPARTMENTS = [
    "Finance",
    "People Team",
    "Sales Operations",
    "Customer Support",
    "Supply Chain",
    "Legal",
    "Engineering",
]

ERROR_CODES = ["E1042", "E2210", "E3307", "E4419", "E5021", "E6110"]

OPENERS = [
    "Hi team,",
    "Hello support,",
    "Urgent:",
    "Hi,",
    "Good morning,",
    "Flagging an issue:",
]

CLOSERS = [
    "Thanks in advance.",
    "Please advise.",
    "This is blocking our work.",
    "Appreciate a quick fix.",
    "Let me know the next steps.",
    "",
]


@dataclass
class Example:
    instruction: str  # the ticket text
    category: str
    priority: str
    action: str

    @property
    def prompt(self) -> str:
        return f"Ticket: {self.instruction}\nResponse:"

    @property
    def response(self) -> str:
        return (
            f"Category: {self.category}\n"
            f"Priority: {self.priority}\n"
            f"Action: {self.action}"
        )

    @property
    def text(self) -> str:
        """Full training sequence: prompt + response."""
        return f"{self.prompt}\n{self.response}"

    def to_json(self) -> dict:
        return {
            "instruction": self.instruction,
            "category": self.category,
            "priority": self.priority,
            "action": self.action,
            "prompt": self.prompt,
            "response": self.response,
            "text": self.text,
        }


def _compose_ticket(rng: random.Random, symptom: str, app: str, dept: str, code: str) -> str:
    body = symptom.format(app=app, code=code, dept=dept)
    # Many symptoms are statements about an application; weave the app in
    # naturally for a subset of them.
    if "{app}" not in symptom and rng.random() < 0.55:
        body = f"{body} in {app}"
    opener = rng.choice(OPENERS)
    closer = rng.choice(CLOSERS)
    parts = [opener, body.capitalize() if not body[0].isupper() else body]
    ticket = " ".join(p for p in parts if p)
    if closer:
        ticket = f"{ticket} {closer}"
    return ticket


def generate_examples(seed: int = 13, per_combination: int = 6) -> list[Example]:
    """Generate the full pool of examples (all symptom x app combinations,
    so held-out splits contain genuinely unseen combinations)."""
    rng = random.Random(seed)
    examples: list[Example] = []
    for symptom, (category, priority, action_tpl) in SYMPTOMS.items():
        for app in APPS:
            dept = rng.choice(DEPARTMENTS)
            code = rng.choice(ERROR_CODES)
            action = action_tpl.format(app=app, code=code, dept=dept)
            for _ in range(per_combination):
                ticket = _compose_ticket(rng, symptom, app, dept, code)
                examples.append(
                    Example(
                        instruction=ticket,
                        category=category,
                        priority=priority,
                        action=action,
                    )
                )
    rng.shuffle(examples)
    return examples


def split_examples(
    examples: list[Example],
    seed: int = 99,
    train_frac: float = 0.75,
    val_frac: float = 0.125,
) -> dict[str, list[Example]]:
    """Deterministic stratified split by category."""
    rng = random.Random(seed)
    by_cat: dict[str, list[Example]] = {}
    for ex in examples:
        by_cat.setdefault(ex.category, []).append(ex)
    splits: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    for cat, items in sorted(by_cat.items()):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])
    for split in splits.values():
        rng.shuffle(split)
    return splits


def build_dataset(out_dir: str | Path, seed: int = 13) -> dict[str, int]:
    """Generate, split, and write JSONL files + a data card. Returns counts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    examples = generate_examples(seed=seed)
    splits = split_examples(examples)
    counts: dict[str, int] = {}
    for name, items in splits.items():
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for ex in items:
                fh.write(json.dumps(ex.to_json(), ensure_ascii=False) + "\n")
        counts[name] = len(items)
    (out / "DATA_CARD.md").write_text(_data_card(counts), encoding="utf-8")
    return counts


def _data_card(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    return f"""# Data Card — Enterprise Support Instruction Dataset (Synthetic)

## Summary
A fully synthetic instruction-tuning dataset for enterprise IT support /
operations triage. Tickets are generated from hand-written symptom
templates, entity pools (applications, departments, error codes) and
rule-based runbook actions. **No real tickets, people, or organisations
are represented.**

## Format
Each JSONL record contains: `instruction` (ticket text), `category`,
`priority`, `action`, plus pre-rendered `prompt`, `response`, and `text`
(prompt + response) fields. The response format is fixed:

```
Category: <one of 6 categories>
Priority: <P1|P2|P3|P4>
Action: <runbook-style imperative sentence(s)>
```

## Splits
| Split | Examples |
|-------|----------|
| Train | {counts.get('train', 0)} |
| Validation | {counts.get('val', 0)} |
| Test | {counts.get('test', 0)} |
| **Total** | **{total}** |

Splits are stratified by category and deterministic (seeded). Because
examples are combinations of symptoms x applications, the held-out splits
contain symptom/application pairings never seen in training — the
evaluation therefore measures compositional generalisation, not recall.

## Categories
{chr(10).join(f'- {c}' for c in CATEGORIES)}

## Provenance & Limitations
- Generated by `src/lora_llm/data.py` (seed 13) from 16 symptom templates
  x 8 applications with surface-form variation (openers/closers).
- The label mapping is rule-based by construction; a model can in
  principle reach 100% — the point of the dataset is to demonstrate the
  LoRA fine-tuning workflow, not to benchmark frontier capability.
- Language is English only; style is intentionally narrow (enterprise
  support desk). Do not use for training production support systems.
"""


def load_split(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":  # pragma: no cover
    import sys

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/dataset"
    counts = build_dataset(out_dir)
    print(f"Wrote dataset to {out_dir}: {counts}")
