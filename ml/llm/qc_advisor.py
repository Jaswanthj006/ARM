"""QC advisor: turns structured defect findings into grounded operator advice.

Design (from the PS5 mentoring session): give Llama structured findings
(defect class, severity, location, confidence), not raw pixels. It explains
the defect, suggests actions and writes the log entry.

The same prompt is used on the phone (Kotlin port in the Android app) and here
on the Mac for testing. Llama only sees:
  * the findings from the vision model, and
  * the repair guide entries for the defect types that were found,
so its advice is grounded in both. Decisions are checked against fixed rules
afterwards; if Llama disagrees with the rules, the rules win and this is logged.
"""
import json
import re

# Short repair guide per defect type. Severity and decision rules are fixed;
# Llama writes the explanation and steps in plain language from these facts.
SOP = {
    "missing_hole": {
        "severity": "medium",
        "meaning": "A drilled hole for a component lead or via is absent.",
        "causes": ["drill skipped the hit", "broken or worn drill bit", "wrong drill file"],
        "fix": ["Confirm on the drawing that a hole is required at this pad",
                "Re-drill with the correct bit if the pad and annular ring are intact",
                "Check the drill bit wear log and replace the bit"],
    },
    "mouse_bite": {
        "severity": "medium",
        "meaning": "Small notches eaten into the edge of a copper trace.",
        "causes": ["uneven etching", "poor resist adhesion", "dirty panel before imaging"],
        "fix": ["Measure the remaining trace width against the minimum width",
                "If below minimum, reject; if above, mark for rework review",
                "Check resist lamination and panel cleaning"],
    },
    "open_circuit": {
        "severity": "high",
        "meaning": "A trace is broken, so the circuit is not connected.",
        "causes": ["over-etching", "scratch during handling", "void in copper plating"],
        "fix": ["Confirm the break with a continuity test",
                "Reject the board, or repair with a jumper wire only if the customer allows it",
                "Check etch time and handling fixtures on this panel"],
    },
    "short": {
        "severity": "high",
        "meaning": "Copper bridges two conductors that should be separate.",
        "causes": ["under-etching", "solder or copper splash", "dust under the photoresist"],
        "fix": ["Confirm the short with a continuity test",
                "Remove the bridge with a fine blade under magnification",
                "Re-test continuity and insulation, then re-inspect"],
    },
    "spur": {
        "severity": "low",
        "meaning": "A small copper protrusion sticking out of a trace.",
        "causes": ["resist residue", "under-etching at that spot"],
        "fix": ["Measure the gap from the spur to the nearest conductor",
                "If the gap is below the spacing rule, trim the spur",
                "Otherwise accept and log it"],
    },
    "spurious_copper": {
        "severity": "low",
        "meaning": "An unwanted island of copper left on the board.",
        "causes": ["incomplete etching", "contamination on the panel"],
        "fix": ["Check that it does not touch or come close to any conductor",
                "Remove it with a fine blade if it is near a conductor",
                "Check developer and etch uniformity across the panel"],
    },
    "missing_pad": {   # v3 detector class 6
        "severity": "high",
        "meaning": "A copper pad is missing, so the component cannot be soldered at that point.",
        "causes": ["over-etching", "poor resist adhesion lifting the pad", "wrong or damaged artwork"],
        "fix": ["Confirm on the drawing that a pad is required at this point",
                "Reject the board, or rebuild the pad with a pad repair kit only if the customer allows it",
                "Check resist adhesion and etch time on this panel"],
    },
}
SEV_RANK = {"low": 1, "medium": 2, "high": 3}

SYSTEM = (
    "You are a PCB quality-control assistant on a factory floor. "
    "The decision (pass, rework or reject) is already made by fixed rules; never change it. "
    "Explain it to the operator using only the findings and the repair guide. "
    "Do not invent defects, locations or numbers. "
    "Answer in exactly this format, one item per line, nothing else:\n"
    "SUMMARY: <one sentence naming the worst defect and its location>\n"
    "STEP: <short action>   (2 to 4 STEP lines, worst defect first)\n"
    "CAUSE: <likely cause, a few words>\n"
    "LOG: <one line for the inspection log>"
)
# The reply is started for the model so it always follows the format.
PREFILL = "SUMMARY:"
CLEAN_ADVICE = {"summary": "No defects found. The board can move to the next station.",
                "steps": ["Move the board to the next station"], "likely_cause": "",
                "log": "Visual inspection passed, no defects found."}


def rule_decision(findings):
    """Fixed decision rule. Llama explains the decision, it never makes it."""
    if not findings:
        return "pass"
    worst = max(SEV_RANK[SOP[f["type"]]["severity"]] for f in findings)
    return "reject" if worst == 3 else "rework"


def build_messages(board_id, findings):
    facts = [{
        "type": f["type"], "severity": SOP[f["type"]]["severity"],
        "location": f.get("location", "unknown"), "confidence": round(float(f["conf"]), 2),
    } for f in findings]
    facts.sort(key=lambda f: -SEV_RANK[f["severity"]])
    guide = {t: {"meaning": SOP[t]["meaning"], "causes": SOP[t]["causes"], "fix": SOP[t]["fix"]}
             for t in dict.fromkeys(f["type"] for f in facts)}
    user = (
        f"Board: {board_id}\n"
        f"Decision (fixed): {rule_decision(findings).upper()}\n"
        f"Findings, worst first: {json.dumps(facts)}\n"
        f"Repair guide: {json.dumps(guide)}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_advice(reply, findings):
    """Turn the labelled lines into a dict. The decision always comes from the rules."""
    if not findings:
        return dict(CLEAN_ADVICE, decision="pass"), []
    text = reply if reply.lstrip().upper().startswith("SUMMARY") else PREFILL + reply
    data = {"summary": "", "steps": [], "likely_cause": "", "log": ""}
    for line in text.splitlines():
        m = re.match(r"\s*(SUMMARY|STEP|CAUSE|LOG)\s*\d*\s*[:.-]\s*(.+)", line, re.I)
        if not m:
            continue
        key, val = m.group(1).upper(), m.group(2).strip().strip('"')
        if key == "STEP":
            if len(data["steps"]) < 4 and val not in data["steps"]:
                data["steps"].append(val)
        elif not data[{"SUMMARY": "summary", "CAUSE": "likely_cause", "LOG": "log"}[key]]:
            data[{"SUMMARY": "summary", "CAUSE": "likely_cause", "LOG": "log"}[key]] = val
    issues = [f"missing {k}" for k in ("summary", "steps", "likely_cause", "log") if not data[k]]
    if not data["summary"]:
        return None, ["no SUMMARY line"]
    data["decision"] = rule_decision(findings)
    known = {f["type"].replace("_", " ") for f in findings}
    for word in ("open circuit", "missing hole", "mouse bite", "spurious copper", "short", "spur"):
        said = data["summary"].lower().replace("_", " ")
        if word in said and not any(word in k for k in known):
            if word == "spur" and "spurious" in said:
                continue
            issues.append(f"mentions {word}, which was not found")
    return data, issues


# Sample findings used to test advice quality (board-level, like the phone sends).
SAMPLES = [
    ("BC-000101", []),
    ("BC-000102", [{"type": "short", "conf": 0.93, "location": "4 mm right of U2"}]),
    ("BC-000103", [{"type": "spur", "conf": 0.71, "location": "9 mm above U1"}]),
    ("BC-000104", [{"type": "open_circuit", "conf": 0.88, "location": "12 mm left of U3"},
                   {"type": "spur", "conf": 0.66, "location": "near U3"}]),
    ("BC-000105", [{"type": "missing_hole", "conf": 0.95, "location": "6 mm below U1"}]),
    ("BC-000106", [{"type": "mouse_bite", "conf": 0.79, "location": "8 mm right of U4"},
                   {"type": "mouse_bite", "conf": 0.74, "location": "10 mm right of U4"}]),
    ("BC-000107", [{"type": "spurious_copper", "conf": 0.69, "location": "board edge near J1"}]),
    ("BC-000108", [{"type": "short", "conf": 0.91, "location": "3 mm above U5"},
                   {"type": "missing_hole", "conf": 0.84, "location": "near U5"},
                   {"type": "spur", "conf": 0.62, "location": "7 mm left of U2"}]),
]
