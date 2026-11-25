package com.boardcheck.app

import org.json.JSONArray
import org.json.JSONObject

/**
 * QC advisor: turns structured defect findings into grounded operator advice.
 * Port of ml/llm/qc_advisor.py, so the phone and the Mac tests use the same prompt.
 *
 * Llama only sees the findings and the repair guide for those defect types.
 * The pass / rework / reject decision is made by fixed rules; Llama explains it.
 */
object QcAdvisor {

    data class Finding(val type: String, val conf: Float, val location: String)

    data class Sop(val severity: String, val meaning: String, val causes: List<String>, val fix: List<String>)

    val SOP = mapOf(
        "missing_hole" to Sop("medium", "A drilled hole for a component lead or via is absent.",
            listOf("drill skipped the hit", "broken or worn drill bit", "wrong drill file"),
            listOf("Confirm on the drawing that a hole is required at this pad",
                "Re-drill with the correct bit if the pad and annular ring are intact",
                "Check the drill bit wear log and replace the bit")),
        "mouse_bite" to Sop("medium", "Small notches eaten into the edge of a copper trace.",
            listOf("uneven etching", "poor resist adhesion", "dirty panel before imaging"),
            listOf("Measure the remaining trace width against the minimum width",
                "If below minimum, reject; if above, mark for rework review",
                "Check resist lamination and panel cleaning")),
        "open_circuit" to Sop("high", "A trace is broken, so the circuit is not connected.",
            listOf("over-etching", "scratch during handling", "void in copper plating"),
            listOf("Confirm the break with a continuity test",
                "Reject the board, or repair with a jumper wire only if the customer allows it",
                "Check etch time and handling fixtures on this panel")),
        "short" to Sop("high", "Copper bridges two conductors that should be separate.",
            listOf("under-etching", "solder or copper splash", "dust under the photoresist"),
            listOf("Confirm the short with a continuity test",
                "Remove the bridge with a fine blade under magnification",
                "Re-test continuity and insulation, then re-inspect")),
        "spur" to Sop("low", "A small copper protrusion sticking out of a trace.",
            listOf("resist residue", "under-etching at that spot"),
            listOf("Measure the gap from the spur to the nearest conductor",
                "If the gap is below the spacing rule, trim the spur",
                "Otherwise accept and log it")),
        "spurious_copper" to Sop("low", "An unwanted island of copper left on the board.",
            listOf("incomplete etching", "contamination on the panel"),
            listOf("Check that it does not touch or come close to any conductor",
                "Remove it with a fine blade if it is near a conductor",
                "Check developer and etch uniformity across the panel")),
        // v3: a copper pad that should be there is gone, so the component lead cannot be soldered.
        "missing_pad" to Sop("high", "A copper pad is missing, so the component cannot be soldered at that point.",
            listOf("over-etching", "poor resist adhesion lifting the pad", "wrong or damaged artwork"),
            listOf("Confirm on the drawing that a pad is required at this point",
                "Reject the board, or rebuild the pad with a pad repair kit only if the customer allows it",
                "Check resist adhesion and etch time on this panel")),
    )
    private val RANK = mapOf("low" to 1, "medium" to 2, "high" to 3)

    /** Operator-facing severity for one finding: Critical / Major / Minor (from the repair guide). */
    fun severity(type: String) = when (SOP[type]?.severity) { "high" -> "critical"; "medium" -> "major"; else -> "minor" }

    // Llama writes only 2 short lines (SUMMARY, CAUSE). The repair steps come straight from the
    // repair guide (SOP), so they are always correct and cost no generation time.
    private const val SYSTEM =
        "You are a PCB quality-control assistant. The decision is fixed by rules; never change it. " +
        "Use only the findings, guide and trend; invent nothing. Reply in exactly 2 short lines:\n" +
        "SUMMARY: <worst defect, where, and why it matters, one sentence>\nCAUSE: <most likely process cause, a few words>"
    const val PREFILL = "SUMMARY: The worst defect is"

    fun decision(findings: List<Finding>): String {
        if (findings.isEmpty()) return "pass"
        val worst = findings.maxOf { RANK[SOP[it.type]?.severity ?: "low"] ?: 1 }
        return if (worst == 3) "reject" else "rework"
    }

    /** Llama 3 chat prompt. The runner adds <|begin_of_text|> itself. trend: e.g. "spur on 4 of the last 5 boards". */
    fun prompt(boardId: String, findings: List<Finding>, trend: String? = null): String {
        val sorted = sortWorst(findings)
        val guide = sorted.map { it.type }.distinct().take(2).joinToString("\n") { t ->
            val s = SOP[t]!!; "$t: ${s.meaning} Causes: ${s.causes.joinToString(", ")}."
        }
        val list = sorted.take(3).joinToString("; ") { "${it.type} (${severity(it.type)}) at ${it.location}" } +
            if (sorted.size > 3) "; and ${sorted.size - 3} smaller findings" else ""
        val user = "Board $boardId. Decision: ${decision(findings).uppercase()}.\n" +
            "Findings: $list\nGuide:\n$guide\n" +
            (if (trend != null) "Trend: $trend. Blame the process step, not this board alone.\n" else "") +
            "Answer with the 2 labelled lines SUMMARY and CAUSE only."
        return SYSTEM_PREFIX + userTurn(user)
    }

    fun sortWorst(f: List<Finding>) = f.sortedWith(compareByDescending<Finding> { RANK[SOP[it.type]?.severity] ?: 1 }.thenByDescending { it.conf })

    /** Fixed part of every prompt. The phone prefills it once and reuses its KV cache (prefix cache). */
    val SYSTEM_PREFIX = "<|start_header_id|>system<|end_header_id|>\n\n$SYSTEM<|eot_id|>"

    /** The per-board part that follows SYSTEM_PREFIX. */
    fun boardPart(boardId: String, findings: List<Finding>, trend: String? = null): String =
        prompt(boardId, findings, trend).removePrefix(SYSTEM_PREFIX)

    private fun userTurn(user: String) = "<|start_header_id|>user<|end_header_id|>\n\n$user<|eot_id|>" +
        "<|start_header_id|>assistant<|end_header_id|>\n\n$PREFILL"

    /** source: llama / cache / rules; grounded: Llama's text only names defects and causes that are in the findings and guide. */
    data class Advice(val decision: String, val summary: String, val steps: List<String>, val cause: String, val log: String,
                      val source: String = "rules", val grounded: Boolean = true, val trend: String? = null) {
        fun text() = buildString {
            append(summary); steps.forEachIndexed { i, s -> append("\n${i + 1}. $s") }
            if (cause.isNotEmpty()) append("\nLikely cause: $cause")
            if (trend != null) append("\nTrend: $trend")
        }
        fun toJson(): JSONObject = JSONObject().put("decision", decision).put("summary", summary)
            .put("steps", JSONArray(steps)).put("likely_cause", cause).put("log", log)
            .put("source", source).put("grounded", grounded).put("trend", trend ?: JSONObject.NULL)
    }

    val CLEAN = Advice("pass", "No defects found. The board can move to the next station.",
        listOf("Move the board to the next station"), "", "Visual inspection passed, no defects found.")

    /** Repair steps from the guide: the worst defect's fixes, then the next type's first fix. */
    fun steps(findings: List<Finding>): List<String> {
        val types = sortWorst(findings).map { it.type }.distinct()
        val w = sortWorst(findings).first()
        val first = SOP[types[0]]!!.fix.mapIndexed { i, f -> if (i == 0) "$f (${types[0].replace('_', ' ')}, ${w.location})" else f }
        return (first + types.drop(1).take(2).map { t -> "${SOP[t]!!.fix.first()} (${t.replace('_', ' ')})" }).take(4)
    }

    private val TYPE_WORDS = mapOf("missing_hole" to listOf("missing hole", "hole"), "mouse_bite" to listOf("mouse bite", "notch", "bite"),
        "open_circuit" to listOf("open", "break", "broken"), "short" to listOf("short", "bridge"),
        "spur" to listOf("spur", "protrusion"), "spurious_copper" to listOf("spurious", "copper island", "island"),
        "missing_pad" to listOf("missing pad", "pad"))

    /** Parse Llama's 2 lines and check them against the findings (grounding guard). */
    fun parse(reply: String, findings: List<Finding>, trend: String? = null): Advice {
        if (findings.isEmpty()) return CLEAN
        val text = PREFILL + reply
        var summary = ""; var cause = ""
        val parts = text.split(Regex("(?=\\b(SUMMARY|STEP|CAUSE|LOG)\\s*\\d*\\s*[:.-])"))
        for (p in parts) {
            val m = Regex("^(SUMMARY|STEP|CAUSE|LOG)\\s*\\d*\\s*[:.-]\\s*(.+)", RegexOption.DOT_MATCHES_ALL).find(p.trim()) ?: continue
            val v = m.groupValues[2].trim().trim('"').lines().first().trim()
                .replace(Regex("^([-*•]|\\d+[.)])\\s*([-*•]|\\d+[.)])?\\s*"), "").trim()
            if (v.isEmpty()) continue
            when (m.groupValues[1]) {
                "SUMMARY" -> if (summary.isEmpty()) summary = v
                "CAUSE" -> if (cause.isEmpty()) cause = v
            }
        }
        val rule = ruleAdvice(findings, trend)
        // Grounding guard: the summary must name the worst defect and no defect type that was not found;
        // the cause must match a cause in the guide for a found type. Otherwise use the guide's text.
        val found = findings.map { it.type }.toSet()
        val worst = sortWorst(findings).first().type
        val lower = summary.lowercase()
        val sumOk = summary.length > 20 && TYPE_WORDS[worst]!!.any { it in lower } &&
            TYPE_WORDS.filterKeys { it !in found }.values.flatten().filter { it.length > 4 }.none { it in lower }
        val causeWords = found.flatMap { SOP[it]!!.causes }.flatMap { it.lowercase().split(" ", "-") }.filter { it.length > 4 }.toSet()
        val causeOk = cause.isNotEmpty() && causeWords.any { it in cause.lowercase() }
        return Advice(decision(findings), if (sumOk) summary else rule.summary, steps(findings),
            if (causeOk) cause else rule.cause, logLine(findings), "llama", sumOk && causeOk, trend)
    }

    private fun worst(f: List<Finding>) = sortWorst(f).first()

    /** Inspection log line, written from the findings (not by Llama), so it is always exact. */
    fun logLine(f: List<Finding>): String = "${decision(f).uppercase()}: " +
        f.groupingBy { it.type }.eachCount().entries.joinToString(", ") { "${it.value}x ${it.key.replace('_', ' ')}" }

    /** Advice straight from the repair guide; fallback when Llama fails or times out. */
    fun ruleAdvice(f: List<Finding>, trend: String? = null): Advice {
        val w = worst(f); val s = SOP[w.type]!!
        return Advice(decision(f), "${w.type.replace('_', ' ').replaceFirstChar { it.uppercase() }} (${severity(w.type)}) at the ${w.location}. ${s.meaning}",
            steps(f), s.causes.first(), logLine(f), "rules", true, trend)
    }

    /** True once CAUSE is written, so generation can stop early. */
    fun complete(reply: String): Boolean =
        reply.contains("CAUSE") && reply.substringAfterLast("CAUSE").let { it.contains('\n') || it.length > 50 }

    val SAMPLES = listOf(
        "BC-000102" to listOf(Finding("short", 0.93f, "4 mm right of U2")),
        "BC-000104" to listOf(Finding("open_circuit", 0.88f, "12 mm left of U3"), Finding("spur", 0.66f, "near U3")),
        "BC-000105" to listOf(Finding("missing_hole", 0.95f, "6 mm below U1")),
        "BC-000108" to listOf(Finding("short", 0.91f, "3 mm above U5"), Finding("missing_hole", 0.84f, "near U5"),
            Finding("spur", 0.62f, "7 mm left of U2")),
    )
}
