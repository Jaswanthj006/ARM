"""Run the exported Llama .pte (ExecuTorch runtime) on the sample findings.

Measures load time, generation time and tokens/second on this machine, and
checks each reply: valid JSON, all keys present, decision agrees with the rule.
Writes ml/results/advisor_<name>.json for the technical report.

Usage:  ml/.venv-et/bin/python ml/llm/test_advisor.py ml/llm/pte_8da4w
Note: these are Mac numbers. Phone numbers are measured in the Android app.
"""
import json
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer
from optimum.executorch import ExecuTorchModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from qc_advisor import PREFILL, SAMPLES, build_messages, parse_advice, rule_decision  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    pte_dir = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "pte_8da4w")
    tok = AutoTokenizer.from_pretrained(str(HERE / "Llama-3.2-1B-Instruct"))
    t = time.perf_counter()
    model = ExecuTorchModelForCausalLM.from_pretrained(str(pte_dir))
    load_s = time.perf_counter() - t
    pte = next(pte_dir.glob("*.pte"))
    print(f"loaded {pte.name} ({pte.stat().st_size / 1e9:.2f} GB) in {load_s:.1f} s")

    rows = []
    for board, findings in SAMPLES:
        if not findings:  # clean board: fixed message, Llama is not called
            data, issues = parse_advice("", findings)
            rows.append(dict(board=board, findings=[], rule="pass", skipped_llm=True, seconds=0, tokens_per_s=None,
                             valid_json=True, issues=issues, advice=data, raw=""))
            print(f"\n[{board}] clean board · Llama skipped"); continue
        prompt = tok.apply_chat_template(build_messages(board, findings), tokenize=False, add_generation_prompt=True) + PREFILL
        n_in = len(tok(prompt, add_special_tokens=False).input_ids)
        t = time.perf_counter()
        out = model.text_generation(tokenizer=tok, prompt=prompt, max_seq_len=n_in + 160)
        dt = time.perf_counter() - t
        # The runtime returns prompt + answer with special tokens stripped; the answer follows the last "assistant" header.
        reply = out.rsplit("assistant\n\n", 1)[-1].split("<|eot_id|>")[0].strip()
        n_out = len(tok(reply, add_special_tokens=False).input_ids)
        data, issues = parse_advice(reply, findings)
        rows.append(dict(board=board, findings=findings, rule=rule_decision(findings), prompt_tokens=n_in,
                         output_tokens=n_out, seconds=round(dt, 2), tokens_per_s=round(n_out / dt, 1) if dt else None,
                         valid_json=data is not None, issues=issues, advice=data, raw=reply))
        print(f"\n[{board}] {len(findings)} findings · {n_in} in / {n_out} out tokens · {dt:.1f} s")
        print("  issues:", issues or "none")
        print("  reply:", reply[:400].replace("\n", " "))

    ok = sum(r["valid_json"] for r in rows)
    clean = sum(r["valid_json"] and not r["issues"] for r in rows)
    summary = dict(model=pte.name, size_gb=round(pte.stat().st_size / 1e9, 2), load_s=round(load_s, 1),
                   samples=len(rows), valid_json=f"{ok}/{len(rows)}", no_issues=f"{clean}/{len(rows)}",
                   decision="set by rules, Llama explains it",
                   median_seconds=sorted(r["seconds"] for r in rows)[len(rows) // 2],
                   median_tokens_per_s=sorted(r["tokens_per_s"] or 0 for r in rows)[len(rows) // 2],
                   machine="Mac (Apple M4), CPU, ExecuTorch XNNPACK")
    out = HERE.parent / "results" / f"advisor_{pte_dir.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, runs=rows), indent=2))
    print("\nSUMMARY", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
