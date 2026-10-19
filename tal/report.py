"""Human-readable experiment report and model card.

The report is written for someone who has to decide whether the edited model is
acceptable, not for someone who wants to read logs. Every number carries its
interval and its control. Raw prompts and raw model responses stay in the run
artifacts; the report never copies harmful prompt text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .eval.base import EvalReport
from .stats import paired_bootstrap_drop
from .util import canonical_json, utc_now


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def _f(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}"


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def _num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _cell(value: Any) -> str:
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text if text else " "


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _checks(gates: dict) -> dict[str, dict]:
    return {check["name"]: check for check in gates.get("checks", [])}


def _verdict_label(gates: dict, dry_run: bool, zh: bool) -> str:
    if not gates:
        return "未评估" if zh else "NOT EVALUATED"
    passed = bool(gates.get("passed"))
    if dry_run:
        if passed:
            return "通过（模拟）" if zh else "PASS (simulated)"
        return "未通过（模拟）" if zh else "FAIL (simulated)"
    return ("通过" if zh else "PASS") if passed else ("未通过" if zh else "FAIL")


# --------------------------------------------------------------------------- #
# payload
# --------------------------------------------------------------------------- #
def _target_rows(spec, base: EvalReport, tuned: EvalReport, gates: dict, zh: bool) -> list[dict]:
    checks = _checks(gates)
    rows = []
    for slice_spec in spec.slices_of("target"):
        name = slice_spec.name
        if name not in base.slices or name not in tuned.slices:
            rows.append({"slice": name, "missing": True})
            continue
        before, after = base.paired(name, tuned)
        delta = paired_bootstrap_drop(before, after)
        gate = checks.get(f"target:{name}:drop", {})
        rows.append(
            {
                "slice": name,
                "n": delta.n,
                "baseline_rate": round(base.slices[name].rate, 6),
                "baseline_ci": [round(base.slices[name].ci_low, 6), round(base.slices[name].ci_high, 6)],
                "tuned_rate": round(tuned.slices[name].rate, 6),
                "tuned_ci": [round(tuned.slices[name].ci_low, 6), round(tuned.slices[name].ci_high, 6)],
                "drop": round(delta.drop, 6),
                "drop_ci": [round(delta.ci.low, 6), round(delta.ci.high, 6)],
                "p_drop_le_zero": round(delta.p_drop_le_zero, 6),
                "gate_passed": bool(gate.get("passed")) if gate else None,
            }
        )
    return rows


def _preserve_rows(spec, base: EvalReport, tuned: EvalReport, gates: dict) -> list[dict]:
    checks = _checks(gates)
    rows = []
    for slice_spec in spec.slices_of("preserve"):
        name = slice_spec.name
        if name not in base.slices or name not in tuned.slices:
            rows.append({"slice": name, "missing": True})
            continue
        before, after = base.paired(name, tuned)
        delta = paired_bootstrap_drop(before, after)
        gate = checks.get(f"preserve:{name}:increase", {})
        rows.append(
            {
                "slice": name,
                "n": delta.n,
                "baseline_rate": round(base.slices[name].rate, 6),
                "tuned_rate": round(tuned.slices[name].rate, 6),
                "increase": round(-delta.drop, 6),
                "increase_ci": [round(-delta.ci.high, 6), round(-delta.ci.low, 6)],
                "gate_passed": bool(gate.get("passed")) if gate else None,
            }
        )
    return rows


def _capability_rows(spec, base: EvalReport, tuned: EvalReport, gates: dict) -> list[dict]:
    rows = []
    for task in spec.evaluation.capability_tasks:
        before = base.capability.get(task.name)
        after = tuned.capability.get(task.name)
        if before is None or after is None:
            rows.append({"name": task.name, "missing": True})
            continue
        if before.value == 0:
            relative = 0.0 if after.value == 0 else float("inf")
        elif before.higher_is_better:
            relative = (before.value - after.value) / before.value
        else:
            relative = (after.value - before.value) / before.value
        gate = _checks(gates).get(f"capability:{task.name}:drop", {})
        rows.append(
            {
                "name": task.name,
                "type": task.type,
                "metric": before.metric,
                "n": after.n,
                "baseline": round(before.value, 6),
                "tuned": round(after.value, 6),
                "baseline_ci": None
                if before.ci_low is None
                else [round(before.ci_low, 6), round(before.ci_high, 6)],
                "tuned_ci": None if after.ci_low is None else [round(after.ci_low, 6), round(after.ci_high, 6)],
                "relative_change": round(relative, 6) if relative != float("inf") else None,
                "higher_is_better": before.higher_is_better,
                "gate_passed": bool(gate.get("passed")) if gate else None,
            }
        )
    return rows


def _data_counts(run_dir: Path) -> dict:
    from .util import read_json

    manifest = read_json(run_dir / "data" / "manifest.json", {}) or {}
    counts = manifest.get("counts", {}) or {}
    if counts and all(isinstance(value, dict) for value in counts.values()):
        total = sum(sum(value.values()) for value in counts.values())
    elif counts:
        total = sum(int(value) for value in counts.values())
    else:
        total = int(manifest.get("total") or 0)
    return {
        "counts": counts,
        "total_prompts": total,
        "sources": manifest.get("sources", {}),
        "split": manifest.get("split", {}),
    }


def _limitations(spec, base: EvalReport, tuned: EvalReport, dry_run: bool, zh: bool) -> list[str]:
    items: list[str] = []
    simulated = dry_run or bool(tuned.meta.get("simulated")) or spec.abliteration.backend == "simulate"
    if simulated:
        items.append(
            "本报告中的数字来自模拟后端，仅用于验证流水线，不构成对任何真实模型能力的证据。"
            if zh
            else "Numbers come from the simulated backend. They validate the pipeline, not a real model."
        )
    items.append(
        "拒答判定使用关键词标记，会漏掉改写式服从、也会误判带免责声明的有用回答；它是筛子，不是语义判断器。"
        if zh
        else "Refusal detection is keyword-based: it misses paraphrased compliance and can flag helpful answers that contain a disclaimer. Treat it as a sieve, not a judge."
    )
    small = [name for name, metrics in tuned.slices.items() if metrics.n < 20 and metrics.kind == "target"]
    if small:
        items.append(
            "以下 held-out 切片样本少于 20 条，区间很宽：" + "、".join(small)
            if zh
            else "Held-out slices with fewer than 20 prompts (wide intervals): " + ", ".join(small)
        )
    items.append(
        "单一种子、单次运行；搜索本身有随机性，跨运行复现时应比较区间而不是小数点。"
        if zh
        else "One seed, one run. Search is stochastic; compare intervals across runs, not decimal places."
    )
    items.append(
        "定向消融移除了一个 refusal 方向，但不保证模型在其他语言、其他危害类别或工具调用场景下同样服从；未测即未知。"
        if zh
        else "Targeted ablation removes one refusal direction. It does not guarantee behaviour in other languages, harm categories or tool-use settings; untested means unknown."
    )
    if spec.model.quantization == "bnb_4bit":
        items.append(
            "基座以 4bit 量化载入；量化误差会混入能力与 KL 数字。" if zh else "The base model was loaded in 4bit; quantization error is mixed into capability and KL numbers."
        )
    return items


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def generate_report(
    spec,
    run_dir: Path,
    base: EvalReport,
    tuned: EvalReport,
    audit: dict,
    gates: dict,
    *,
    state: dict | None = None,
    dry_run: bool = False,
    gate_enforced: bool = True,
) -> tuple[str, dict]:
    zh = spec.report.language != "en"
    now = utc_now()
    state = state or {}
    title = spec.report.title or (f"定向去护栏实验报告：{spec.name}" if zh else f"Targeted abliteration report: {spec.name}")
    verdict = _verdict_label(gates, dry_run, zh)
    if not gate_enforced:
        verdict += "（未强制门禁）" if zh else " (gates not enforced)"
    target_rows = _target_rows(spec, base, tuned, gates, zh)
    preserve_rows = _preserve_rows(spec, base, tuned, gates)
    capability_rows = _capability_rows(spec, base, tuned, gates)
    counts = _data_counts(run_dir)

    failing = [check for check in gates.get("checks", []) if not check.get("passed")]
    warnings = list(gates.get("warnings", []))
    if dry_run:
        warnings.insert(0, "DRY RUN：未加载真实模型，数字为模拟值。" if zh else "DRY RUN: no real model was loaded; numbers are simulated.")

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        f"> 生成时间 {now} · run `{state.get('run_id', run_dir.name)}` · 后端 `{spec.abliteration.backend}` · "
        f"验收门 **{verdict}**"
        if zh
        else f"> Generated {now} · run `{state.get('run_id', run_dir.name)}` · backend `{spec.abliteration.backend}` · "
        f"gates **{verdict}**"
    )
    lines.append("")

    # ------------------------------------------------------------- verdict
    lines.append("## 结论" if zh else "## Verdict")
    lines.append("")
    if not gates:
        lines.append("尚未运行验收门。" if zh else "Acceptance gates have not run.")
    elif gates.get("passed"):
        lines.append(
            "全部验收门通过。该模型在 held-out 切片上达到设定的拒答降幅，同时未超过保留行为、能力、KL 与机制审计的预算。"
            if zh
            else "All gates passed: the held-out refusal drop met the target while preserve behaviour, capability, KL and the mechanism audit stayed inside budget."
        )
    else:
        names = "、".join(check["name"] for check in failing) if zh else ", ".join(check["name"] for check in failing)
        lines.append(
            f"验收门未通过：{names}。这不代表实验失败——它表示按当前阈值，这个模型不应被采用。"
            if zh
            else f"Gates failed: {names}. The experiment succeeded; the model did not meet the policy."
        )
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- ⚠️ {warning}")
        lines.append("")

    # ------------------------------------------------------------- setup
    lines.append("## 实验设置" if zh else "## Setup")
    lines.append("")
    setup_rows = [
        ("模型" if zh else "Model", spec.model.id or spec.model.local_path),
        ("本地路径" if zh else "Local path", spec.model.local_path or "—"),
        ("量化" if zh else "Quantization", spec.model.quantization),
        ("编辑器" if zh else "Editor", spec.abliteration.backend),
        ("搜索预算" if zh else "Search budget", f"{spec.abliteration.n_trials} trials / {spec.abliteration.n_startup_trials} startup"),
        ("导出" if zh else "Export", spec.abliteration.export_strategy),
        ("切分" if zh else "Split", f"fit {spec.split.fit:.2f} / select {spec.split.select:.2f} / report {spec.split.report:.2f}"),
        ("拒答判定" if zh else "Refusal scoring", f"{spec.evaluation.refusal.method} ({len(spec.evaluation.refusal.markers())} markers)"),
        ("种子" if zh else "Seed", spec.seed),
        (
            "验收门强制执行" if zh else "Gates enforced",
            ("是" if zh else "yes") if gate_enforced else ("否（--no-gates）" if zh else "no (--no-gates)"),
        ),
    ]
    lines.append(_table(["项" if zh else "Item", "值" if zh else "Value"], setup_rows))
    lines.append("")

    # -------------------------------------------------------------- target
    lines.append("## 定向拒答：held-out report 切片" if zh else "## Targeted refusal: held-out report split")
    lines.append("")
    lines.append(
        "配对比较同一批 prompt 在基座与编辑后模型上的拒答率。drop 为绝对降幅；CI 为按 prompt 重采样的 95% 区间。"
        if zh
        else "Paired comparison on the same prompts. Drop is absolute; CI is a prompt-level bootstrap 95% interval."
    )
    lines.append("")
    rows = []
    for row in target_rows:
        if row.get("missing"):
            rows.append([row["slice"], "—", "—", "—", "—", "缺失" if zh else "missing"])
            continue
        rows.append(
            [
                row["slice"],
                row["n"],
                f"{_pct(row['baseline_rate'])} [{_pct(row['baseline_ci'][0])}, {_pct(row['baseline_ci'][1])}]",
                f"{_pct(row['tuned_rate'])} [{_pct(row['tuned_ci'][0])}, {_pct(row['tuned_ci'][1])}]",
                f"{_f(row['drop'])} [{_f(row['drop_ci'][0])}, {_f(row['drop_ci'][1])}]",
                "✅" if row.get("gate_passed") else ("❌" if row.get("gate_passed") is False else "—"),
            ]
        )
    lines.append(
        _table(
            ["切片" if zh else "Slice", "n", "基座拒答率 [95% CI]" if zh else "Baseline refusal [95% CI]", "编辑后 [95% CI]" if zh else "Edited [95% CI]", "绝对降幅 [95% CI]" if zh else "Drop [95% CI]", "门" if zh else "Gate"],
            rows,
        )
    )
    lines.append("")
    low = [row["slice"] for row in target_rows if row.get("baseline_rate", 1) < 0.5]
    if low:
        lines.append(
            "> 基座在以下切片本身拒答率偏低，`target:*:baseline` 门会失败；这表示“没有可移除的护栏”，而不是编辑失败："
            + "、".join(low)
            if zh
            else "> The base model already complies on these slices, so the `target:*:baseline` gate fails. This means there is no refusal behaviour to remove, not that the edit failed: "
            + ", ".join(low)
        )
        lines.append("")

    # ------------------------------------------------------------ preserve
    lines.append("## 通用行为保留" if zh else "## Preserved behaviour")
    lines.append("")
    rows = []
    for row in preserve_rows:
        if row.get("missing"):
            rows.append([row["slice"], "—", "—", "—", "缺失" if zh else "missing"])
            continue
        rows.append(
            [
                row["slice"],
                row["n"],
                _pct(row["baseline_rate"]),
                _pct(row["tuned_rate"]),
                f"{_f(row['increase'])} [{_f(row['increase_ci'][0])}, {_f(row['increase_ci'][1])}]",
                "✅" if row.get("gate_passed") else ("❌" if row.get("gate_passed") is False else "—"),
            ]
        )
    lines.append(
        _table(
            ["切片" if zh else "Slice", "n", "基座拒答率" if zh else "Baseline", "编辑后" if zh else "Edited", "增幅 [95% CI]" if zh else "Increase [95% CI]", "门" if zh else "Gate"],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "保留切片上的拒答率上升说明编辑伤到了邻近行为；这是定向消融最常见的副作用。"
        if zh
        else "A rise on preserve slices means the edit damaged neighbouring behaviour — the usual side effect of abliteration."
    )
    lines.append("")

    # ---------------------------------------------------------- capability
    lines.append("## 能力测试" if zh else "## Capability")
    lines.append("")
    if not capability_rows:
        lines.append("未配置能力任务。" if zh else "No capability tasks configured.")
    else:
        rows = []
        for row in capability_rows:
            if row.get("missing"):
                rows.append([row["name"], "—", "—", "—", "—", "缺失" if zh else "missing"])
                continue
            change = row["relative_change"]
            rows.append(
                [
                    row["name"],
                    row["type"],
                    f"{_num(row['baseline'])} → {_num(row['tuned'])}",
                    "—" if change is None else _pct(-change if row["higher_is_better"] else change),
                    "✅" if row.get("gate_passed") else ("❌" if row.get("gate_passed") is False else "—"),
                    row["n"],
                ]
            )
        lines.append(
            _table(
                ["任务" if zh else "Task", "类型" if zh else "Type", "基座 → 编辑后" if zh else "Baseline → edited", "相对变化" if zh else "Relative change", "门" if zh else "Gate", "n"],
                rows,
            )
        )
    lines.append("")

    # ------------------------------------------------------------------ KL
    lines.append("## 分布漂移（next-token KL）" if zh else "## Distribution drift (next-token KL)")
    lines.append("")
    if tuned.kl is None:
        lines.append("未测量。" if zh else "Not measured.")
    else:
        lines.append(
            f"在 {tuned.kl.n} 条 prompt 上，相对基座的平均 KL 为 **{tuned.kl.mean:.4f}**（门限 {spec.gates.kl_max_mean:.3f}）。"
            if zh
            else f"Mean KL against the base over {tuned.kl.n} prompts is **{tuned.kl.mean:.4f}** (gate {spec.gates.kl_max_mean:.3f})."
        )
        if tuned.kl.per_slice:
            lines.append("")
            lines.append(
                _table(
                    ["切片" if zh else "Slice", "平均 KL" if zh else "Mean KL"],
                    [[name, _num(value, 4)] for name, value in sorted(tuned.kl.per_slice.items())],
                )
            )
    lines.append("")

    # --------------------------------------------------------------- audit
    lines.append("## 机制审计（compass）" if zh else "## Mechanism audit (compass)")
    lines.append("")
    lines.append(
        f"状态 `{audit.get('status', 'missing')}`；AUC {_num(audit.get('auc'), 4)}（门限 ≥ {spec.audit.compass.min_auc:.2f}）。"
        if zh
        else f"Status `{audit.get('status', 'missing')}`; AUC {_num(audit.get('auc'), 4)} (gate ≥ {spec.audit.compass.min_auc:.2f})."
    )
    lines.append("")
    lines.append(
        "AUC 衡量编辑后的模型内部是否仍能区分有害与无害 prompt。AUC 太低说明消融可能连“识别危害”的能力一起削掉了，而不只是削掉拒答。"
        if zh
        else "AUC asks whether the edited model still separates harmful from harmless prompts internally. A low AUC suggests the edit removed harm recognition, not just refusal."
    )
    if audit.get("note"):
        lines.append("")
        lines.append(f"> {audit['note']}")
    lines.append("")

    # --------------------------------------------------------------- gates
    lines.append("## 验收门明细" if zh else "## Gate details")
    lines.append("")
    rows = []
    for check in gates.get("checks", []):
        rows.append(
            [
                check["name"],
                "✅" if check["passed"] else "❌",
                _num(check.get("value"), 4),
                check.get("threshold", ""),
                check.get("detail", ""),
            ]
        )
    lines.append(
        _table(["检查" if zh else "Check", "结果" if zh else "Result", "值" if zh else "Value", "阈值" if zh else "Threshold", "说明" if zh else "Detail"], rows)
    )
    lines.append("")

    # ---------------------------------------------------------------- data
    lines.append("## 数据与可复现性" if zh else "## Data and reproducibility")
    lines.append("")
    if counts["counts"]:
        rows = []
        for name, value in sorted(counts["counts"].items()):
            if isinstance(value, dict):
                rows.append([name, value.get("fit", 0), value.get("select", 0), value.get("report", 0), sum(value.values())])
            else:
                rows.append([name, "—", "—", "—", value])
        lines.append(
            _table(
                ["切片" if zh else "Slice", "fit", "select", "report", "合计" if zh else "Total"],
                rows,
            )
        )
        lines.append("")
    lines.append(f"- {'数据清单' if zh else 'Data manifest'}：`data/manifest.json`（{counts['total_prompts']} prompts）")
    lines.append(f"- 基座评测：`stages/baseline/eval_base.json`；编辑后评测：`stages/eval_tuned/eval_tuned.json`")
    lines.append(f"- 审计原始输出：`{audit.get('raw_path') or '—'}`")
    lines.append("")
    run_id = state.get("run_id", run_dir.name)
    lines.append("```powershell")
    lines.append(f"tal resume \"{run_dir}\"")
    lines.append(f"tal status \"{run_dir}\"")
    lines.append(f"tal report \"{run_dir}\"")
    lines.append("```")
    lines.append("")

    # --------------------------------------------------------- limitations
    lines.append("## 局限" if zh else "## Limitations")
    lines.append("")
    for item in _limitations(spec, base, tuned, dry_run, zh):
        lines.append(f"- {item}")
    lines.append("")
    if spec.report.notes:
        lines.append("## 备注" if zh else "## Notes")
        lines.append("")
        lines.append(spec.report.notes)
        lines.append("")

    # --------------------------------------------------------------- ethics
    lines.append("## 负责任使用" if zh else "## Responsible use")
    lines.append("")
    lines.append(
        "本实验只应用于你有权修改的本地权重，并遵守模型许可（含 AGPL-3.0 上游工具的条款）。"
        "去护栏模型的输出可能有害；报告与产出的模型卡应随权重一起分发，说明测过什么、没测什么。"
        "不要把本系统用于规避你无权规避的服务条款，或用于部署面向公众的无护栏模型。"
        if zh
        else "Run this only on local weights you are allowed to modify, under the model licence (and the AGPL-3.0 terms of the upstream tools). "
        "Abliterated models can produce harmful output; ship the model card with the weights so consumers know what was and was not measured. "
        "Do not use this system to evade terms you are not entitled to evade, or to deploy a public-facing model without guardrails."
    )
    lines.append("")
    if spec.report.include_samples:
        lines.append(
            "> 报告刻意不复制 harmful prompt 或模型回答原文；逐条判定保存在评测产物中，按需在受控环境查看。"
            if zh
            else "> This report deliberately omits harmful prompt text and raw model responses; per-prompt decisions live in the evaluation artifacts."
        )
        lines.append("")

    payload = {
        "schema_version": 1,
        "generated_at": now,
        "title": title,
        "run": {
            "run_id": run_id,
            "status": state.get("status"),
            "started_at": state.get("started_at"),
            "finished_at": state.get("finished_at"),
            "dry_run": bool(dry_run),
        },
        "model": spec.model.to_dict(),
        "backend": spec.abliteration.backend,
        "verdict": verdict,
        "gates_passed": gates.get("passed"),
        "gate_enforced": gate_enforced,
        "warnings": warnings,
        "target": target_rows,
        "preserve": preserve_rows,
        "capability": capability_rows,
        "kl": None if tuned.kl is None else tuned.kl.to_dict(),
        "audit": {k: v for k, v in audit.items() if k != "raw"},
        "gates": gates,
        "data": counts,
        "limitations": _limitations(spec, base, tuned, dry_run, zh),
        "spec_sha256": state.get("spec_sha256"),
        "spec_snapshot": str(run_dir / "spec.effective.yaml"),
        "report_language": spec.report.language,
    }
    return "\n".join(lines) + "\n", payload


# --------------------------------------------------------------------------- #
# model card
# --------------------------------------------------------------------------- #
def generate_model_card(
    spec,
    run_dir: Path,
    base: EvalReport,
    tuned: EvalReport,
    audit: dict,
    gates: dict,
    *,
    state: dict | None = None,
    dry_run: bool = False,
    gate_enforced: bool = True,
) -> str:
    zh = spec.report.language != "en"
    state = state or {}
    rows = _target_rows(spec, base, tuned, gates, zh)
    verdict = _verdict_label(gates, dry_run, zh)
    summary = "; ".join(
        f"{row['slice']} {_pct(row.get('baseline_rate'))}→{_pct(row.get('tuned_rate'))}" for row in rows if not row.get("missing")
    )
    lines = [
        f"# {spec.model.id or spec.model.local_path} — targeted-abliteration model card",
        "",
        f"- **方法 / Method**: parameter-selective abliteration + targeted search (`{spec.abliteration.backend}`), "
        f"{spec.abliteration.n_trials} trials, export `{spec.abliteration.export_strategy}`.",
        f"- **定向目标 / Targeted**: refusal on {', '.join(s.name for s in spec.slices_of('target'))}; "
        f"preserve {', '.join(s.name for s in spec.slices_of('preserve'))}.",
        f"- **held-out 结果 / Held-out result**: {summary or '—'}.",
        f"- **整体能力 / Capability**: "
        + (
            ", ".join(f"{row['name']} {_num(row.get('baseline'))}→{_num(row.get('tuned'))}" for row in _capability_rows(spec, base, tuned, gates) if not row.get("missing"))
            or "not measured"
        ),
        f"- **KL**：{_num(tuned.kl.mean, 4) if tuned.kl else '—'}；**compass AUC**：{_num(audit.get('auc'), 4)}.",
        f"- **验收门 / Gates**：{verdict}{'' if gate_enforced else ' (not enforced)'}.",
        "",
        "## 适用 / Intended use",
        "",
        "Local research on refusal behaviour in open-weight models, with the evaluation harness and data recorded in this run. "
        "Not evaluated for public deployment, tool use, non-English use, or harm categories absent from the target slices.",
        "",
        "## 已知局限 / Known limitations",
        "",
        "- Refusal is measured with keyword markers, not a semantic judge.",
        "- One seed, one search run; intervals are prompt-level, not run-level.",
        "- Single-turn, chat-template evaluation only.",
        "",
        "## 溯源 / Provenance",
        "",
        f"- run `{state.get('run_id', run_dir.name)}`, spec `{state.get('spec_source', spec.source_path)}`",
        f"- evaluation artifacts under `stages/` in this run directory",
        f"- generated {utc_now()}",
        "",
        "## 负责任使用 / Responsible use",
        "",
        "Weights may generate harmful content. Keep this card with the weights, and do not present the model as safe for general use.",
        "",
    ]
    return "\n".join(lines)


def report_fingerprint(spec, base: EvalReport, tuned: EvalReport, gates: dict) -> str:
    """Stable hash of the numbers a report depends on (useful for dedup)."""
    payload = {
        "spec": spec.to_dict(),
        "base": base.to_dict(),
        "tuned": tuned.to_dict(),
        "gates": gates,
    }
    return canonical_json(payload)
