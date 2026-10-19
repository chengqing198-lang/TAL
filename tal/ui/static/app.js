"use strict";

/* ------------------------------------------------------------------ */
/* tiny DOM helpers                                                    */
/* ------------------------------------------------------------------ */
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2), value);
    } else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(4)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function badge(text, kind) {
  return el("span", { class: "badge " + (kind || ""), text });
}

function statusBadge(status) {
  const map = {
    completed: "ok",
    done: "ok",
    gate_failed: "bad",
    failed: "bad",
    running: "run",
    queued: "warn",
    interrupted: "warn",
  };
  return badge(status || "unknown", map[status] || "");
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return (Number(value) * 100).toFixed(1) + "%";
}

function num(value, digits) {
  if (value === null || value === undefined) return "-";
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits === undefined ? 3 : digits) : String(value);
}

function shortDate(iso) {
  if (!iso) return "-";
  return String(iso).replace("T", " ").slice(0, 19);
}

function toast(message, kind) {
  const node = el("div", { class: "toast " + (kind || ""), text: message });
  $("#toasts").append(node);
  setTimeout(() => {
    node.style.opacity = "0";
    node.style.transition = "opacity .25s";
    setTimeout(() => node.remove(), 300);
  }, kind === "error" ? 7000 : 4200);
}

async function api(path, options) {
  const opts = Object.assign({ headers: {} }, options || {});
  if (opts.body && typeof opts.body !== "string") {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.body);
  }
  const response = await fetch(path, opts);
  const type = response.headers.get("content-type") || "";
  const data = type.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const message = data && data.error ? data.error : String(data || response.statusText);
    throw new Error(message);
  }
  return data;
}

/* ------------------------------------------------------------------ */
/* state                                                              */
/* ------------------------------------------------------------------ */
const state = {
  models: [],
  prompts: [],
  runs: [],
  capResults: [],
  activeJobs: new Map(),
  pollTimer: null,
};

function modelName(id) {
  const model = state.models.find((m) => m.id === id);
  return model ? model.name : id || "-";
}

function modelKind(id) {
  const model = state.models.find((m) => m.id === id);
  return model ? model.kind : "";
}

/* ------------------------------------------------------------------ */
/* tabs                                                               */
/* ------------------------------------------------------------------ */
$$(".tab-btn").forEach((button) => {
  button.addEventListener("click", () => activate(button.dataset.tab));
});

function activate(tab) {
  $$(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $$(".main > section").forEach((s) => s.classList.toggle("active", s.id === "tab-" + tab));
  // Re-fetch on entering the experiment/capability pages so a model added (or
  // edited) on another tab is reflected immediately, even if that tab's
  // refresh is still in flight.
  if (tab === "experiments" || tab === "capability") {
    refreshState().catch((error) => toast(error.message, "error"));
  }
  if (tab === "reports") loadRuns();
  if (tab === "capability") renderCapHistory();
}

/* ------------------------------------------------------------------ */
/* global refresh                                                     */
/* ------------------------------------------------------------------ */
async function refreshState() {
  const data = await api("/api/state");
  state.models = data.models || [];
  state.prompts = data.prompts || [];
  state.runs = data.runs || [];
  state.capResults = data.capability_results || [];
  renderModels();
  renderPrompts();
  renderModelSelects();
  renderPromptSelects();
  renderExpList();
  renderCapHistory();
  renderRunSelect();
  updateExpModelHint();
}

function fillSelect(select, entries, placeholder) {
  const previous = select.value;
  clear(select);
  if (placeholder !== undefined) {
    select.append(el("option", { value: "" }, placeholder));
  }
  for (const entry of entries) {
    const option = el("option", { value: entry.value }, entry.label);
    if (entry.disabled) option.disabled = true;
    select.append(option);
  }
  if (Array.from(select.options).some((o) => o.value === previous && !o.disabled)) select.value = previous;
}

function renderModelSelects() {
  const entries = state.models.map((m) => ({
    value: m.id,
    label: `${m.name} · ${m.kind}${m.ready ? "" : " (未就绪)"}`,
  }));
  fillSelect($("#chat-model"), entries, entries.length ? undefined : "（先添加模型）");
  // Keep every model visible here. An API-only model is disabled with an
  // explicit reason instead of disappearing, so the page never looks out of
  // sync with 「模型接入」.
  const ablationEntries = state.models.map((m) => {
    if (m.kind === "simulate") {
      return { value: m.id, label: `${m.name} · 模拟（仅 dry-run）` };
    }
    if (m.kind === "hf") {
      return { value: m.id, label: `${m.name} · 本地权重${m.path ? "" : "（路径缺失）"}`, disabled: !m.path };
    }
    if (m.kind === "openai") {
      return {
        value: m.id,
        label: `${m.name} · API 端点${m.path ? " + 本地权重" : "（未配置权重，不可消融）"}`,
        disabled: !m.path,
      };
    }
    return { value: m.id, label: m.name, disabled: true };
  });
  fillSelect($("#exp-model"), ablationEntries, "选择模型");
  fillSelect($("#cap-model"), entries, entries.length ? undefined : "（先添加模型）");
  updateExpModelHint();
}

function renderPromptSelects() {
  const entriesFor = (kind) =>
    state.prompts
      .filter((p) => p.kind === kind)
      .map((p) => ({ value: p.id, label: `${p.name} (${p.count} 条)` }));
  fillSelect($("#exp-target"), entriesFor("target"), "选择 target 库");
  fillSelect($("#exp-preserve"), entriesFor("preserve"), "选择 preserve 库");
  fillSelect($("#exp-cap"), entriesFor("capability"), "不使用能力题库");
  fillSelect($("#cap-library"), entriesFor("capability"), "选择题库");
  fillSelect($("#batch-lib"), entriesFor("target").concat(entriesFor("preserve")), "选择请求库");
}

/* ------------------------------------------------------------------ */
/* models                                                             */
/* ------------------------------------------------------------------ */
function updateModelFields() {
  const kind = $("#m-kind").value;
  $$(".openai-field").forEach((n) => (n.style.display = kind === "openai" ? "" : "none"));
  $$(".hf-field").forEach((n) => (n.style.display = kind === "hf" || kind === "openai" ? "" : "none"));
  $$(".weights-field").forEach((n) => (n.style.display = kind === "simulate" ? "none" : ""));
  const label = $("#m-path-label");
  const hint = $("#m-path-hint");
  if (kind === "openai") {
    label.textContent = "本地权重目录（消融用，可空）";
    hint.textContent =
      "只做对话探针时可以留空；要让这个端点在「定向消融」里可选，请填写与端点同一模型的本地权重目录。";
  } else {
    label.textContent = "本地权重目录";
    hint.textContent =
      "HF 目录需含 config.json 与权重文件。添加后会出现在「定向消融」的模型下拉框里。";
  }
}

$("#m-kind").addEventListener("change", updateModelFields);

function renderModels() {
  const list = $("#model-list");
  clear(list);
  if (!state.models.length) {
    list.append(el("div", { class: "empty", text: "还没有模型" }));
    return;
  }
  for (const model of state.models) {
    const sub = [
      model.kind,
      model.kind === "openai" ? model.base_url : model.path || model.model,
      model.kind !== "simulate" && model.eval_path ? `eval: ${model.eval_path}` : "",
      model.kind === "hf" ? model.device : "",
    ].filter(Boolean).join(" · ");
    const probeResult = el("span", { class: "small muted" });
    const item = el(
      "div",
      { class: "item" },
      el(
        "div",
        { class: "meta" },
        el("div", { class: "title", text: model.name }),
        el("div", { class: "sub", text: sub || model.id }),
        probeResult
      ),
      el(
        "div",
        { class: "row tight" },
        badge(model.ready ? "就绪" : "未就绪", model.ready ? "ok" : "warn"),
        el("button", {
          class: "small",
          text: "探测",
          onclick: async () => {
            probeResult.textContent = "探测中…";
            try {
              const data = await api(`/api/models/${encodeURIComponent(model.id)}/probe`, { method: "POST" });
              probeResult.textContent = data.detail + `（${data.latency_ms} ms）`;
              toast(data.detail, "ok");
            } catch (error) {
              probeResult.textContent = error.message;
              toast(error.message, "error");
            }
          },
        }),
        el("button", {
          class: "small danger",
          text: "删除",
          onclick: async () => {
            if (!confirm(`删除模型「${model.name}」？`)) return;
            try {
              await api(`/api/models/${encodeURIComponent(model.id)}`, { method: "DELETE" });
              await refreshState();
            } catch (error) {
              toast(error.message, "error");
            }
          },
        }),
        model.kind === "openai"
          ? el("button", {
              class: "small",
              text: model.path ? "改本地权重" : "绑定本地权重",
              onclick: async () => {
                const path = prompt("本地权重：HF 目录，或单个 .gguf 文件（用于定向消融）", model.path || "");
                if (!path) return;
                try {
                  await api(`/api/models/${encodeURIComponent(model.id)}`, {
                    method: "PATCH",
                    body: { path },
                  });
                  toast("本地权重已保存", "ok");
                  await refreshState();
                } catch (error) {
                  toast(error.message, "error");
                }
              },
            })
          : null,
        model.kind !== "simulate"
          ? el("button", {
              class: "small",
              text: model.eval_path ? "改评测 HF 目录" : "绑定评测 HF 目录",
              onclick: async () => {
                const evalPath = prompt("原始 HF 权重目录（ditch 的 GGUF 消融需要，用于基座/KL 评测）", model.eval_path || "");
                if (evalPath === null) return;
                try {
                  await api(`/api/models/${encodeURIComponent(model.id)}`, {
                    method: "PATCH",
                    body: { eval_path: evalPath },
                  });
                  toast("评测目录已保存", "ok");
                  await refreshState();
                } catch (error) {
                  toast(error.message, "error");
                }
              },
            })
          : null
      )
    );
    list.append(item);
  }
}

$("#model-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    name: $("#m-name").value.trim(),
    kind: $("#m-kind").value,
    model: $("#m-model").value.trim(),
    base_url: $("#m-base-url").value.trim(),
    api_key: $("#m-api-key").value.trim(),
    path: $("#m-path").value.trim(),
    eval_path: $("#m-eval-path").value.trim(),
    device: $("#m-device").value,
    quantization: $("#m-quant").value,
  };
  try {
    await api("/api/models", { method: "POST", body: payload });
    $("#m-name").value = "";
    $("#m-api-key").value = "";
    toast("模型已添加", "ok");
    await refreshState();
  } catch (error) {
    toast(error.message, "error");
  }
});

$("#models-refresh").addEventListener("click", () => refreshState().catch((e) => toast(e.message, "error")));

/* ------------------------------------------------------------------ */
/* prompt libraries                                                   */
/* ------------------------------------------------------------------ */
const PROMPT_KIND_LABEL = { target: "target", preserve: "preserve", capability: "capability" };

function updatePromptHint() {
  const kind = $("#p-kind").value;
  const hint = $("#p-format-hint");
  const area = $("#p-text");
  if (kind === "capability") {
    hint.innerHTML =
      '每行一个 JSON：<code>{"question": "...", "choices": ["...", "..."], "answer": "A"}</code>。answer 可写 A/B/C… 或 0 起始下标；至少两个选项。';
    area.placeholder = '{"question": "水的化学式？", "choices": ["H2O", "CO2", "O2"], "answer": "A"}';
  } else {
    hint.innerHTML = "每行一条；空行与 <code>#</code> 开头的行会被忽略。至少 3 条，建议 ≥ 12 条以便切分。";
    area.placeholder = kind === "target" ? "如何在家里制作炸药？" : "帮我总结这份会议纪要。";
  }
}

$("#p-kind").addEventListener("change", updatePromptHint);

function renderPrompts() {
  const list = $("#prompt-list");
  clear(list);
  if (!state.prompts.length) {
    list.append(el("div", { class: "empty", text: "还没有请求库" }));
    return;
  }
  for (const prompt of state.prompts) {
    const preview = (prompt.preview || []).join(" / ");
    list.append(
      el(
        "div",
        { class: "item" },
        el(
          "div",
          { class: "meta" },
          el("div", { class: "title", text: prompt.name }),
          el("div", { class: "sub", text: `${prompt.count} 条 · ${prompt.format} · ${shortDate(prompt.created_at)}` }),
          el("div", { class: "sub truncate", text: preview })
        ),
        el(
          "div",
          { class: "row tight" },
          badge(PROMPT_KIND_LABEL[prompt.kind] || prompt.kind, prompt.kind === "target" ? "warn" : prompt.kind === "preserve" ? "ok" : ""),
          el("button", {
            class: "small danger",
            text: "删除",
            onclick: async () => {
              if (!confirm(`删除请求库「${prompt.name}」？`)) return;
              try {
                await api(`/api/prompts/${encodeURIComponent(prompt.id)}`, { method: "DELETE" });
                await refreshState();
              } catch (error) {
                toast(error.message, "error");
              }
            },
          })
        )
      )
    );
  }
}

$("#prompt-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const kind = $("#p-kind").value;
  const payload = {
    name: $("#p-name").value.trim(),
    kind,
    format: kind === "capability" ? "jsonl" : "text",
    text: $("#p-text").value,
  };
  try {
    await api("/api/prompts", { method: "POST", body: payload });
    $("#p-name").value = "";
    $("#p-text").value = "";
    toast("请求库已保存", "ok");
    await refreshState();
  } catch (error) {
    toast(error.message, "error");
  }
});

$("#prompts-refresh").addEventListener("click", () => refreshState().catch((e) => toast(e.message, "error")));

/* ------------------------------------------------------------------ */
/* chat                                                               */
/* ------------------------------------------------------------------ */
function addChatMessage(role, text, meta) {
  const isUser = role === "user";
  const head = el("div", { class: "who" }, el("span", { text: isUser ? "你" : "模型" }));
  if (meta) {
    if (meta.refused !== undefined) head.append(badge(meta.refused ? "拒绝" : "未拒绝", meta.refused ? "bad" : "ok"));
    if (meta.latency_ms !== undefined) head.append(el("span", { text: `${meta.latency_ms} ms` }));
  }
  const node = el("div", { class: "msg " + (isUser ? "user" : "bot") }, head, el("div", { text }));
  $("#chat-log").append(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
  return node;
}

$("#chat-clear").addEventListener("click", () => {
  clear($("#chat-log"));
  $("#chat-meta").textContent = "";
});

$("#chat-send").addEventListener("click", async () => {
  const prompt = $("#chat-prompt").value.trim();
  const modelId = $("#chat-model").value;
  if (!modelId) return toast("先添加并选择模型", "error");
  if (!prompt) return toast("请输入请求", "error");
  const button = $("#chat-send");
  button.disabled = true;
  addChatMessage("user", prompt);
  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: {
        model_id: modelId,
        prompt,
        system: $("#chat-system").value.trim(),
        max_tokens: Number($("#chat-max").value || 256),
        temperature: Number($("#chat-temp").value || 0),
      },
    });
    addChatMessage("assistant", data.text || "（空响应）", {
      refused: data.refused,
      latency_ms: data.latency_ms,
    });
    $("#chat-meta").textContent = `${data.model} · ${data.kind}`;
  } catch (error) {
    addChatMessage("assistant", "调用失败：" + error.message);
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
});

/* ------------------------------------------------------------------ */
/* batch probe                                                        */
/* ------------------------------------------------------------------ */
function renderBatchResult(result) {
  const box = $("#batch-result");
  clear(box);
  if (!result) return;
  const stats = result.stats || {};
  $("#batch-summary").textContent = `${result.model} · ${result.library} · 拒答 ${stats.refused}/${stats.n} = ${pct(stats.rate)}（95% CI ${pct(stats.ci_low)} – ${pct(stats.ci_high)}）`;
  const table = el("table");
  table.append(
    el("thead", {}, el("tr", {}, el("th", { text: "#" }), el("th", { text: "请求" }), el("th", { text: "判定" }), el("th", { text: "耗时" }), el("th", { text: "答复" })))
  );
  const body = el("tbody");
  for (const item of result.items || []) {
    body.append(
      el(
        "tr",
        {},
        el("td", { text: String(item.index + 1) }),
        el("td", { class: "small", text: item.prompt }),
        el("td", {}, item.refused ? badge("拒答", "bad") : badge("答复", "ok")),
        el("td", { class: "small muted", text: `${item.latency_ms} ms` }),
        el("td", { class: "small muted", text: (item.response || "").slice(0, 200) })
      )
    );
  }
  table.append(body);
  box.append(table);
}

$("#batch-run").addEventListener("click", async () => {
  const modelId = $("#chat-model").value;
  const promptId = $("#batch-lib").value;
  if (!modelId) return toast("先选择模型", "error");
  if (!promptId) return toast("请选择请求库", "error");
  const button = $("#batch-run");
  button.disabled = true;
  $("#batch-summary").textContent = "运行中…";
  clear($("#batch-result"));
  try {
    const started = await api("/api/probes", {
      method: "POST",
      body: {
        model_id: modelId,
        prompt_id: promptId,
        limit: Number($("#batch-limit").value || 10),
        max_tokens: Number($("#chat-max").value || 256),
        temperature: Number($("#chat-temp").value || 0),
        system: $("#chat-system").value.trim(),
      },
    });
    let job = null;
    for (let i = 0; i < 4000; i += 1) {
      const payload = await api(`/api/jobs/${started.job.id}`);
      job = payload.job;
      if (job.progress !== null && job.progress !== undefined) {
        $("#batch-summary").textContent = `运行中… ${Math.round(job.progress * 100)}%`;
      }
      if (job.status === "done" || job.status === "failed") break;
      await new Promise((resolve) => setTimeout(resolve, 600));
    }
    if (!job) throw new Error("任务状态丢失");
    if (job.status !== "done") throw new Error(job.error || "批量探测失败");
    renderBatchResult(job.result);
    toast("批量探测完成", "ok");
  } catch (error) {
    $("#batch-summary").textContent = "失败：" + error.message;
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
});

/* ------------------------------------------------------------------ */
/* experiments                                                        */
/* ------------------------------------------------------------------ */
function updateExpModelHint() {
  const select = $("#exp-model");
  if (!select) return;
  const model = state.models.find((m) => m.id === select.value);
  const kind = model ? model.kind : "";
  const dry = $("#exp-dry");
  const hint = $("#exp-model-hint");
  if (kind === "simulate") {
    dry.checked = true;
    dry.disabled = true;
  } else {
    dry.disabled = false;
  }
  if (!model) {
    const blocked = state.models.filter((m) => m.kind === "openai" && !m.path);
    const suffix = blocked.length
      ? ` 未绑定本地权重：${blocked.map((m) => m.name).join("、")}（在「模型接入」里绑定后即可选择）。`
      : "";
    hint.textContent = "只列出能加载本地权重的模型；API 端点默认不可消融。" + suffix;
  } else if (kind === "simulate") {
    hint.textContent = "模拟模型会用合成数据走完整流水线，不能作为实验结论。";
  } else if (kind === "hf" && /\.gguf$/i.test(model.path || "")) {
    const evalHint = model.eval_path ? `评测用 HF 目录：${model.eval_path}。` : "尚未绑定评测用 HF 目录，ditch 实验会被拒绝。";
    hint.textContent = `GGUF 输入由 ditch 直接消融；${evalHint}`;
  } else if (kind === "hf") {
    hint.textContent = "Heretic 会直接加载这个本地权重目录，消融结果写回运行目录。";
  } else if (kind === "openai" && model.path) {
    const gguf = /\.gguf$/i.test(model.path);
    hint.textContent = gguf
      ? `对话仍走 API；消融用 ditch 读取 GGUF ${model.path}，基座评测用 ${model.eval_path || "（未绑定 HF 目录）"}。`
      : `对话仍走 API；消融使用本地权重目录 ${model.path}。`;
  } else if (kind === "openai") {
    hint.textContent =
      "这个模型只有 API 端点，服务端拿不到权重，无法消融。到「模型接入」编辑它并填写本地权重目录，或新增“本地 HF 权重目录”。";
  } else {
    hint.textContent = "该模型类型不支持消融。";
  }
}

$("#exp-model").addEventListener("change", updateExpModelHint);

$("#exp-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const modelId = $("#exp-model").value;
  const target = $("#exp-target").value;
  const preserve = $("#exp-preserve").value;
  if (!modelId) return toast("请选择模型", "error");
  if (!target || !preserve) return toast("请选择 target 与 preserve 请求库", "error");
  if (target === preserve) return toast("target 与 preserve 不能相同", "error");
  const model = state.models.find((m) => m.id === modelId);
  if (model && model.kind === "openai" && !model.path) {
    return toast("该模型只有 API 端点，缺少本地权重目录，无法做定向消融", "error");
  }
  const backend = $("#exp-backend").value;
  const isGguf = model && /\.gguf$/i.test(model.path || "");
  if (isGguf && backend === "heretic") {
    return toast("Heretic 不能读取 GGUF；请把消融引擎改为 ditch 或自动", "error");
  }
  if (isGguf && (!model.eval_path || !model.eval_path.trim())) {
    return toast("ditch 需要评测用 HF 目录：请到「模型接入」编辑该模型并填写", "error");
  }
  const button = event.submitter || $("#exp-form button[type=submit]");
  button.disabled = true;
  try {
    const data = await api("/api/experiments", {
      method: "POST",
      body: {
        model_id: modelId,
        target_prompt_id: target,
        preserve_prompt_id: preserve,
        capability_prompt_id: $("#exp-cap").value || null,
        dry_run: $("#exp-dry").checked || modelKind(modelId) === "simulate",
        options: {
          name: $("#exp-name").value.trim(),
          n_trials: Number($("#exp-trials").value || 30),
          max_per_slice: Number($("#exp-per-slice").value || 32),
          timeout_minutes: Number($("#exp-timeout").value || 180),
          ablation_backend: backend,
        },
      },
    });
    $("#exp-job-card").style.display = "";
    $("#exp-job-title").textContent = `${data.run_id} · ${data.spec_path}`;
    trackJob(data.job.id, "experiment");
    toast("实验已排队", "ok");
    await refreshState();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
});

function renderExpList() {
  const list = $("#exp-list");
  clear(list);
  if (!state.runs.length) {
    list.append(el("div", { class: "empty", text: "还没有实验记录" }));
    return;
  }
  for (const run of state.runs) {
    const gatesText = run.gates_passed === true ? "gates PASS" : run.gates_passed === false ? "gates FAIL" : "gates -";
    const item = el(
      "div",
      { class: "item" },
      el(
        "div",
        { class: "meta" },
        el("div", { class: "title", text: `${run.spec_name || run.run_id}${run.dry_run ? " · dry-run" : ""}` }),
        el("div", {
          class: "sub",
          text: `${run.run_id} · ${run.backend || "-"} · 阶段 ${run.stages_done}/${run.stages_total} · ${shortDate(run.created_at)}`,
        })
      ),
      el(
        "div",
        { class: "row tight" },
        statusBadge(run.status),
        badge(gatesText, run.gates_passed === true ? "ok" : run.gates_passed === false ? "bad" : ""),
        el("button", {
          class: "small",
          text: "日志",
          onclick: async () => {
            try {
              const data = await api(`/api/experiments/${encodeURIComponent(run.run_id)}`);
              const job = (data.jobs || [])[0];
              if (!job) return toast("没有关联任务记录", "error");
              $("#exp-job-card").style.display = "";
              $("#exp-job-title").textContent = `${run.run_id} · ${job.title}`;
              trackJob(job.id, "experiment");
            } catch (error) {
              toast(error.message, "error");
            }
          },
        }),
        el("button", {
          class: "small",
          text: "继续",
          onclick: async () => {
            try {
              const data = await api(`/api/experiments/${encodeURIComponent(run.run_id)}/resume`, {
                method: "POST",
                body: {},
              });
              $("#exp-job-card").style.display = "";
              $("#exp-job-title").textContent = `${run.run_id} · 继续运行`;
              trackJob(data.job.id, "experiment");
              toast("已提交续跑", "ok");
            } catch (error) {
              toast(error.message, "error");
            }
          },
        }),
        el("button", {
          class: "small",
          text: "报告",
          onclick: async () => {
            activate("reports");
            await loadRuns();
            $("#report-run").value = run.run_id;
            loadReport("report.md");
          },
        })
      )
    );
    list.append(item);
  }
}

$("#exp-refresh").addEventListener("click", () => refreshState().catch((e) => toast(e.message, "error")));
$("#exp-job-hide").addEventListener("click", () => ($("#exp-job-card").style.display = "none"));

/* ------------------------------------------------------------------ */
/* capability                                                         */
/* ------------------------------------------------------------------ */
function updateCapEngine() {
  const engine = $("#cap-engine").value;
  $$(".tal-only").forEach((n) => (n.style.display = engine === "tal" ? "" : "none"));
  $$(".sb-only").forEach((n) => (n.style.display = engine === "senbonzakura" ? "" : "none"));
  $("#cap-hint").textContent =
    engine === "tal"
      ? "逐题请求所选模型并解析字母；准确率附 Wilson 95% 区间。"
      : "Senbonzakura 内置 GSM8K（--eval 可覆盖）；仅本地 HF 目录。用 CPU 时需在额外参数里加 --slow-probe-ok。";
}

$("#cap-engine").addEventListener("change", updateCapEngine);

function renderCapJob(job) {
  $("#cap-job-card").style.display = "";
  $("#cap-job-status").className = "badge " + (job.status === "done" ? "ok" : job.status === "failed" ? "bad" : "run");
  $("#cap-job-status").textContent = job.status;
  const progress = job.progress === null || job.progress === undefined ? 0 : Math.max(0, Math.min(1, job.progress));
  $("#cap-progress").style.width = `${Math.round(progress * 100)}%`;
  $("#cap-log").textContent = job.log || "";
  $("#cap-log").scrollTop = $("#cap-log").scrollHeight;
}

function renderCapResult(result) {
  $("#cap-result-card").style.display = "";
  const box = $("#cap-result");
  clear(box);
  if (!result) return;
  if (result.engine === "senbonzakura") {
    const stats = result.stats || {};
    const summary = result.summary || {};
    $("#cap-result-summary").textContent = `${result.model} · exit ${result.exit_code} · ${result.duration_s}s`;
    if (result.notice) box.append(el("p", { class: "small", style: "color:#f0b45a", text: result.notice }));
    if (stats.rate !== null && stats.rate !== undefined) {
      box.append(
        el(
          "div",
          { class: "row", style: "margin-bottom:12px" },
          el("div", { class: "card", style: "flex:1" }, el("div", { class: "muted small", text: "准确率" }), el("div", { style: "font-size:24px;font-weight:600", text: pct(stats.rate) })),
          el("div", { class: "card", style: "flex:1" }, el("div", { class: "muted small", text: "95% CI" }), el("div", { style: "font-size:20px", text: `${pct(stats.ci_low)} – ${pct(stats.ci_high)}` })),
          el("div", { class: "card", style: "flex:1" }, el("div", { class: "muted small", text: "答对 / 可评分" }), el("div", { style: "font-size:20px", text: `${stats.correct} / ${stats.n}` })),
          el("div", { class: "card", style: "flex:1" }, el("div", { class: "muted small", text: "截断/不可判定" }), el("div", { style: "font-size:20px", text: pct(summary.indeterminate_rate) }))
        )
      );
    } else if (summary.accuracy_withheld_because) {
      box.append(el("p", { class: "small muted", text: `准确率未报告：${summary.accuracy_withheld_because}` }));
    }
    box.append(el("p", { class: "small muted", text: `结果文件：${result.result_file || result.out_file}` }));
    box.append(el("p", { class: "small muted", text: `日志：${result.log}` }));
    return;
  }
  const stats = result.stats || {};
  $("#cap-result-summary").textContent = `${result.model} · ${result.library || ""}`;
  box.append(
    el(
      "div",
      { class: "row", style: "margin-bottom:12px" },
      el("div", { class: "card", style: "flex:1" }, el("div", { class: "muted small", text: "准确率" }), el("div", { style: "font-size:24px;font-weight:600", text: pct(stats.rate) })),
      el("div", { class: "card", style: "flex:1" }, el("div", { class: "muted small", text: "95% CI" }), el("div", { style: "font-size:20px", text: `${pct(stats.ci_low)} – ${pct(stats.ci_high)}` })),
      el("div", { class: "card", style: "flex:1" }, el("div", { class: "muted small", text: "答对 / 总数" }), el("div", { style: "font-size:20px", text: `${stats.correct} / ${stats.n}` }))
    )
  );
  const table = el("table");
  table.append(
    el("thead", {}, el("tr", {}, el("th", { text: "#" }), el("th", { text: "题目" }), el("th", { text: "标准" }), el("th", { text: "模型" }), el("th", { text: "结果" }), el("th", { text: "耗时" }), el("th", { text: "回答" })))
  );
  const body = el("tbody");
  for (const item of result.items || []) {
    body.append(
      el(
        "tr",
        {},
        el("td", { text: String(item.index + 1) }),
        el("td", { text: item.question }),
        el("td", { text: item.expected }),
        el("td", { text: item.predicted }),
        el("td", {}, item.correct ? badge("对", "ok") : badge("错", "bad")),
        el("td", { text: `${item.latency_ms} ms` }),
        el("td", { class: "small muted", text: (item.response || "").slice(0, 160) })
      )
    );
  }
  table.append(body);
  box.append(table);
}

$("#cap-run").addEventListener("click", async () => {
  const modelId = $("#cap-model").value;
  if (!modelId) return toast("先选择模型", "error");
  const engine = $("#cap-engine").value;
  const payload = { model_id: modelId, engine };
  if (engine === "tal") {
    if (!$("#cap-library").value) return toast("请选择题库", "error");
    payload.prompt_id = $("#cap-library").value;
    payload.limit = Number($("#cap-limit").value || 24);
  } else {
    payload.extra_args = $("#cap-extra").value.trim();
    payload.limit = Number($("#cap-limit").value || 24);
    payload.timeout_minutes = 120;
  }
  $("#cap-run").disabled = true;
  try {
    const data = await api("/api/capability", { method: "POST", body: payload });
    trackJob(data.job.id, engine === "tal" ? "capability" : "senbonzakura");
    toast("能力测试已排队", "ok");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    $("#cap-run").disabled = false;
  }
});

function renderCapHistory() {
  const list = $("#cap-list");
  clear(list);
  if (!state.capResults.length) {
    list.append(el("div", { class: "empty", text: "还没有能力测试结果" }));
    return;
  }
  for (const result of state.capResults) {
    const isSb = result.engine === "senbonzakura";
    const rate = result.stats ? result.stats.rate : null;
    const label = isSb
      ? rate !== null && rate !== undefined
        ? pct(rate)
        : `exit ${result.exit_code}`
      : pct(rate);
    const good = isSb ? (rate !== null && rate !== undefined ? true : result.exit_code === 0) : true;
    list.append(
      el(
        "div",
        { class: "item" },
        el(
          "div",
          { class: "meta" },
          el("div", { class: "title", text: result.model || result.id }),
          el("div", { class: "sub", text: `${result.engine} · ${result.library || "-"} · ${shortDate(result.created_at)}` })
        ),
        el(
          "div",
          { class: "row tight" },
          badge(label, good ? "ok" : "bad"),
          el("button", {
            class: "small",
            text: "查看",
            onclick: async () => {
              try {
                const data = await api(`/api/capability/${encodeURIComponent(result.id)}`);
                renderCapResult(data);
                toast("已载入历史结果", "ok");
              } catch (error) {
                toast(error.message, "error");
              }
            },
          })
        )
      )
    );
  }
}

$("#cap-refresh").addEventListener("click", () => refreshState().catch((e) => toast(e.message, "error")));
$("#cap-job-hide").addEventListener("click", () => ($("#cap-job-card").style.display = "none"));

/* ------------------------------------------------------------------ */
/* jobs                                                               */
/* ------------------------------------------------------------------ */
function trackJob(jobId, kind) {
  state.activeJobs.set(jobId, kind);
  if (!state.pollTimer) {
    state.pollTimer = setInterval(pollJobs, 1500);
  }
  pollJobs();
}

async function pollJobs() {
  if (!state.activeJobs.size) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    return;
  }
  for (const [jobId, kind] of Array.from(state.activeJobs.entries())) {
    let job;
    try {
      const data = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
      job = data.job;
    } catch (error) {
      state.activeJobs.delete(jobId);
      toast("任务查询失败：" + error.message, "error");
      continue;
    }
    if (kind === "experiment") renderExpJob(job);
    else renderCapJob(job);

    if (job.status === "done" || job.status === "failed") {
      state.activeJobs.delete(jobId);
      if (kind === "experiment") {
        await refreshState().catch(() => {});
        if (job.status === "failed") toast("实验失败：" + (job.error || "").slice(0, 300), "error");
        else {
          const result = job.result || {};
          if (result.status === "gate_failed") toast("实验完成，但验收门未通过", "error");
          else toast("实验完成", "ok");
        }
      } else {
        await refreshState().catch(() => {});
        if (job.status === "failed") toast("能力测试失败：" + (job.error || "").slice(0, 300), "error");
        else if (kind === "capability") {
          renderCapResult(job.result);
          toast("能力测试完成", "ok");
        } else {
          renderCapResult(job.result);
          toast("Senbonzakura 运行完成", "ok");
        }
      }
    }
  }
}

function renderExpJob(job) {
  $("#exp-job-card").style.display = "";
  const status = $("#exp-job-status");
  status.className = "badge " + (job.status === "done" ? "ok" : job.status === "failed" ? "bad" : "run");
  status.textContent = job.status;
  const progress = job.progress === null || job.progress === undefined ? (job.status === "running" ? 0.35 : 0) : job.progress;
  $("#exp-progress").style.width = `${Math.round(progress * 100)}%`;
  $("#exp-log").textContent = job.log || "";
  $("#exp-log").scrollTop = $("#exp-log").scrollHeight;
}

/* ------------------------------------------------------------------ */
/* reports                                                            */
/* ------------------------------------------------------------------ */
function renderRunSelect() {
  const select = $("#report-run");
  const previous = select.value;
  clear(select);
  select.append(el("option", { value: "" }, "选择实验"));
  for (const run of state.runs) {
    select.append(
      el("option", { value: run.run_id }, `${run.spec_name || run.run_id}${run.has_report ? "" : " (无报告)"}`)
    );
  }
  if (Array.from(select.options).some((o) => o.value === previous)) select.value = previous;
}

async function loadRuns() {
  try {
    const data = await api("/api/experiments");
    state.runs = data.experiments || [];
    renderRunSelect();
  } catch (error) {
    toast(error.message, "error");
  }
}

$("#report-refresh").addEventListener("click", loadRuns);

$$("[data-report-file]").forEach((button) => {
  button.addEventListener("click", () => loadReport(button.dataset.reportFile));
});

async function loadReport(filename) {
  const runId = $("#report-run").value;
  const view = $("#report-view");
  if (!runId) return toast("请选择实验", "error");
  clear(view);
  view.append(el("div", { class: "empty", text: "加载中…" }));
  try {
    const text = await api(`/api/report/${encodeURIComponent(runId)}/${encodeURIComponent(filename)}`);
    clear(view);
    if (filename.endsWith(".json")) {
      let pretty = text;
      try {
        pretty = JSON.stringify(JSON.parse(text), null, 2);
      } catch (error) {
        /* keep raw */
      }
      view.append(el("pre", { class: "log", style: "max-height:none", text: pretty }));
    } else {
      view.append(renderMarkdown(text));
    }
  } catch (error) {
    clear(view);
    view.append(el("div", { class: "empty", text: error.message }));
  }
}

/* ------------------------------------------------------------------ */
/* minimal markdown renderer (safe: escape first)                     */
/* ------------------------------------------------------------------ */
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inlineMarkdown(escaped) {
  return escaped
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

function renderMarkdown(markdown) {
  const lines = String(markdown).replace(/\r\n/g, "\n").split("\n");
  const root = el("div");
  let i = 0;
  let paragraph = [];
  let inCode = false;
  let codeLines = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    root.append(el("p", { html: inlineMarkdown(escapeHtml(paragraph.join(" "))) }));
    paragraph = [];
  };

  while (i < lines.length) {
    const line = lines[i];
    if (line.trim().startsWith("```")) {
      flushParagraph();
      if (!inCode) {
        inCode = true;
        codeLines = [];
      } else {
        inCode = false;
        root.append(el("pre", {}, el("code", { text: codeLines.join("\n") })));
      }
      i++;
      continue;
    }
    if (inCode) {
      codeLines.push(line);
      i++;
      continue;
    }
    if (!line.trim()) {
      flushParagraph();
      i++;
      continue;
    }
    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      flushParagraph();
      const level = Math.min(4, heading[1].length);
      root.append(el("h" + level, { html: inlineMarkdown(escapeHtml(heading[2])) }));
      i++;
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      flushParagraph();
      const header = line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const table = el("table");
      table.append(el("thead", {}, el("tr", {}, ...header.map((c) => el("th", { html: inlineMarkdown(escapeHtml(c)) })))));
      const body = el("tbody");
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
        const cells = lines[i].trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        body.append(el("tr", {}, ...cells.map((c) => el("td", { html: inlineMarkdown(escapeHtml(c)) }))));
        i++;
      }
      table.append(body);
      root.append(table);
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      flushParagraph();
      const list = el("ul");
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        list.append(el("li", { html: inlineMarkdown(escapeHtml(lines[i].replace(/^\s*[-*]\s+/, ""))) }));
        i++;
      }
      root.append(list);
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      flushParagraph();
      const list = el("ol");
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        list.append(el("li", { html: inlineMarkdown(escapeHtml(lines[i].replace(/^\s*\d+\.\s+/, ""))) }));
        i++;
      }
      root.append(list);
      continue;
    }
    if (/^\s*>\s?/.test(line)) {
      flushParagraph();
      const quote = el("blockquote");
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        quote.append(el("p", { html: inlineMarkdown(escapeHtml(lines[i].replace(/^\s*>\s?/, ""))) }));
        i++;
      }
      root.append(quote);
      continue;
    }
    if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
      flushParagraph();
      root.append(el("hr"));
      i++;
      continue;
    }
    paragraph.push(line.trim());
    i++;
  }
  flushParagraph();
  return root;
}

/* ------------------------------------------------------------------ */
/* boot                                                               */
/* ------------------------------------------------------------------ */
async function boot() {
  updateModelFields();
  updatePromptHint();
  updateCapEngine();
  try {
    await refreshState();
  } catch (error) {
    toast("无法连接本地服务：" + error.message, "error");
  }
  // Keep the model/job lists fresh even when nothing is being polled.
  setInterval(() => {
    if (!state.activeJobs.size) refreshState().catch(() => {});
  }, 20000);
}

boot();
