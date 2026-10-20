# TAL 开发工作日志

TAL（Targeted Abliteration Lab）是一个面向本地模型的定向消融实验控制面：
用 Heretic 或 ditch 编辑权重，用 Senbonzakura 或内置评测做审计，再通过
切片门禁决定这次编辑能不能交付。

> 本项目只用于自有/获授权模型的安全研究与能力评估。target 切片里的请求
> 用于测量拒答行为，不构成对任何实际滥用行为的指导。

## 2026-10-18：控制面与 UI

- 搭起 `tal-lab` 包：`spec.py`、`pipeline.py`、`backends/`、`eval/`、
  `report.py`、`gates.py`、`stats.py`。
- 定义固定 DAG：`prepare → baseline → abliterate → eval_tuned → audit →
  gates → report`；每阶段有指纹、缓存、产物清单和失败状态。
- 加入 paired bootstrap、Wilson 区间、target/preserve/capability 切片，
  以及 KL、审计和门禁。
- 用 stdlib `http.server` + 原生 JS 做了六页签本地 UI：模型接入、请求库、
  对话探测、定向消融、能力测试、报告；后台 Job 线程执行真实流水线。
- 加入 `simulate` 后端，离线也能跑完整 DAG，便于演示和测试。

## 2026-10-19：真实模型、ditch 与 4B

### Qwen3-0.6B

- 下载并校验 `Qwen/Qwen3-0.6B` HF 权重与 `Qwen3-0.6B-Q8_0.gguf`。
- 先用 Heretic 路线做真实 HF 消融，确认 spec、GPU 锁、插件 scorer、
  HF 导出、baseline/eval、报告都能串起来。
- 再跑 ditch：`qwen06-ditch.yaml`（HF 输入）和
  `qwen06-ditch-gguf.yaml`（GGUF 输入 + HF 评测目录）。
- 在 0.6B 上，target 拒答率很低（held-out 6 条约 16.7%），消融前后没有
  显著下降；门禁因此失败。这是模型本身“几乎没有可移除护栏”，不是
  ditch 链路失败。报告已明确写出该结论。

### ditch 接入

- 新增 `tal/backends/ditch.py`：生成无头命令、调用 ditch、提升 HF 产物、
  记录版本与 metadata。
- `tal/spec.py` 增加 `backend: ditch` 和 `model.eval_path`：GGUF 只用于
  编辑，基座/KL/能力评测仍读同源 HF 目录。
- UI 的消融引擎增加 `auto / heretic / ditch`；GGUF 自动选 ditch，
  Heretic 明确拒绝 GGUF。模型接入页增加“评测用 HF 目录”。
- Windows 上 ditch 0.7.0 的 `study.zig::appendLine` 以只写方式创建
  checkpoint 后调用 `file.length`，会报 `AccessDenied`；补
  `createFile(..., .{ .truncate = false, .read = true })` 后正常。
  修补版二进制和 patch 放在 `tools/ditch/ditch-0.7.0-tal-patched/`。
- ditch 的 `--trial-index` 从 **1** 开始（Heretic 从 0 开始），后端已按 1
  传；`batch_size` 走 `--batch-size`，避免自动 batch 搜索。
- 对 CUDA 模型默认追加 `--device auto`；大于 2GB 的 GGUF 默认
  `--batch-size 2`；target 切片带 `/no_think` 时追加
  `--response-prefix ""`，跳过代价很高的公共前缀探测。

### 评测与内存问题

- ditch 导出的 tokenizer 可能缺 `model_input_names`，fast tokenizer 会
  生成 `token_type_ids`，Qwen 的 `generate` 不接受；`eval/hf.py` 加载后
  从 `model_input_names` 中剔除它。
- Qwen3-4B 的 bf16 评测在 8GB 显存 + 16GB 内存上会触发原生
  `0xC0000005`，不是普通 Python OOM。`eval/hf.py` 增加加载前预检：
  按权重体积、空闲显存/内存估算，空间不足时抛 `StageError`，避免整个
  UI 进程被带崩。
- 4-bit 路线：`bitsandbytes 0.50.2` Windows wheel 可用，
  `load_in_4bit + nf4 + bfloat16` 能把 4B 装进约 2.65GB 显存；
  UI 模型可把 `quantization` 设为 `bnb_4bit`。
- 实测 4B：bnb 4-bit baseline 约 56 秒；`/no_think` 下 target 拒答
  6/6（100%）。ditch Vulkan 编辑 6m52s 成功导出两个 shard，但
  `eval_tuned` 加载约 8GB 合并权重时，16GB 内存（同时开着 llama.cpp）
  不足，run 被标记为 failed。结论：4B 的完整评测需要更多内存，或先停掉
  本地推理服务。

### Bug 修复

- UI 定向消融页曾只列 OpenAI 端点、不列本地模型；改为列出全部模型并
  标注是否可消融。
- 网页端绑定 `eval_path` 曾无效：8765 上残留的旧 UI 进程运行旧代码，
  `PATCH /api/models/<id>` 忽略 `eval_path`。重启 UI 后恢复；浏览器刷新
  不会重新加载服务端 Python，改后端代码后必须重启进程。
- 旧 UI 进程曾把 `config.json` 中文名写成乱码，已用 UTF-8 重写恢复。
- UI 默认开启 Senbonzakura 审计，但本机未安装；改为未显式指定时按
  `which("senbonzakura")` 自动决定，避免昂贵实验在最后一步失败。

## 2026-10-20：层间透视

- 新增 `tal/interpret.py`：对本地 HF 权重逐层取 last hidden state，用
  harmful / harmless 对照的均值差估计单位拒答方向；输出 `projection`、
  相对 harmless 分布的 `z`、logit-lens top token / 拒答 token 质量。
- 支持指定一个已完成的消融 run，重新加载其 `stages/abliterate/model`，
  逐层计算 `Δprojection`，标出消融影响最大的层。
- 可选 `sweep`：每次只 hook 一层并减去该层方向，读末层投影变化；这是
  单方向因果提示，不是完整电路证明。
- UI 新增“层间透视”页签：模型/对比 run/请求库/请求/system/对照数/
  sweep 开关，任务走 `JobManager`；结果用原生 SVG 画 z 折线与 Δ投影柱，
  历史结果存 `ui-data/interpret/layers-*.json`。
- 0.6B 实测（CPU，2 条对照）：29 层，`onset=L4`、`peak=L21`、
  `peak_z≈11.1`；对比 `qwen06-ditch-smoke2` 导出，`ablation_layer=L21`、
  `ablation_drop≈-6.46`，L16–L28 的 Δ 全为负、L1–L15 不变，与 ditch
  只改后段 attention/MLP 的行为一致。

## 测试与验证

- `python -m unittest discover -s tests`：**67 项全过**。
- 覆盖 spec/指纹/缓存、统计区间、门禁、Heretic TOML、ditch argv 与
  GGUF 校验、层间透视的路径/prompt/汇总/合并、UI store/specbuilder、
  评测内存预检。
- 真实运行留档在本地 `ui-data/runs/` 与 `runs/`（不随仓库发布，避免
  权重、报告和本地路径进入版本库）。

## 已知限制

- 本机是 16GB RAM + RTX 4060 8GB；4B 的 bnb 4-bit 评测在同时运行
  llama.cpp 时内存不足，尚未完成 `eval_tuned`。
- Heretic 与 Senbonzakura 的可选依赖未在当前机器安装；ditch 路径是
  本机验证最完整的真实编辑器。
- UI 无鉴权，只监听 loopback；不要暴露到公网。
- 仓库不包含模型权重、GGUF、运行产物或 API 密钥。

## 下一步

- 在内存更充足的机器上完成 4B 的 `eval_tuned` + 报告闭环。
- 安装并验证 Heretic / Senbonzakura 后，补齐三后端对照。
- 给 UI 增加“评测设备/量化”显式开关和更友好的内存预检提示。
- 把 ditch 的 Windows checkpoint 补丁提交到上游，减少本地 patch。
