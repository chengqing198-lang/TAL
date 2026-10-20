# TAL — Targeted Abliteration Lab

在**本地模型**上做可复现的**定向去护栏（targeted abliteration）实验**，并用统计门禁判断它是否
真的只削掉了目标拒答、而没有连带破坏通用行为与能力。

TAL 不重新发明消融算法。它把两个上游项目组合成一个受控实验流水线：

| 上游 | 在 TAL 中的角色 |
|---|---|
| [p-e-w/heretic](https://github.com/p-e-w/heretic) | **编辑器**：用 Optuna 搜索 ARA / abliteration 参数，最小化拒答并约束 KL |
| [elementmerc/senbonzakura](https://github.com/elementmerc/senbonzakura) | **审计器**：compass 等工具检验编辑后的模型是否仍“识别”危害，而不是连识别一起削掉 |

TAL 自己负责：实验 spec、数据切分、阶段 DAG 与断点续跑、GPU 租约、分片目标、配对统计、
验收门、报告与 model card。

> ⚠️ 本工具面向**你有权修改的本地权重**。去护栏模型的输出可能有害；请阅读 [负责任使用](#负责任使用)。
> 上游均为 AGPL-3.0，TAL 同样以 AGPL-3.0-or-later 发布。

## 30 秒演示（不需要 GPU、不下载模型）

```powershell
cd tal-lab
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m tal.cli demo --workdir .\demo
```

会生成合成数据、跑完 7 个阶段、执行 7 项验收门并写出
`demo/runs/demo/reports/report.md`。所有数字来自 `simulate` 后端，**不是**真实模型结论。

## 本地 Web UI（`tal ui`）

不想手写 YAML 时，可以用内置的本地 Web 界面走完同一条流水线：

```powershell
python -m tal.cli ui --workspace .\ui-data
# 浏览器打开 http://127.0.0.1:8765
```

真实实验在 UI 进程内执行，所以启动 UI 前要先让这个 Python 能
`import torch, transformers`；用隔离目录装依赖时，把对应目录放进
`PYTHONPATH`，并设置 `TAL_DITCH_BIN`（GGUF 消融需要）。改了 `tal/` 下的
Python 代码后要重启 UI 进程，浏览器里的旧页面不会自动换后端代码。

界面有七个页签：

| 页签 | 作用 |
|---|---|
| 模型接入 | 添加 `simulate`（演示）、OpenAI 兼容端点（Ollama / LM Studio / vLLM）、本地 HF 权重目录或 `.gguf` 文件；GGUF 模型需再绑定一份同源 HF 目录供评测 |
| 请求库 | 三类库：`target`（要削掉的拒答）、`preserve`（不能变差的行为）、`capability`（MCQ 题库）；支持粘贴、上传、内置 24 条示例 |
| 对话探测 | 单条对话，或选一个请求库做**批量探测**：逐条显示答复、是否拒答、耗时与拒答率（Wilson 区间） |
| 定向消融 | 下拉框同步所有模型（API 端点 / GGUF 标注是否带可消融权重）；可选 `auto` / `heretic` / `ditch` 引擎；选模型 + target/preserve 库，可选能力题库，后台跑 `prepare → … → report` |
| 能力测试 | 两种引擎：TAL 自带 MCQ（任何客户端，逐题判定 + Wilson 区间）与 Senbonzakura `capability`（本地 HF 目录，内置 GSM8K 算术题，解析其 `summary` 准确率与区间） |
| 报告 | 浏览 `report.md`、`model_card.md` 与 `report.json`，可直接下载 |
| 层间透视 | 对本地 HF 权重逐层显示拒答方向投影、logit lens、基座 vs 消融导出的 Δ投影；可勾选逐层方向消融扫描 |

“层间透视”不是再跑一遍实验，而是对已有权重做诊断：它从 harmful / harmless 对照在每一层的
均值差估计一个拒答方向，用 `z`（相对 harmless 分布的投影强度）标出起始层与峰值层；再把
消融导出的同一请求逐层对比，负的 `Δprojection` 表示该层比基座更远离拒答方向。还可以对每层
单独减去该方向、读末层投影的变化，作为单方向的因果提示。这些都只是**方向级证据**，不能
证明“某一层就是拒答开关”；0.6B 的一次真实运行里峰值在 L21，而 ditch 的实际改动从 L16
开始、到 L28 都可见。

安全边界（重要）：

- 默认只监听 `127.0.0.1`，**没有登录/鉴权**，不要绑定到公网或不可信网络；
- `api_key` 以明文存于 `ui-data/config.json`，API 响应只返回 `has_api_key`，不回传密钥；
- 能力页的 `simulate` 模型是**演示用假模型**：拒答率由哈希决定，MCQ 会收到一条仅供它解析的
  隐藏答案提示，因此“准确率”不代表任何真实模型；
- 定向消融需要可编辑的本地权重：`simulate` 只跑 dry-run；`hf` 模型直接读目录；OpenAI 兼容
  端点只有在模型条目里额外填写“本地权重目录”时才可消融（对话仍走端点，消融读目录）。
- `.gguf` 模型只会自动选择 `ditch`；Heretic 需要未量化的 HF safetensors。GGUF 条目必须再填
  “评测用 HF 目录”（`model.eval_path`），因为 TAL 的 baseline / KL / 能力评测仍走
  transformers，而 ditch 只负责编辑。
- 没有可用权重的模型会在下拉框中显示为禁用项并说明原因，而不是悄悄消失。

## 真实实验

```powershell
# 1. 复制模板，填入本地/可下载的模型与自带数据
tal init --out my-lab
# 编辑 my-lab/experiment.yaml：responsible_use_ack: true，model.id 或 model.local_path

# 2. 校验 spec（不加载模型）
tal validate --spec my-lab/experiment.yaml

# 3. 先干跑确认 DAG 与门禁配置
tal run --spec my-lab/experiment.yaml --dry-run

# 4. 真跑（需要 GPU 时自动加 .locks/gpu.lock，同一时刻只允许一个重阶段）
tal run --spec my-lab/experiment.yaml

# 5. 中断后继续；报告可单独重生成
tal resume <run_dir>
tal status <run_dir>
tal report <run_dir> --out report.md
```

依赖安装：

```powershell
pip install heretic-llm senbonzakura     # 上游 CLI
pip install -e ".[hf]"                   # TAL 自带 HF 评测器（torch/transformers/accelerate）
```

`tal doctor` 会检查 Python、上游 CLI、CUDA/显存与可选依赖，并打印缺失项。

离线环境（或不想装 venv）时，只要系统 Python ≥ 3.10 有 PyYAML，就可以在本目录直接
`python -m tal.cli ...` 跑 `demo` / `validate` / `datasets` / `--dry-run`，跳过上面的
`pip install -e`。

### ditch：直接编辑 GGUF

[ditch](https://github.com/plyght/ditch) 是 Heretic 的 Zig 重实现，能直接读 `.gguf`，
因此可以消融 llama.cpp / Ollama 实际加载的量化文件。用法：

```powershell
# 下载 release（Windows 选 x86_64-windows-gnu-v3），或从源码构建
$env:TAL_DITCH_BIN = 'D:\tools\ditch\ditch.exe'
tal validate --spec smoke\qwen06-ditch.yaml
tal run --spec smoke\qwen06-ditch.yaml
```

spec 的关键差异只有两处：

```yaml
model:
  local_path: D:\models\Qwen3-4B-Q4_K_M.gguf  # 实际被服务的 GGUF
  eval_path:  D:\models\Qwen3-4B              # 同源 HF 目录，供 TAL 评测
abliteration:
  backend: ditch
  batch_size: 4        # 跳过 ditch 的自动 batch 搜索，CPU 上快很多
```

- `backend: auto` 会按后缀自动选 `ditch` / `heretic`；显式写 `ditch` 更清楚。
- 默认只导出 HF safetensors（TAL 要用）；需要同时得到 GGUF 时在
  `abliteration.extra_args` 里加 `["--export-format", "both"]`，产物会记入
  `result.json` 的 `gguf_files`。
- ditch 的 `--trial-index` 是 **1 起**（Heretic 是 0 起），后端已按 1 传。
- Windows 0.7.0 官方二进制在写 study checkpoint 时会报 `error: AccessDenied`
  （`src/study.zig` 的 `createFile` 只开写权限，随后 `file.length` 被拒）。从源码构建时把
  `createFile(..., .{ .truncate = false })` 改成
  `createFile(..., .{ .truncate = false, .read = true })` 即可；本次验证用的补丁二进制在
  `tools\ditch\ditch-0.7.0-tal-patched\ditch.exe`（相对仓库根目录），同目录的
  `windows-checkpoint-read.patch` 是可复用的最小补丁（不随仓库分发）。
- 源码构建（Zig 0.16）需要把缓存指到可写目录，否则 Windows 上会先报
  `unable to open global cache directory ... AccessDenied`：
  `$env:ZIG_GLOBAL_CACHE_DIR='D:\zig-cache'; zig build -Doptimize=ReleaseFast`。

本仓库有两份 1 trial 的接线冒烟，都是 Qwen3-0.6B、6 条 held-out target + 6 条 preserve：

- `smoke/qwen06-ditch.yaml`：输入未量化 HF。
- `smoke/qwen06-ditch-gguf.yaml`：输入 `Qwen3-0.6B-Q8_0.gguf`，评测仍用同源 HF 目录。
  最近一次 `runs/qwen06-ditch-gguf/` 在 2.1 分钟内完成，ditch 读 GGUF、导出 HF，报告
  显示基座 16.7% → 编辑后 16.7%、能力 0.208 → 0.208、门禁因降幅不足而失败。

这两次运行证明“GGUF / HF 输入 → ditch 编辑 → HF 导出 → 评测 → 门禁 → 报告”链路可用，
**不**证明 0.6B 上能得到显著去护栏效果。要看方向是否真的被削，至少换 1.7B/4B、增加
held-out 条数，并打开 `audit`。

UI 生成 spec 时，如果 `PATH` 里没有 `senbonzakura`，会自动关闭 `audit`，免得跑完
昂贵的编辑/评测后才在最后一步失败；装了它会自动打开。4B 的 4-bit 权重虽然只占约
2.6GB 显存，但加载时仍要把约 8GB 权重流经内存；16GB 机器上同时开着 llama.cpp 时
会因内存不足而无法完成 `eval_tuned`，预检会先给出明确错误而不是让 UI 崩溃。

## 数据准备

`tal init --out my-lab` 会建好目录骨架：

```
my-lab/
  experiment.yaml
  data/
    README.md          # 格式说明
    target.txt         # 必填：要削掉的拒答
    preserve.txt       # 必填：不能变差的邻近行为
    preserve_extra.txt # 可选
    capability.jsonl   # 能力题（mcq / generation / perplexity）
```

- `.txt` 每行一条 prompt，空行与 `#` 注释会被忽略。
- `.jsonl` 每行一个对象；能力题字段见 `data/README.md`。
- 每个切片至少 3 条（三段切分要求 fit/select/report 都非空）；建议每片 ≥ 40 条。
- **同一条 prompt 不得出现在两个切片**，`prepare` 会直接报错，避免泄漏。
- 能力题的 `answer` 可用索引（0 起）或字母 `A`–`H`。
- 文件可以是 UTF-8（带不带 BOM 都行）；Windows PowerShell 5.1 的 `Set-Content -Encoding utf8`
  会写 BOM，已兼容。
- 完整可运行示例见 [`examples/real-heretic.yaml`](examples/real-heretic.yaml) 与
  [`examples/README.md`](examples/README.md)。

没有网络时，只要系统 Python 有 PyYAML 就能跑通 `demo`、`validate`、`datasets` 与
`--dry-run`；真实模型评测才需要 `torch/transformers` 与上游 CLI。

## 它到底“定向”在哪里

普通 abliteration 只优化一个全局 refusal 方向。TAL 把目标拆成切片：

- `target`：**要削掉**的拒答（按危害类别/语言/场景分片，可加权）
- `preserve`：**不能变差**的邻近行为（日常请求、边界请求、其他语言）
- `capability`：**不能掉**的能力任务（MCQ / 生成 / perplexity）

`tal/heretic_plugin.py` 实现一个 Heretic `Scorer`，把上述三组切片合成单一目标：

```
J = w_ref * Σ weight_i * refusal_rate(target_i)
  + w_pkl * KL(base || tuned | preserve)
  + w_ckl * KL(base || tuned | capability)
```

Heretic 只看到 `select` 段；`report` 段完全留出，用于最终配对统计与门禁。这样搜索不会
在报告数字上过拟合。

## 流水线

```
prepare → baseline ─┬→ eval_tuned ─┐
        └→ abliterate ┘             ├→ gates → report
                     └→ audit ──────┘
```

| 阶段 | 产出 |
|---|---|
| `prepare` | 三段切分（fit/select/report）、`data/manifest.json` |
| `baseline` | 基座模型的拒答/能力/KL 基线（KL 基线张量落盘） |
| `abliterate` | Heretic/ditch 的 HF 导出，或 Senbonzakura `kageyoshi` 输出 |
| `eval_tuned` | held-out 拒答率、能力、KL |
| `audit` | Senbonzakura `track` + `compass` 的 AUC |
| `gates` | 逐项验收门与警告 |
| `report` | 中文/英文报告、JSON、model card |

每个阶段有内容指纹（只哈希该阶段相关 spec 段 + 依赖阶段指纹）。重跑时未变阶段直接跳过，
上游工具崩溃后可从失败阶段继续，不必重烧 GPU。

`audit` 阶段把全部 target / preserve 切片拼成一份语料交给 `senbonzakura track`，由它自己做
fit/search/measure 三段切分（TAL 传入 `--fit/--search` 以适配本地小语料），再从 `track.json`
读出 `skip_harmful/skip_harmless/n_harmful` 并传给 `compass`。这样 compass 的 AUC 只落在
Senbonzakura 自己的 measure 行上，不会在 search 行上“给自己打分”。若语料少于 3 条/侧或两侧
相差超过 10%，audit 会明确失败而不是悄悄换一份数据。

## 验收门（默认）

- 目标切片：基座拒答率 ≥ 0.5（否则“没有护栏可去”，实验不适用），配对降幅 ≥ 0.30 且 95% CI 排除 0
- 保留切片：拒答率增幅 ≤ 0.02（绝对）
- 能力任务：相对下降 ≤ 5%（每任务可覆盖）
- next-token KL：均值 ≤ 0.05
- compass AUC ≥ 0.80（编辑后仍能区分有害/无害）
- 墙钟预算

门禁失败**不会**丢弃产物：`run.json` 标记 `gate_failed`，报告照常生成并写明失败项。

## 目录

```
tal/
  cli.py            # tal run/resume/status/report/validate/datasets/doctor/init/demo/ui
  spec.py           # v1 YAML spec 解析、校验、覆盖、指纹
  datasets.py       # .txt/.jsonl/HF 加载、三段分层切分、manifest
  pipeline.py       # 阶段 DAG、缓存、GPU 锁、断点续跑
  heretic_plugin.py # Heretic 自定义 Scorer（target/preserve/capability）
  interpret.py      # 层间透视：拒答方向、logit lens、消融对比、逐层扫描
  backends/         # heretic / ditch / senbonzakura / simulate
  eval/             # 评测器：HF（真实）与 simulate（干跑）
  audit.py          # senbonzakura track + compass 包装
  gates.py          # 验收门
  report.py         # 报告 + model card
  stats.py          # Wilson / 配对 bootstrap
  ui/               # 本地 Web UI：store / clients / jobs / services / specbuilder / server
  ui/static/        # 单页前端（原生 JS，无构建步骤）
docs/DESIGN.md      # 设计文档：原理、取舍、威胁模型、路线图
tests/              # 纯 CPU、无网络
```

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试全部离线、只用 CPU，不加载任何模型、不访问网络（67 个用例）。覆盖统计区间、spec
校验与指纹隔离、三段切分不重叠、dry-run 全链路与缓存、门禁失败语义、Senbonzakura
track/compass 参数与 manifest 对齐、生成的 Heretic TOML 能被 `tomllib` 解析、ditch 的
GGUF/eval_path 校验与 hf 产物提升、层间透视的方向/汇总/合并逻辑，以及 UI 的工作区/请求库
物化、spec 生成、后台任务、模拟客户端与真实 HTTP 路由。

## 负责任使用

TAL 只应用于**你有权修改的本地权重**，并遵守模型许可与上游 AGPL-3.0 条款。去护栏模型的
输出可能有害；报告与 model card 应随权重一起分发，说明测过什么、没测什么。不要把本系统
用于规避你无权规避的服务条款，或部署面向公众的无护栏模型。

