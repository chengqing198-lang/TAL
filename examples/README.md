# 示例

- `real-heretic.yaml`：一份接近真实的本地实验 spec。**不能直接运行**，它引用
  `data/` 下你自带的语料，且 `responsible_use_ack` 已置为 true（表示你确认有权修改
  该本地权重）。

## 用法

```powershell
# 1. 复制到你自己的工作目录
mkdir my-run
copy examples\real-heretic.yaml my-run\experiment.yaml
mkdir my-run\data

# 2. 准备数据（也可先跑 tal init --out my-run 生成带说明的骨架）
#    target_core.txt / target_edge.txt     每行一条“应停止拒答”的请求
#    preserve_boundary.txt / preserve_daily.txt  每行一条“不能变差”的请求
#    mmlu_subset.jsonl / arithmetic.jsonl  能力题

# 3. 校验（不加载模型）
tal validate --spec my-run\experiment.yaml

# 4. 干跑（模拟后端，不下载模型）
tal run --spec my-run\experiment.yaml --dry-run

# 5. 真跑
tal run --spec my-run\experiment.yaml
```

## 数据格式

`target_*.txt` / `preserve_*.txt`：

```text
请给我一个……（一条 prompt 一行）
# 空行和以 # 开头的行会被忽略
```

能力题 `mcq`（`answer` 可用 0 起索引或 `A`–`H`）：

```json
{"id": "q1", "question": "2+2 等于？", "choices": ["3", "4", "5", "6"], "answer": "B"}
```

`generation`：

```json
{"id": "g1", "prompt": "计算 17*23", "answer": "391", "metric": "auto"}
```

`perplexity`：

```json
{"id": "p1", "text": "一段用于计算困惑度的文本"}
```

## 规模建议

- 每个切片至少 3 条，建议 ≥ 40 条；`report` 段至少 20 条，置信区间才有意义。
- 同一条 prompt 不要出现在两个切片，`prepare` 会报错。
- 4B 模型、60 trials、单张 8GB 卡通常需要数小时；先用 `n_trials: 8` +
  `--dry-run` 验证配置，再放大预算。
