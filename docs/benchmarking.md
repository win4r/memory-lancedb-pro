# Retrieval Benchmark & Score Trace

## 定位

benchmark 是检索系统的**观测与调优工具**，不是质量门。

当前阶段：
- ✅ 能稳定运行
- ✅ 能输出可比对的 JSON
- ✅ 能展示每条结果的分数演化
- ⚠️ 不适合直接作为 CI 阻断条件（语料太小）

## 架构背景

本插件上游使用 **Jina AI** 作为统一的 embedding + reranking provider：

- **Embedding**：默认使用 `jina-embeddings-v3`（或本地替代）生成向量
- **Reranking**：默认使用 `jina-reranker-v3` 做 cross-encoder 重排

关键影响：
- 向量维度取决于 Jina embedding model 的配置（上游默认 1024，可配置为 2048）
- Rerank 422 错误通常是 Jina reranker API 对输入格式的限制（documents 必须是 object 而非 string）
- benchmark 中出现的 `rerank` stage 反映 Jina reranker 的实际行为
- 当 reranker 不可用时，系统会 fallback 到本地 cosine similarity 重排

## 使用

### CLI 命令

```bash
# 人类可读输出（默认）
pnpm openclaw memory-pro benchmark

# 稳定 JSON 报告（适合 diff / 脚本比对）
pnpm openclaw memory-pro benchmark --json

# 每条 query 一行 JSON（适合流式处理）
pnpm openclaw memory-pro benchmark --jsonl

# 指定自定义 fixture 文件
pnpm openclaw memory-pro benchmark --fixtures ./my-fixtures.json

# 严格模式：gate 级别 fixture 失败时退出码为 2
pnpm openclaw memory-pro benchmark --strict
```

### 独立 Runner

```bash
node test/benchmark-runner.mjs
node test/benchmark-runner.mjs --json
node test/benchmark-runner.mjs --jsonl
node test/benchmark-runner.mjs --strict
```

## Fixture 格式

```json
{
  "id": "pref_editor",
  "category": "preference",
  "level": "baseline",
  "strict": false,
  "datasetAssumption": "small_corpus",
  "expectedBehavior": "说明性描述",
  "query": "用户喜欢什么编辑器",
  "expect": {
    "minResults": 1,
    "maxResults": 10,
    "top1Contains": "Neovim",
    "top1MustNotContain": "不相关的关键词",
    "top1MinScore": 0.5,
    "top1MaxScore": 0.95,
    "note": "备注信息，不影响判定"
  }
}
```

### Fixture 分级

| Level | 用途 | 判定 |
|-------|------|------|
| `smoke` | 只验证能跑通 | 不参与 pass/fail 统计，记为 informational |
| `baseline` | 趋势观察 | 参与统计，但不阻断 |
| `gate` | 严格质量门 | `--strict` 模式下失败会退出码 2 |

### 如何新增 Fixture

1. 编辑 `test/benchmark-fixtures.json`
2. 必填字段：`id`, `query`, `category`
3. 推荐填写：`level`, `datasetAssumption`, `expectedBehavior`
4. `expect` 中的断言按需添加

### 已知限制

- **小语料 negative 不稳定**：6 条记忆的库中，任何 query 都会有 cosine > 0 的结果。用 `top1MaxScore` 而不是 `maxResults: 0`
- **不是最终质量门**：当前适合趋势比较，不适合单次通过率判断
- **FTS 状态依赖会话**：每次 CLI 调用是独立进程，FTS 可能不可用

## Score Trail

`--debug` 模式下，每条结果显示分数演化：

```
1. [7de1a208] 用户喜欢 Neovim (75%, vector)
   scores: vector_search=67% → recency_boost=77% → importance_weight=75% → length_norm=75% → time_decay=75%
```

### ScoreStep 字段

| 字段 | 说明 |
|------|------|
| `stage` | 阶段名称 |
| `stageType` | `seed` / `transform` / `rerank` / `filter` |
| `scoreBefore` | 进入该阶段前的分数 |
| `score` | 该阶段后的分数 |
| `delta` | 分数变化量 |
| `reason` | 可选的变化原因 |

### 阶段分类

| stageType | 阶段 | 说明 |
|-----------|------|------|
| `seed` | `vector_search` / `fused` / `bm25_only` | 初始分数来源 |
| `transform` | `recency_boost` / `importance_weight` / `length_norm` / `time_decay` | 打分变换 |
| `rerank` | `rerank` | cross-encoder 或 cosine 重排 |
| `filter` | `hard_min_score:eliminated` | 被过滤淘汰的结果 |

### 如何解读

- **top1 为什么压过 top2**：比较两者的 `delta` 差异，特别是 `recency_boost` 和 `importance_weight`
- **某条结果消失了**：查看是否有 `hard_min_score:eliminated` 标记
- **rerank 的影响**：比较 `rerank` 阶段的 `scoreBefore` vs `score`

## 输出格式

### 文本格式（人类可读）

```
Retrieval Benchmark Report
======================================================================
Fixture source: test/benchmark-fixtures.json
Total: 7 queries

✔ [preference] pref_editor
  query: "用户喜欢什么编辑器"
  results: 6, latency: 234ms
  #1: [vector] 75% "用户喜欢 Neovim..."
  trail: vector_search=67% → recency_boost=77% → ...
```

### JSON 格式（程序可读）

```bash
pnpm openclaw memory-pro benchmark --json > baseline.json
# 改参数后
pnpm openclaw memory-pro benchmark --json > tuned.json
# diff 比较
diff baseline.json tuned.json
```

关键字段：
- `summary.gatePass` / `summary.gateFail`
- `summary.baselinePass` / `summary.baselineFail`
- `results[].scoreTrails[].trail[]`
- `results[].failureReasons[]`
