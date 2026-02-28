<div align="center">

# 🧠 memory-lancedb-pro · OpenClaw Plugin

**[OpenClaw](https://github.com/openclaw/openclaw) 增强型 LanceDB 长期记忆插件**

混合检索（Vector + BM25）· 跨编码器 Rerank · 多 Scope 隔离 · 管理 CLI

[![OpenClaw Plugin](https://img.shields.io/badge/OpenClaw-Plugin-blue)](https://github.com/openclaw/openclaw)
[![LanceDB](https://img.shields.io/badge/LanceDB-Vectorstore-orange)](https://lancedb.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](README.md) | **简体中文**

</div>

---

## 📺 视频教程

> **观看完整教程 — 涵盖安装、配置，以及混合检索的底层原理。**

[![YouTube Video](https://img.shields.io/badge/YouTube-立即观看-red?style=for-the-badge&logo=youtube)](https://youtu.be/MtukF1C8epQ)
🔗 **https://youtu.be/MtukF1C8epQ**

[![Bilibili Video](https://img.shields.io/badge/Bilibili-立即观看-00A1D6?style=for-the-badge&logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV1zUf2BGEgn/)
🔗 **https://www.bilibili.com/video/BV1zUf2BGEgn/**

---

## 为什么需要这个插件？

OpenClaw 内置的 `memory-lancedb` 插件仅提供基本的向量搜索。**memory-lancedb-pro** 在此基础上进行了全面升级：

| 功能 | 内置 `memory-lancedb` | **memory-lancedb-pro** |
|------|----------------------|----------------------|
| 向量搜索 | ✅ | ✅ |
| BM25 全文检索 | ❌ | ✅ |
| 混合融合（Vector + BM25） | ❌ | ✅ |
| 跨编码器 Rerank（Jina） | ❌ | ✅ |
| 时效性加成 | ❌ | ✅ |
| 时间衰减 | ❌ | ✅ |
| 长度归一化 | ❌ | ✅ |
| MMR 多样性去重 | ❌ | ✅ |
| 多 Scope 隔离 | ❌ | ✅ |
| 噪声过滤 | ❌ | ✅ |
| 自适应检索 | ❌ | ✅ |
| 管理 CLI | ❌ | ✅ |
| Session 记忆 | ❌ | ✅ |
| Task-aware Embedding | ❌ | ✅ |
| 任意 OpenAI 兼容 Embedding | 有限 | ✅（OpenAI、Gemini、Jina、Ollama 等） |

---

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                   index.ts (入口)                        │
│  插件注册 · 配置解析 · 生命周期钩子 · 自动捕获/回忆       │
└────────┬──────────┬──────────┬──────────┬───────────────┘
         │          │          │          │
    ┌────▼───┐ ┌────▼───┐ ┌───▼────┐ ┌──▼──────────┐
    │ store  │ │embedder│ │retriever│ │   scopes    │
    │ .ts    │ │ .ts    │ │ .ts    │ │    .ts      │
    └────────┘ └────────┘ └────────┘ └─────────────┘
         │                     │
    ┌────▼───┐           ┌─────▼──────────┐
    │migrate │           │noise-filter.ts │
    │ .ts    │           │adaptive-       │
    └────────┘           │retrieval.ts    │
                         └────────────────┘
    ┌─────────────┐   ┌──────────┐
    │  tools.ts   │   │  cli.ts  │
    │ (Agent API) │   │ (CLI)    │
    └─────────────┘   └──────────┘
```

### 文件说明

| 文件 | 用途 |
|------|------|
| `index.ts` | 插件入口。注册到 OpenClaw Plugin API，解析配置，挂载 `before_agent_start`（自动回忆）、`agent_end`（自动捕获）、`command:new`（Session 记忆）等钩子 |
| `openclaw.plugin.json` | 插件元数据 + 完整 JSON Schema 配置声明（含 `uiHints`） |
| `package.json` | NPM 包信息，依赖 `@lancedb/lancedb`、`openai`、`@sinclair/typebox` |
| `cli.ts` | CLI 命令实现：`memory list/search/stats/delete/delete-bulk/export/import/reembed/migrate` |
| `src/store.ts` | LanceDB 存储层。表创建 / FTS 索引 / Vector Search / BM25 Search / CRUD / 批量删除 / 统计 |
| `src/embedder.ts` | Embedding 抽象层。兼容 OpenAI API 的任意 Provider（OpenAI、Gemini、Jina、Ollama 等），支持 task-aware embedding（`taskQuery`/`taskPassage`） |
| `src/retriever.ts` | 混合检索引擎。Vector + BM25 → RRF 融合 → Jina Cross-Encoder Rerank → Recency Boost → Importance Weight → Length Norm → Time Decay → Hard Min Score → Noise Filter → MMR Diversity |
| `src/scopes.ts` | 多 Scope 访问控制。支持 `global`、`agent:<id>`、`custom:<name>`、`project:<id>`、`user:<id>` 等 Scope 模式 |
| `src/tools.ts` | Agent 工具定义：`memory_recall`、`memory_store`、`memory_forget`（核心）+ `memory_stats`、`memory_list`（管理） |
| `src/noise-filter.ts` | 噪声过滤器。过滤 Agent 拒绝回复、Meta 问题、寒暄等低质量记忆 |
| `src/adaptive-retrieval.ts` | 自适应检索。判断 query 是否需要触发记忆检索（跳过问候、命令、简单确认等） |
| `src/migrate.ts` | 迁移工具。从旧版 `memory-lancedb` 插件迁移数据到 Pro 版 |

---

## 核心特性

### 1. 混合检索 (Hybrid Retrieval)

```
Query → embedQuery() ─┐
                       ├─→ RRF 融合 → Rerank → 时效加成 → 重要性加权 → 过滤
Query → BM25 FTS ─────┘
```

- **向量搜索**: 语义相似度搜索（cosine distance via LanceDB ANN）
- **BM25 全文搜索**: 关键词精确匹配（LanceDB FTS 索引）
- **融合策略**: Vector score 为基础，BM25 命中给予 15% 加成（非传统 RRF，经过调优）
- **可配置权重**: `vectorWeight`、`bm25Weight`、`minScore`

### 2. 跨编码器 Rerank

- **Jina Reranker API**: `jina-reranker-v3`（5s 超时保护）
- **混合评分**: 60% cross-encoder score + 40% 原始融合分
- **降级策略**: API 失败时回退到 cosine similarity rerank

### 3. 多层评分管线

| 阶段 | 公式 | 效果 |
|------|------|------|
| **时效加成** | `exp(-ageDays / halfLife) * weight` | 新记忆分数更高（默认半衰期 14 天，权重 0.10） |
| **重要性加权** | `score *= (0.7 + 0.3 * importance)` | importance=1.0 → ×1.0，importance=0.5 → ×0.85 |
| **长度归一化** | `score *= 1 / (1 + 0.5 * log2(len/anchor))` | 防止长条目凭关键词密度霸占所有查询（锚点：500 字符） |
| **时间衰减** | `score *= 0.5 + 0.5 * exp(-ageDays / halfLife)` | 旧条目逐渐降权，下限 0.5×（60 天半衰期） |
| **硬最低分** | 低于阈值直接丢弃 | 移除不相关结果（默认 0.35） |
| **MMR 多样性** | cosine 相似度 > 0.85 → 降级 | 防止近似重复结果 |

### 4. 多 Scope 隔离

- **内置 Scope 模式**: `global`、`agent:<id>`、`custom:<name>`、`project:<id>`、`user:<id>`
- **Agent 级访问控制**: 通过 `scopes.agentAccess` 配置每个 Agent 可访问的 Scope
- **默认行为**: Agent 可访问 `global` + 自己的 `agent:<id>` Scope

### 5. 自适应检索

- 跳过不需要记忆的 query（问候、slash 命令、简单确认、emoji）
- 强制检索含记忆相关关键词的 query（"remember"、"之前"、"上次"等）
- 支持 CJK 字符的更低阈值（中文 6 字符 vs 英文 15 字符）

### 6. 噪声过滤

在自动捕获和工具存储阶段同时生效：
- 过滤 Agent 拒绝回复（"I don't have any information"）
- 过滤 Meta 问题（"do you remember"）
- 过滤寒暄（"hi"、"hello"、"HEARTBEAT"）

### 7. Session 记忆

- `/new` 命令触发时可保存上一个 Session 的对话摘要到 LanceDB
- 默认关闭（`enabled: false`），因为 OpenClaw 已有原生 .jsonl 会话保存
- 开启会导致大段摘要污染检索质量，建议仅在需要语义搜索历史会话时开启
- 可配置消息数量（默认 15 条）

### 8. 自动捕获 & 自动回忆

- **Auto-Capture**（`agent_end` hook）: 从对话中提取 preference/fact/decision/entity，去重后存储（每次最多 3 条）
- **Auto-Recall**（`before_agent_start` hook）: 注入 `<relevant-memories>` 上下文（最多 3 条）

---

## 安装

### AI 安装指引（防幻觉版）

如果你是用 AI 按 README 操作，**不要假设任何默认值**。请先运行以下命令，并以真实输出为准：

```bash
openclaw config get agents.defaults.workspace
openclaw config get plugins.load.paths
openclaw config get plugins.slots.memory
openclaw config get plugins.entries.memory-lancedb-pro
```

建议：
- `plugins.load.paths` 建议优先用**绝对路径**（除非你已确认当前 workspace）。
- 如果配置里使用 `${JINA_API_KEY}`（或任何 `${...}` 变量），务必确保运行 Gateway 的**服务进程环境**里真的有这些变量（systemd/launchd/docker 通常不会继承你终端的 export）。
- 修改插件配置后，运行 `openclaw gateway restart` 使其生效。

### Jina API Key（Embedding + Rerank）如何填写

- **Embedding**：将 `embedding.apiKey` 设置为你的 Jina key（推荐用环境变量 `${JINA_API_KEY}`）。
- **Rerank**（当 `retrieval.rerankProvider: "jina"`）：通常可以直接复用同一个 Jina key，填到 `retrieval.rerankApiKey`。
- 如果你选择了其它 rerank provider（如 `siliconflow` / `pinecone`），则 `retrieval.rerankApiKey` 应填写对应提供商的 key。

Key 存储建议：
- 不要把 key 提交到 git。
- 使用 `${...}` 环境变量没问题，但务必确保运行 Gateway 的**服务进程环境**里真的有该变量（systemd/launchd/docker 往往不会继承你终端的 export）。

### 什么是 “OpenClaw workspace”？

在 OpenClaw 中，**agent workspace（工作区）** 是 Agent 的工作目录（默认：`~/.openclaw/workspace`）。
根据官方文档，workspace 是 OpenClaw 的 **默认工作目录（cwd）**，因此 **相对路径会以 workspace 为基准解析**（除非你使用绝对路径）。

> 说明：OpenClaw 的配置文件通常在 `~/.openclaw/openclaw.json`，与 workspace 是分开的。

**最常见的安装错误：** 把插件 clone 到别的目录，但在配置里仍然写类似 `"paths": ["plugins/memory-lancedb-pro"]` 的**相对路径**。相对路径的解析基准会受 Gateway 启动方式/工作目录影响，容易指向错误位置。

为避免歧义：建议用**绝对路径**（方案 B），或把插件放在 `<workspace>/plugins/`（方案 A）并保持配置一致。

### 方案 A（推荐）：克隆到 workspace 的 `plugins/` 目录下

```bash
# 1) 进入你的 OpenClaw workspace（默认：~/.openclaw/workspace）
#    （可通过 agents.defaults.workspace 改成你自己的路径）
cd /path/to/your/openclaw/workspace

# 2) 把插件克隆到 workspace/plugins/ 下
git clone https://github.com/win4r/memory-lancedb-pro.git plugins/memory-lancedb-pro

# 3) 安装依赖
cd plugins/memory-lancedb-pro
npm install
```

然后在 OpenClaw 配置（`openclaw.json`）中使用相对路径：

```json
{
  "plugins": {
    "load": {
      "paths": ["plugins/memory-lancedb-pro"]
    },
    "entries": {
      "memory-lancedb-pro": {
        "enabled": true,
        "config": {
          "embedding": {
            "apiKey": "${JINA_API_KEY}",
            "model": "jina-embeddings-v5-text-small",
            "baseURL": "https://api.jina.ai/v1",
            "dimensions": 1024,
            "taskQuery": "retrieval.query",
            "taskPassage": "retrieval.passage",
            "normalized": true
          }
        }
      }
    },
    "slots": {
      "memory": "memory-lancedb-pro"
    }
  }
}
```

### 方案 B：插件装在任意目录，但配置里必须写绝对路径

```json
{
  "plugins": {
    "load": {
      "paths": ["/absolute/path/to/memory-lancedb-pro"]
    }
  }
}
```

### 重启

```bash
openclaw gateway restart
```

> **注意：** 如果之前使用了内置的 `memory-lancedb`，启用本插件时需同时禁用它。同一时间只能有一个 memory 插件处于活动状态。

### 验证是否安装成功（推荐）

1）确认插件已被发现/加载：

```bash
openclaw plugins list
openclaw plugins info memory-lancedb-pro
```

2）如果发现异常，运行插件诊断：

```bash
openclaw plugins doctor
```

3）确认 memory slot 已指向本插件：

```bash
# 期望看到：plugins.slots.memory = "memory-lancedb-pro"
openclaw config get plugins.slots.memory
```

---

## 配置

<details>
<summary><strong>完整配置示例（点击展开）</strong></summary>

```json
{
  "embedding": {
    "apiKey": "${JINA_API_KEY}",
    "model": "jina-embeddings-v5-text-small",
    "baseURL": "https://api.jina.ai/v1",
    "dimensions": 1024,
    "taskQuery": "retrieval.query",
    "taskPassage": "retrieval.passage",
    "normalized": true
  },
  "dbPath": "~/.openclaw/memory/lancedb-pro",
  "autoCapture": true,
  "autoRecall": true,
  "retrieval": {
    "mode": "hybrid",
    "vectorWeight": 0.7,
    "bm25Weight": 0.3,
    "minScore": 0.3,
    "rerank": "cross-encoder",
    "rerankApiKey": "${JINA_API_KEY}",
    "rerankModel": "jina-reranker-v3",
    "candidatePoolSize": 20,
    "recencyHalfLifeDays": 14,
    "recencyWeight": 0.1,
    "filterNoise": true,
    "lengthNormAnchor": 500,
    "hardMinScore": 0.35,
    "timeDecayHalfLifeDays": 60
  },
  "enableManagementTools": false,
  "scopes": {
    "default": "global",
    "definitions": {
      "global": { "description": "共享知识库" },
      "agent:discord-bot": { "description": "Discord 机器人私有" }
    },
    "agentAccess": {
      "discord-bot": ["global", "agent:discord-bot"]
    }
  },
  "sessionMemory": {
    "enabled": false,
    "messageCount": 15
  }
}
```

</details>

### Embedding 提供商

本插件支持 **任意 OpenAI 兼容的 Embedding API**：

| 提供商 | 模型 | Base URL | 维度 |
|--------|------|----------|------|
| **Jina**（推荐） | `jina-embeddings-v5-text-small` | `https://api.jina.ai/v1` | 1024 |
| **OpenAI** | `text-embedding-3-small` | `https://api.openai.com/v1` | 1536 |
| **Google Gemini** | `gemini-embedding-001` | `https://generativelanguage.googleapis.com/v1beta/openai/` | 3072 |
| **Ollama**（本地） | `nomic-embed-text` | `http://localhost:11434/v1` | _与本地模型输出一致_（建议显式设置 `embedding.dimensions`） |

---

## （可选）从 Session JSONL 自动蒸馏记忆（全自动）

OpenClaw 会把每个 Agent 的完整会话自动落盘为 JSONL：

- `~/.openclaw/agents/<agentId>/sessions/*.jsonl`

但 JSONL 含大量噪声（tool 输出、系统块、重复回调等），**不建议直接把原文塞进 LanceDB**。

**推荐方案（2026-02+）**：使用 **/new 非阻塞沉淀管线**（Hooks + systemd worker），在你执行 `/new` 时异步提取高价值经验并写入 LanceDB Pro：

- 触发：`command:new`（你在聊天里发送 `/new`）
- Hook：只投递一个很小的 task.json（毫秒级，不调用 LLM，不阻塞 `/new`）
- Worker：systemd 常驻进程监听队列，读取 session `.jsonl`，用 Gemini **Map-Reduce** 抽取 0～20 条高信噪比记忆
- 写入：通过 `openclaw memory-pro import` 写入 LanceDB Pro（插件内部仍会 embedding + 查重）
- 通知：可选（可做到即使 0 条也通知）

示例文件：
- `examples/new-session-distill/`

---

Legacy 方案：本插件也提供一个安全的 extractor 脚本 `scripts/jsonl_distill.py`，配合 OpenClaw 的 `cron` + 独立 distiller agent，实现“增量蒸馏 → 高质量记忆入库”：（适合不依赖 `/new` 的全自动场景）

- 只读取每个 JSONL 文件**新增尾巴**（byte offset cursor），避免重复和 token 浪费
- 生成一个小型 batch JSON
- 由 distiller agent 把 batch 蒸馏成短、原子、可复用的记忆，再用 `memory_store` 写入

### 你会得到什么

- ✅ 全自动（每小时）
- ✅ 多 Agent 支持（main + 各 bot）
- ✅ 只处理新增内容（不回读）
- ✅ 防自我吞噬：默认排除 `memory-distiller` 自己的 session

### 脚本输出位置

- Cursor：`~/.openclaw/state/jsonl-distill/cursor.json`
- Batches：`~/.openclaw/state/jsonl-distill/batches/`

> 脚本只读 session JSONL，不会修改原始日志。

### （可选）启用 Agent 来源白名单（提高信噪比）

默认情况下，extractor 会扫描 **所有 Agent**（但会排除 `memory-distiller` 自身，防止自我吞噬）。

如果你只想从某些 Agent 蒸馏（例如只蒸馏 `main` + `code-agent`），可以设置环境变量：

```bash
export OPENCLAW_JSONL_DISTILL_ALLOWED_AGENT_IDS="main,code-agent"
```

- 不设置 / 空 / `*` / `all`：扫描全部（默认）
- 逗号分隔列表：只扫描列表内 agentId

### 推荐部署（独立 distiller agent）

#### 1）创建 distiller agent（示例用 gpt-5.2）

```bash
openclaw agents add memory-distiller \
  --non-interactive \
  --workspace ~/.openclaw/workspace-memory-distiller \
  --model openai-codex/gpt-5.2
```

#### 2）初始化 cursor（模式 A：从现在开始，不回溯历史）

先确定插件目录（PLUGIN_DIR）：

```bash
# 如果你按推荐方式 clone 到 workspace：
#   PLUGIN_DIR="$HOME/.openclaw/workspace/plugins/memory-lancedb-pro"
PLUGIN_DIR="/path/to/memory-lancedb-pro"

python3 "$PLUGIN_DIR/scripts/jsonl_distill.py" init
```

#### 3）创建每小时 Cron（Asia/Shanghai）

建议 cron message 以 `run ...` 开头，这样本插件的自适应检索会跳过自动 recall 注入（节省 token）。

```bash
MSG=$(cat <<'EOF'
run jsonl memory distill

Goal: Distill ONLY new content from OpenClaw session JSONL tails into high-quality LanceDB memories.

Hard rules:
- Incremental only: exec the extractor. Do NOT scan full history.
- If extractor returns action=noop: stop immediately.
- Store only reusable memories (rules, pitfalls, decisions, preferences, stable facts). Skip routine chatter.
- Each memory: idiomatic English + final line `Keywords (zh): ...` (3-8 short phrases).
- Keep each memory < 500 chars and atomic.
- Caps: <= 3 memories per agent per run; <= 3 global per run.
- Scope:
  - broadly reusable -> global
  - agent-specific -> agent:<agentId>

Workflow:
1) exec: python3 <PLUGIN_DIR>/scripts/jsonl_distill.py run
2) Determine batch file (created/pending)
3) memory_store(...) for selected memories
4) exec: python3 <PLUGIN_DIR>/scripts/jsonl_distill.py commit --batch-file <batchFile>
EOF
)

openclaw cron add \
  --agent memory-distiller \
  --name "jsonl-memory-distill (hourly)" \
  --cron "0 * * * *" \
  --tz "Asia/Shanghai" \
  --session isolated \
  --wake now \
  --timeout-seconds 420 \
  --stagger 5m \
  --no-deliver \
  --message "$MSG"
```

### scope 策略（非常重要）

当蒸馏“所有 agents”时，务必显式设置 scope：

- 跨 agent 通用规则/偏好/坑 → `scope=global`
- agent 私有 → `scope=agent:<agentId>`

否则不同 bot 的记忆会相互污染。

### 回滚

- 禁用/删除 cron：`openclaw cron disable <jobId>` / `openclaw cron rm <jobId>`
- 删除 distiller agent：`openclaw agents delete memory-distiller`
- 删除 cursor 状态：`rm -rf ~/.openclaw/state/jsonl-distill/`

---

## CLI 命令

```bash
# 列出记忆
openclaw memory-pro list [--scope global] [--category fact] [--limit 20] [--json]

# 搜索记忆
openclaw memory-pro search "query" [--scope global] [--limit 10] [--json]

# 查看统计
openclaw memory-pro stats [--scope global] [--json]

# 按 ID 删除记忆（支持 8+ 字符前缀）
openclaw memory-pro delete <id>

# 批量删除
openclaw memory-pro delete-bulk --scope global [--before 2025-01-01] [--dry-run]

# 导出 / 导入
openclaw memory-pro export [--scope global] [--output memories.json]
openclaw memory-pro import memories.json [--scope global] [--dry-run]

# 使用新模型重新生成 Embedding
openclaw memory-pro reembed --source-db /path/to/old-db [--batch-size 32] [--skip-existing]

# 从内置 memory-lancedb 迁移
openclaw memory-pro migrate check [--source /path]
openclaw memory-pro migrate run [--source /path] [--dry-run] [--skip-existing]
openclaw memory-pro migrate verify [--source /path]
```

---

## 数据库 Schema

LanceDB 表 `memories`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string (UUID) | 主键 |
| `text` | string | 记忆文本（FTS 索引） |
| `vector` | float[] | Embedding 向量 |
| `category` | string | `preference` / `fact` / `decision` / `entity` / `other` |
| `scope` | string | Scope 标识（如 `global`、`agent:main`） |
| `importance` | float | 重要性分数 0-1 |
| `timestamp` | int64 | 创建时间戳 (ms) |
| `metadata` | string (JSON) | 扩展元数据 |

---

## 依赖

| 包 | 用途 |
|----|------|
| `@lancedb/lancedb` ≥0.26.2 | 向量数据库（ANN + FTS） |
| `openai` ≥6.21.0 | OpenAI 兼容 Embedding API 客户端 |
| `@sinclair/typebox` 0.34.48 | JSON Schema 类型定义（工具参数） |

---

## License

MIT

---

## Buy Me a Coffee

[!["Buy Me A Coffee"](https://storage.ko-fi.com/cdn/kofi2.png?v=3)](https://ko-fi.com/aila)

## 我的微信群和微信二维码

<img src="https://github.com/win4r/AISuperDomain/assets/42172631/d6dcfd1a-60fa-4b6f-9d5e-1482150a7d95" width="186" height="300">
<img src="https://github.com/win4r/AISuperDomain/assets/42172631/7568cf78-c8ba-4182-aa96-d524d903f2bc" width="214.8" height="291">
<img src="https://github.com/win4r/AISuperDomain/assets/42172631/fefe535c-8153-4046-bfb4-e65eacbf7a33" width="207" height="281">
