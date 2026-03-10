# memory-lancedb-pro v1.1.0 — 智能记忆增强

> **日期**: 2026-03-06  
> **作者**: CJY  
> **概述**: 基于对 AI Agent 记忆系统的深入理解，对记忆的写入质量、生命周期管理和去重能力进行了全面改进与完善

---

## 一、改进动机

原有记忆系统在**检索侧**表现优异（Vector+BM25 混合检索、cross-encoder 重排序、多维评分），但在以下方面存在提升空间：

- **记忆写入质量**：依赖正则表达式触发捕获，容易漏捕有价值信息或误捕噪声
- **记忆结构层次**：扁平文本存储，缺乏分层索引能力
- **记忆生命周期**：简单时间衰减，无法模拟人类记忆的遗忘与强化规律
- **去重能力**：仅基于向量相似度的粗粒度去重，缺乏语义级判断

本次改进针对这三个维度进行了系统性增强。

---

## 二、变更摘要

| 改进维度     | 核心变更                                  | 效果                               |
| ------------ | ----------------------------------------- | ---------------------------------- |
| 智能提取     | LLM 驱动的 6 类别提取 + L0/L1/L2 分层存储 | 记忆写入更精准、结构更丰富         |
| 生命周期管理 | Weibull 衰减模型 + 三层晋升/降级          | 重要记忆持久保留，过时记忆自然淡化 |
| 智能去重     | 向量预过滤 + LLM 语义决策                 | 避免冗余记忆，支持信息演化合并     |

### 当前落地状态（2026-03-06）

以下能力已接入运行主路径，而不再只是"模块存在"：

- `SmartExtractor` 已挂到 `agent_end`，新写入默认产出 `memory_category`、`tier`、`l0_abstract`、`l1_overview`、`l2_content`
- `before_agent_start` 自动召回后会回写 `access_count` 与 `last_accessed_at`
- `src/retriever.ts` 已接入 `decayEngine.applySearchBoost()`，检索结果按生命周期分数重排
- `tierManager` 已接入 recall 后维护流程，根据 recall 计数和 lifecycle score 回写 tier 变更
- `memory_store`、regex fallback、迁移、session memory、upgrade 都统一写入 smart metadata
- `openclaw.plugin.json` 已暴露 `decay` / `tier` 配置项
- 生命周期回归测试已加入 `npm test`

仍保留的兼容路径：

- 当 lifecycle decay 未启用或不可用时，retriever 仍可回退到旧的 `Recency Boost → Importance Weight → Time Decay` 排序链
- 旧格式数据仍可读取，并可通过 `openclaw memory-pro upgrade` 统一补齐 metadata

---

## 三、新增文件

### 1. `src/memory-categories.ts` — 6 类别分类系统

设计了语义明确的记忆分类体系，将记忆分为两大类六小类：

- **用户记忆**：`profile`（身份属性）、`preferences`（偏好习惯）、`entities`（持续存在的实体）、`events`（发生的事件）
- **Agent 记忆**：`cases`（问题-解决方案对）、`patterns`（可复用的处理流程）

每个类别有不同的合并策略：

- `profile` → 始终合并（用户身份信息持续累积）
- `preferences` / `entities` / `patterns` → 支持智能合并
- `events` / `cases` → 仅新增或跳过（独立记录，保留历史完整性）

---

### 2. `src/llm-client.ts` — LLM 客户端

封装了 LLM 调用接口，专注于结构化 JSON 输出：

- 复用现有 OpenAI SDK 依赖，零新增包
- 内置 JSON 容错解析：支持 markdown 代码块包裹和平衡大括号提取
- 低温度 (0.1) 保证输出一致性
- 30 秒超时保护，失败时优雅降级

---

### 3. `src/extraction-prompts.ts` — 记忆提取提示模板

精心设计了 3 个提示模板：

| 函数                      | 用途                                                |
| ------------------------- | --------------------------------------------------- |
| `buildExtractionPrompt()` | 从对话中提取 6 类别 L0/L1/L2 记忆，含 few-shot 示例 |
| `buildDedupPrompt()`      | CREATE / MERGE / SKIP 去重决策                      |
| `buildMergePrompt()`      | 将新旧记忆合并为三层结构                            |

提取提示包含完整的记忆价值判断标准、类别决策逻辑表、常见混淆澄清规则和 6 个 few-shot 示例。

---

### 4. `src/smart-extractor.ts` — 智能提取管线

实现了完整的 LLM 驱动提取流水线：

```
对话文本 → LLM 提取 → 候选记忆 → 向量去重 → LLM 决策 → 持久化
```

核心设计：

- **两阶段去重**：先用向量相似度（阈值 0.7）快速筛选候选，再用 LLM 进行语义级判断
- **类别感知合并**：不同类别应用不同合并策略
- **L0/L1/L2 三层存储**：L0 一句话索引用于检索注入，L1 结构化摘要用于精读，L2 完整叙述用于深度回顾
- **向后兼容**：新增的 6 类别自动映射到已有的 5 类别存储，L0/L1/L2 存储在 metadata JSON 中
- **按类别设定重要度**：profile (0.9) > patterns (0.85) > cases/preferences (0.8) > entities (0.7) > events (0.6)

---

### 5. `src/decay-engine.ts` — Weibull 衰减引擎

基于认知心理学中的记忆遗忘曲线研究，实现了复合衰减模型：

**复合分数 = 时效权重 × 时效 + 频率权重 × 频率 + 内在权重 × 内在价值**

三个分量：

| 分量                     | 机制                              | 含义                   |
| ------------------------ | --------------------------------- | ---------------------- |
| **时效 (recency)**       | Weibull 拉伸指数衰减 `exp(-λt^β)` | 越久远的记忆衰减越快   |
| **频率 (frequency)**     | 对数饱和曲线 + 时间加权           | 越常被访问的记忆越活跃 |
| **内在价值 (intrinsic)** | `importance × confidence`         | 高价值记忆天然抵抗遗忘 |

层级特定的衰减形状 (β 参数)：

- **Core** (β=0.8)：亚指数衰减 → 遗忘极慢，衰减地板 0.9
- **Working** (β=1.0)：标准指数衰减，衰减地板 0.7
- **Peripheral** (β=1.3)：超指数衰减 → 遗忘加速，衰减地板 0.5

关键特性：

- **重要性调制半衰期**：`effectiveHL = halfLife × exp(μ × importance)`，重要记忆持续更久
- **搜索结果加权**：检索时自动应用衰减加权，让活跃记忆排名更高
- **过期识别**：识别 composite < 0.3 的过期记忆

---

### 6. `src/tier-manager.ts` — 三层晋升/降级管理器

模拟人类记忆的多级存储模型：

```
Peripheral（外围） ⟷ Working（工作） ⟷ Core（核心）
```

**晋升条件**：

| 方向                 | 条件                                            |
| -------------------- | ----------------------------------------------- |
| Peripheral → Working | 访问次数 ≥ 3 且 衰减分数 ≥ 0.4                  |
| Working → Core       | 访问次数 ≥ 10 且 衰减分数 ≥ 0.7 且 重要度 ≥ 0.8 |

**降级条件**：

| 方向                 | 条件                                             |
| -------------------- | ------------------------------------------------ |
| Working → Peripheral | 衰减分数 < 0.15 或（年龄 > 60 天且访问次数 < 3） |
| Core → Working       | 衰减分数 < 0.15 且 访问次数 < 3（极少触发）      |

---

### 7. `src/smart-metadata.ts` — Smart metadata 归一化辅助模块

统一新老写入入口的生命周期字段，确保所有记忆路径产出一致的 metadata 结构：

- 统一补齐 `memory_category`、`tier`、`l0_abstract`、`l1_overview`、`l2_content`
- 统一处理 `access_count`、`confidence`、`last_accessed_at`
- 将存储记录转换为 lifecycle scoring 所需的 `DecayableMemory`
- 为迁移、upgrade、tool-store、regex fallback 提供同一套默认值与兼容逻辑

---

### 8. `src/memory-upgrader.ts` — 旧记忆升级器

将旧格式记忆（无 L0/L1/L2 / 5 类别）批量升级为新智能记忆格式，统一生命周期管理：

- **反向类别映射**：`fact→profile/cases`（含启发式+LLM 判断）、`preference→preferences`、`entity→entities`、`decision→events`、`other→patterns`
- **L0/L1/L2 生成**：LLM 模式（推荐）或 无 LLM 模式（首句截取+原文保留）
- **元数据补全**：自动填充 `tier: "working"`、`access_count: 0`、`confidence: 0.7`
- **批量处理**：支持 `--batch-size`、`--limit`、`--dry-run` 控制
- **启动检测**：插件启动 5 秒后自动检测旧记忆数量，日志提示升级

---

## 四、修改文件

### `index.ts` — 插件入口

#### 新增导入

```typescript
import { SmartExtractor } from "./src/smart-extractor.js";
import { createLlmClient } from "./src/llm-client.js";
import { createDecayEngine, DEFAULT_DECAY_CONFIG } from "./src/decay-engine.js";
import { createTierManager, DEFAULT_TIER_CONFIG } from "./src/tier-manager.js";
```

#### 新增配置项

```typescript
smartExtraction?: boolean;    // 是否启用 LLM 智能提取（默认 true）
llm?: {
  apiKey?: string;            // LLM API Key（默认复用 embedding.apiKey）
  model?: string;             // LLM 模型（默认 gpt-4o-mini）
  baseURL?: string;           // LLM API 端点
};
extractMinMessages?: number;  // 最少消息数才触发提取（默认 2）
extractMaxChars?: number;     // 送入 LLM 的最大字符数（默认 8000）
```

#### `agent_end` 钩子改进

- 当 `smartExtraction` 启用时，优先使用 SmartExtractor 进行 LLM 6 类别提取
- 当消息数不足或 SmartExtractor 未初始化时，降级回原有正则触发逻辑
- 提取完成后输出统计日志：`smart-extracted N created, M merged, K skipped`

#### `before_agent_start` 钩子改进

- 注入的记忆上下文现在显示 L0 摘要而非原始文本
- 新增 6 类别标签（如 `[preferences:global]`）
- 新增层级标记（`[C]`ore / `[W]`orking / `[P]`eripheral）
- 自动召回后回写 `access_count` 与 `last_accessed_at`

#### 启动时旧记忆检测

- 插件启动 5 秒后异步扫描旧格式记忆数量
- 如发现旧格式记忆，在日志中输出提示：`Run 'openclaw memory-pro upgrade' to convert them.`

### `src/retriever.ts` — 检索器

- 接入 `decayEngine.applySearchBoost()`，检索结果按生命周期分数重排
- 当 lifecycle decay 未启用或不可用时，可回退到旧的 `Recency Boost → Importance Weight → Time Decay` 排序链

### `openclaw.plugin.json` — 插件配置清单

- 暴露 `decay` / `tier` 配置项，允许用户自定义衰减参数和晋升阈值

### `cli.ts` — CLI 命令

#### 新增 `memory-pro upgrade` 命令

```bash
openclaw memory-pro upgrade [--dry-run] [--batch-size N] [--no-llm] [--limit N] [--scope SCOPE]
```

- `--dry-run`：仅统计旧记忆数量，不修改数据
- `--batch-size`：每批处理数量（默认 10）
- `--no-llm`：不调用 LLM，使用简单规则生成 L0/L1/L2
- `--limit`：最大升级数量
- `--scope`：仅升级指定 scope 的记忆

---

## 五、配置指南

### 最简配置（复用已有 API Key）

```json
{
  "embedding": {
    "apiKey": "${OPENAI_API_KEY}",
    "model": "text-embedding-3-small"
  },
  "smartExtraction": true
}
```

### 完整配置

```json
{
  "embedding": {
    "apiKey": "${OPENAI_API_KEY}",
    "model": "text-embedding-3-small"
  },
  "smartExtraction": true,
  "llm": {
    "apiKey": "${OPENAI_API_KEY}",
    "model": "gpt-4o-mini",
    "baseURL": "https://api.openai.com/v1"
  },
  "extractMinMessages": 2,
  "extractMaxChars": 8000
}
```

### 禁用智能提取

```json
{
  "smartExtraction": false
}
```

---

## 六、向后兼容性

| 方面           | 兼容方式                                       |
| -------------- | ---------------------------------------------- |
| LanceDB Schema | 新字段存储在 `metadata` JSON 中，不修改表结构  |
| 记忆类别       | 新 6 类别自动映射到原有 5 类别                 |
| 混合检索       | Vector+BM25 检索管线完全保留                   |
| 去重逻辑       | 仅在 `smartExtraction: true` 时生效            |
| 已有数据       | 旧记忆正常读取，新记忆额外携带 L0/L1/L2 元数据 |
| 配置           | 全部新增配置项均有默认值，零配置即可使用       |
| **旧记忆升级** | `memory-pro upgrade` 命令一键升级为新格式      |
