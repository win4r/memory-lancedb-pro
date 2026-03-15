# Xinference Rerank 支持更新总结

## 修改概述

根据 xinference 官方文档 (https://inference.readthedocs.io/zh-cn/latest/models/model_abilities/rerank.html#rerank) 的要求，更新了 memory-lancedb-pro 项目中 xinference rerank 支持的响应解析逻辑。

## 问题发现

通过对比 xinference 官方文档和项目现有实现，发现以下差异：

### 官方文档格式
```json
{
  "id": "480dca92-8910-11ee-b76a-c2c8e4cad3f5",
  "results": [{
    "index": 0,
    "relevance_score": 0.9999247789382935,
    "document": "A man is eating food."
  }]
}
```

### 项目原有实现假设
```json
{
  "data": [{
    "index": 0,
    "score": 0.95
  }]
}
```

## 具体修改

### 文件：`src/retriever.ts`

#### 1. 更新 `parseRerankResponse` 函数中的 `xinference` case

**修改前：**
```typescript
case "siliconflow":
case "jina":
case "xinference": {
  // Jina / SiliconFlow / Xinference: usually { data: [{ index, score }] }
  // Xinference follows OpenAI-compatible format: { data: [{ index, score }] }
  // Also tolerate results[] for compatibility across gateways.
  return (
    parseItems(data.data, ["score", "relevance_score"]) ??
    parseItems(data.results, ["score", "relevance_score"])
  );
}
```

**修改后：**
```typescript
case "xinference": {
  // Xinference: official API returns { results: [{ index, relevance_score, document }] }
  // Older/local versions may return { data: [{ index, score }] }
  // Prioritize official format (results + relevance_score), fallback to data + score
  return (
    parseItems(data.results, ["relevance_score", "score"]) ??
    parseItems(data.data, ["relevance_score", "score"])
  );
}
case "siliconflow":
case "jina": {
  // Jina / SiliconFlow: { data: [{ index, score }] }
  return (
    parseItems(data.data, ["score", "relevance_score"]) ??
    parseItems(data.results, ["score", "relevance_score"])
  );
}
```

### 修改要点

1. **分离 xinference 处理逻辑**：将 xinference 从 jina/siliconflow 的 case 中分离出来
2. **优先使用官方格式**：优先解析 `data.results` 数组和 `relevance_score` 字段
3. **保持向后兼容**：如果官方格式不存在，回退到 `data.data` 数组和 `score` 字段
4. **字段优先级调整**：对于 xinference，优先查找 `relevance_score`，然后才是 `score`

## 测试验证

### 现有测试
项目中的测试文件 `test/retriever-rerank-regression.mjs` 已经使用了正确的 xinference 官方格式：
```javascript
await runScenario("low-score rerank result", {
  results: [{ index: 0, relevance_score: 0 }],  // 正确格式
});
```

### 测试结果
所有测试通过，包括：
- rerank 回归测试
- 智能提取器分支测试
- 向量搜索测试
- 上下文支持测试

## 兼容性说明

### 向后兼容
修改后的代码保持向后兼容：
1. 优先使用官方格式 (`results` + `relevance_score`)
2. 如果官方格式不存在，回退到旧格式 (`data` + `score`)
3. 支持不同版本的 xinference 部署

### 与其他 Provider 的兼容性
- **Jina/SiliconFlow**：保持不变，优先使用 `data` 数组
- **Pinecone**：保持不变
- **Voyage**：保持不变
- **Xinference**：更新为优先使用官方格式

## 配置示例

更新后的配置使用方式不变：

```json
{
  "retrieval": {
    "rerank": "cross-encoder",
    "rerankProvider": "xinference",
    "rerankEndpoint": "http://localhost:9997/v1/rerank",
    "rerankModel": "Qwen3-Reranker-4B",
    "rerankApiKey": "your-api-key-here"
  }
}
```

## 文档更新建议

建议更新 `XINFERENCE_SUPPORT.md` 文件中的响应格式说明，以反映官方文档的正确格式。

## 总结

本次修改确保了 memory-lancedb-pro 项目中的 xinference rerank 支持符合官方 API 规范，同时保持了向后兼容性。所有现有测试通过，证明修改不影响现有功能。

**修改时间**：2026-03-15 11:30 (北京时间)