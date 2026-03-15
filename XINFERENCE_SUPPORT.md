# Xinference Rerank 支持

本文档说明如何在 memory-lancedb-pro 中使用 Xinference 作为 rerank 提供者。

## 新增功能

已为 memory-lancedb-pro 添加了 Xinference 作为新的 rerank provider 支持。Xinference 是一个开源的模型推理框架，提供类似 OpenAI 的 API 接口。

## 配置示例

在 OpenClaw 配置中添加以下设置以使用 Xinference rerank：

```json
{
  "plugins": {
    "entries": {
      "memory-lancedb-pro": {
        "enabled": true,
        "config": {
          "retrieval": {
            "rerank": "cross-encoder",
            "rerankProvider": "xinference",
            "rerankEndpoint": "http://localhost:9997/v1/rerank",
            "rerankModel": "Qwen3-Reranker-4B",
            "rerankApiKey": "your-api-key-here" // 如果 Xinference 需要 API key
          }
        }
      }
    }
  }
}
```

## 支持的配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rerankProvider` | `string` | `"jina"` | 设置为 `"xinference"` 以使用 Xinference |
| `rerankEndpoint` | `string` | `"http://localhost:9997/v1/rerank"` | Xinference API 端点 |
| `rerankModel` | `string` | `"Qwen3-Reranker-4B"` | 使用的 rerank 模型 |
| `rerankApiKey` | `string` | (可选) | API 密钥（如果 Xinference 需要） |

## Xinference 设置

### 1. 安装和启动 Xinference

```bash
# 安装 Xinference
pip install xinference

# 启动 Xinference 服务器
xinference launch --port 9997

# 下载并启动 rerank 模型
xinference launch --model-name Qwen3-Reranker-4B --model-type rerank
```

### 2. 验证 Xinference 运行

```bash
# 检查服务器状态
curl http://localhost:9997/v1/models

# 测试 rerank 接口
curl -X POST http://localhost:9997/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Reranker-4B",
    "query": "测试查询",
    "documents": ["文档1", "文档2"],
    "top_n": 2
  }'
```

## API 兼容性

Xinference 提供与 OpenAI 兼容的 API 接口：

### 请求格式
```json
{
  "model": "模型名称",
  "query": "查询文本",
  "documents": ["文档1", "文档2", ...],
  "top_n": 10
}
```

### 响应格式

Xinference 官方 API 返回格式：
```json
{
  "id": "请求ID",
  "object": "rerank",
  "created": 时间戳,
  "model": "模型名称",
  "results": [
    {"index": 0, "relevance_score": 0.95, "document": "文档内容"},
    {"index": 1, "relevance_score": 0.87, "document": "文档内容"},
    ...
  ]
}
```

**注意**：某些版本的 Xinference 可能使用兼容格式：
```json
{
  "data": [
    {"index": 0, "score": 0.95},
    {"index": 1, "score": 0.87}
  ]
}
```

项目代码已兼容两种格式，优先使用官方格式 (`results` + `relevance_score`)。

## 故障排除

### 1. 连接失败
- 检查 Xinference 服务器是否运行：`curl http://localhost:9997/v1/models`
- 验证端口是否正确（默认：9997）
- 检查防火墙设置

### 2. API 错误
- 确认模型名称正确
- 检查 API 密钥（如果需要）
- 查看 Xinference 日志：`xinference logs`

### 3. 性能问题
- 调整 `candidatePoolSize` 减少候选文档数量
- 增加超时时间（当前为 5 秒）
- 考虑使用本地缓存

## 与其他 Provider 的比较

| Provider | 请求格式 | 响应格式 | 认证方式 |
|----------|----------|----------|----------|
| **Xinference** | OpenAI 兼容 | `results[index, relevance_score, document]` (官方)<br>`data[index, score]` (兼容) | Bearer Token |
| Jina | `documents[]` | `results[index, relevance_score]` | Bearer Token |
| Pinecone | `documents[{text}]` | `data[index, score]` | Api-Key |
| Voyage | `documents[]` | `data[index, relevance_score]` | Bearer Token |

## 代码修改

本次添加 Xinference 支持涉及以下文件修改：

1. **src/retriever.ts**
   - 扩展 `RerankProvider` 类型包含 `"xinference"`
   - 在 `buildRerankRequest()` 中添加 Xinference case
   - 在 `parseRerankResponse()` 中添加 Xinference case
   - 更新配置注释

2. **index.ts**
   - 更新 `rerankProvider` 类型定义

3. **openclaw.plugin.json**
   - 在 enum 中添加 `"xinference"`
   - 更新帮助文本

## 测试

已通过基础验证测试，确认：
- 请求构建正确
- 响应解析正确
- 与其他 provider 兼容

## 注意事项

1. Xinference 通常运行在本地，适合隐私敏感场景
2. API 密钥可能不是必需的（取决于 Xinference 配置）
3. 响应格式遵循 OpenAI 兼容标准
4. 默认超时时间为 5 秒

## 后续优化建议

1. 添加 Xinference 特定的错误处理
2. 支持更多 Xinference 模型
3. 添加连接测试和健康检查
4. 优化本地缓存的集成