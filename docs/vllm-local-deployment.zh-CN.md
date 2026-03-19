# 使用 vLLM 本地部署 memory-lancedb-pro

本文档提供一个单一、可落地的本地部署流程，适用于使用 vLLM（或 OpenAI 兼容服务）提供 Embedding 与 Rerank 能力的场景。

## 适用场景

- 希望在本地或内网运行 Embedding 与 Rerank 服务。
- 需要避免外网请求或遵循数据合规要求。
- 使用 vLLM / OpenAI 兼容服务提供 Embedding API 与 Rerank API。

## 前置准备

- 已安装并可运行 OpenClaw 与 memory-lancedb-pro 插件。
- 本机或局域网已启动 Embedding 与 Rerank 服务。
- 明确 Embedding 模型输出维度（需与 `embedding.dimensions` 保持一致）。
- 确保 OpenClaw 运行环境能访问本地服务端口（例如 `localhost:8001`、`localhost:8002`）。

## 服务启动约定

请确保你的服务满足以下约定（端口和路径可按需调整，但要与配置一致）：

- Embedding 服务：`http://localhost:8001/v1`
  - 提供 OpenAI 兼容 `/v1/embeddings` 接口。
- Rerank 服务：`http://localhost:8002/v1/rerank`
  - 与 `rerankProvider: "jina"` 的请求格式兼容。

如果你的服务路径或端口不同，请在配置里同步修改 `embedding.baseURL` 与 `retrieval.rerankEndpoint`。

## OpenClaw 配置示例

以下示例展示在 OpenClaw 配置中启用 memory-lancedb-pro，并使用本地 vLLM 服务。重点在于：

- `embedding.requestDimensions: false` 以避免兼容性问题
- `retrieval.timeoutMs` 用于慢速本地 rerank 服务

```json
{
  "plugins": {
    "slots": {
      "memory": "memory-lancedb-pro"
    },
    "entries": {
      "memory-lancedb-pro": {
        "enabled": true,
        "config": {
          "embedding": {
            "apiKey": "none",
            "model": "Qwen3-Embedding-0.6B",
            "baseURL": "http://localhost:8001/v1",
            "dimensions": 1024,
            "requestDimensions": false,
            "normalized": true
          },
          "retrieval": {
            "mode": "hybrid",
            "vectorWeight": 0.7,
            "bm25Weight": 0.3,
            "rerank": "cross-encoder",
            "rerankProvider": "jina",
            "rerankModel": "Qwen3-Reranker-0.6B",
            "rerankEndpoint": "http://localhost:8002/v1/rerank",
            "rerankApiKey": "none",
            "timeoutMs": 15000,
            "candidatePoolSize": 12,
            "minScore": 0.4,
            "hardMinScore": 0.45,
            "filterNoise": true
          }
        }
      }
    }
  }
}
```

## 功能验证

使用当前支持的 OpenClaw 命令与流程进行验证：

1. 基础状态确认：

```bash
openclaw config validate
openclaw status
openclaw gateway status
openclaw plugins info memory-lancedb-pro
```

2. 写入与检索验证（推荐包含一个唯一关键词）：

- 在真实对话中触发 `memory_store` 或让 auto-capture 写入一条包含唯一关键词的记忆。
- 然后使用 CLI 或对话检索确认：

```bash
openclaw memory-pro stats
openclaw memory-pro search "你的唯一关键词" --scope global --limit 5
```

3. 如果检索超时或无结果，先检查本地服务日志，再对照下方常见问题排查。

## 常见问题

- **Embedding 端点拒绝 `dimensions` 字段**：将 `embedding.requestDimensions` 设为 `false`，避免发送维度参数。
- **Rerank 服务响应较慢**：提高 `retrieval.timeoutMs`，例如 15000 或更高。
- **向量维度不匹配**：确保 `embedding.dimensions` 与 Embedding 模型实际输出维度一致，并与现有数据库维度一致。
- **端点路径不一致**：确认 `embedding.baseURL` 是否包含 `/v1`，以及 `retrieval.rerankEndpoint` 是否为 `/v1/rerank`。
