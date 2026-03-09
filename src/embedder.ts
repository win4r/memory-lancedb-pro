/**
 * Embedder with batch support
 */

export interface EmbedderConfig {
  provider: "cloudflare" | "openai" | "local";
  apiKey?: string;
  model?: string;
  batchSize?: number;
}

export interface EmbeddingResult {
  vector: number[];
  tokens?: number;
}

export class Embedder {
  private config: Required<EmbedderConfig>;
  private tokenCount: number = 0;

  constructor(config: EmbedderConfig) {
    this.config = {
      provider: config.provider,
      apiKey: config.apiKey || "",
      model: config.model || "@cf/baai/bge-large-en-v1.5",
      batchSize: config.batchSize ?? 20,
    };
  }

  // =========================================================================
  // Single Embedding
  // =========================================================================

  async embedPassage(text: string): Promise<number[]> {
    const result = await this.embedOne(text, "passage");
    return result.vector;
  }

  async embedQuery(text: string): Promise<number[]> {
    const result = await this.embedOne(text, "query");
    return result.vector;
  }

  private async embedOne(text: string, type: "passage" | "query"): Promise<EmbeddingResult> {
    if (this.config.provider === "cloudflare") {
      return this.embedWithCloudflare(text, type);
    }
    
    // Fallback or other providers
    throw new Error(`Unsupported provider: ${this.config.provider}`);
  }

  // =========================================================================
  // Batch Embedding
  // =========================================================================

  async embedBatch(texts: string[], type: "passage" | "query" = "passage"): Promise<number[][]> {
    const results: number[][] = [];
    
    // Process in batches to avoid rate limits
    for (let i = 0; i < texts.length; i += this.config.batchSize) {
      const batch = texts.slice(i, i + this.config.batchSize);
      const batchResults = await this.embedBatchWithCloudflare(batch, type);
      results.push(...batchResults);
    }
    
    return results;
  }

  async embedPassages(texts: string[]): Promise<number[][]> {
    return this.embedBatch(texts, "passage");
  }

  async embedQueries(texts: string[]): Promise<number[][]> {
    return this.embedBatch(texts, "query");
  }

  // =========================================================================
  // Cloudflare Workers AI
  // =========================================================================

  private async embedWithCloudflare(text: string, type: "passage" | "query"): Promise<EmbeddingResult> {
    const response = await fetch(
      `https://api.cloudflare.com/client/v4/accounts/${process.env.CLOUDFLARE_ACCOUNT_ID}/ai/run/${this.config.model}`,
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${this.config.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: [text],
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`Cloudflare API error: ${response.status}`);
    }

    const data = await response.json();
    const vector = data.result?.data?.[0] || [];
    
    this.tokenCount += Math.ceil(text.length / 4);
    
    return { vector, tokens: Math.ceil(text.length / 4) };
  }

  private async embedBatchWithCloudflare(texts: string[], type: "passage" | "query"): Promise<number[][]> {
    // Cloudflare AI supports batch embedding
    const response = await fetch(
      `https://api.cloudflare.com/client/v4/accounts/${process.env.CLOUDFLARE_ACCOUNT_ID}/ai/run/${this.config.model}`,
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${this.config.apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: texts,
        }),
      }
    );

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Cloudflare API error: ${response.status} - ${error}`);
    }

    const data = await response.json();
    const vectors = data.result?.data || [];
    
    this.tokenCount += texts.reduce((sum, t) => sum + Math.ceil(t.length / 4), 0);
    
    return vectors;
  }

  // =========================================================================
  // Stats
  // =========================================================================

  getTokenCount(): number {
    return this.tokenCount;
  }

  resetTokenCount(): void {
    this.tokenCount = 0;
  }

  getDimension(): number {
    // BGE-Large dimension
    return 1024;
  }
}
