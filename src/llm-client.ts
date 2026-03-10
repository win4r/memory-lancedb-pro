/**
 * LLM Client for memory extraction and dedup decisions.
 * Uses OpenAI-compatible API (reuses the embedding provider config).
 */

import OpenAI from "openai";

export interface LlmClientConfig {
  apiKey: string;
  model: string;
  baseURL?: string;
  timeoutMs?: number;
}

export interface LlmClient {
  /** Send a prompt and parse the JSON response. Returns null on failure. */
  completeJson<T>(prompt: string): Promise<T | null>;
}

/**
 * Extract JSON from an LLM response that may be wrapped in markdown fences
 * or contain surrounding text.
 */
function extractJsonFromResponse(text: string): string | null {
  // Try markdown code fence first (```json ... ``` or ``` ... ```)
  const fenceMatch = text.match(/```(?:json)?\s*\n?([\s\S]*?)```/);
  if (fenceMatch) {
    return fenceMatch[1].trim();
  }

  // Try balanced brace extraction
  const firstBrace = text.indexOf("{");
  if (firstBrace === -1) return null;

  let depth = 0;
  let lastBrace = -1;
  for (let i = firstBrace; i < text.length; i++) {
    if (text[i] === "{") depth++;
    else if (text[i] === "}") {
      depth--;
      if (depth === 0) {
        lastBrace = i;
        break;
      }
    }
  }

  if (lastBrace === -1) return null;
  return text.substring(firstBrace, lastBrace + 1);
}

export function createLlmClient(config: LlmClientConfig): LlmClient {
  const client = new OpenAI({
    apiKey: config.apiKey,
    baseURL: config.baseURL,
    timeout: config.timeoutMs ?? 30000,
  });

  return {
    async completeJson<T>(prompt: string): Promise<T | null> {
      try {
        const response = await client.chat.completions.create({
          model: config.model,
          messages: [
            {
              role: "system",
              content:
                "You are a memory extraction assistant. Always respond with valid JSON only.",
            },
            { role: "user", content: prompt },
          ],
          temperature: 0.1,
        });

        const raw = response.choices?.[0]?.message?.content;
        if (!raw) return null;

        const jsonStr = extractJsonFromResponse(raw);
        if (!jsonStr) return null;

        return JSON.parse(jsonStr) as T;
      } catch (err) {
        // Graceful degradation — return null so caller can fall back
        return null;
      }
    },
  };
}

export { extractJsonFromResponse };
