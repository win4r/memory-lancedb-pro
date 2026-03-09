import type { GraphitiPluginConfig } from "./types.js";

interface LoggerLike {
  info?: (message: string) => void;
  warn?: (message: string) => void;
  debug?: (message: string) => void;
}

interface JsonRpcError {
  code?: number;
  message?: string;
  data?: unknown;
}

interface JsonRpcResponse {
  result?: unknown;
  error?: JsonRpcError;
}

interface JsonRpcHttpResult {
  body: JsonRpcResponse;
  headers: Headers;
}

interface PostJsonRpcOptions {
  sessionId?: string;
  allowEmptyBody?: boolean;
}

interface McpToolDescriptor {
  name?: string;
  description?: string;
  inputSchema?: unknown;
}

function parseSseJsonRpc(bodyText: string): JsonRpcResponse | null {
  const lines = bodyText.split(/\r?\n/);
  const dataChunks: string[] = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line.startsWith("data:")) continue;
    const chunk = line.slice(5).trim();
    if (!chunk || chunk === "[DONE]") continue;
    dataChunks.push(chunk);
  }

  if (dataChunks.length === 0) {
    return null;
  }

  for (const chunk of dataChunks) {
    try {
      const parsed = JSON.parse(chunk);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as JsonRpcResponse;
      }
    } catch {
      // Keep scanning; some events may contain non-JSON payloads before the result event.
    }
  }

  return null;
}

export class GraphitiMcpClient {
  private endpoint: string | null = null;
  private sessionId: string | null = null;
  private toolCache: McpToolDescriptor[] = [];
  private initializePromise: Promise<void> | null = null;
  private toolDiscoveryPromise: Promise<McpToolDescriptor[]> | null = null;
  private requestCounter = 0;

  constructor(
    private readonly config: Pick<GraphitiPluginConfig, "baseUrl" | "timeoutMs" | "transport" | "auth">,
    private readonly logger?: LoggerLike,
  ) {}

  async discoverTools(forceRefresh = false): Promise<McpToolDescriptor[]> {
    if (!forceRefresh && this.toolCache.length > 0) {
      return this.toolCache;
    }

    if (!forceRefresh && this.toolDiscoveryPromise) {
      return this.toolDiscoveryPromise;
    }

    this.toolDiscoveryPromise = (async () => {
      const result = (await this.callMcp("tools/list", {})) as Record<string, unknown>;
      const tools = Array.isArray(result.tools) ? (result.tools as McpToolDescriptor[]) : [];
      this.toolCache = tools;
      return this.toolCache;
    })();

    try {
      return await this.toolDiscoveryPromise;
    } finally {
      this.toolDiscoveryPromise = null;
    }
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<unknown> {
    return this.callMcp("tools/call", { name, arguments: args });
  }

  private async callMcp(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    const endpoint = await this.resolveEndpoint();
    const payload = this.buildRequest(method, params);
    const { body } = await this.postJsonRpc(endpoint, payload, this.config.timeoutMs, {
      sessionId: this.sessionId || undefined,
      allowEmptyBody: false,
    });
    if (body.error) {
      throw new Error(
        `Graphiti MCP ${method} failed: ${body.error.message || "unknown_error"} (code=${String(body.error.code ?? "n/a")})`,
      );
    }
    return body.result;
  }

  private async resolveEndpoint(): Promise<string> {
    if (this.endpoint) {
      return this.endpoint;
    }

    if (this.initializePromise) {
      await this.initializePromise;
      if (this.endpoint) return this.endpoint;
    }

    this.initializePromise = this.initialize();
    try {
      await this.initializePromise;
    } finally {
      this.initializePromise = null;
    }

    if (this.endpoint) return this.endpoint;
    throw new Error("Graphiti MCP endpoint resolution failed without a specific error");
  }

  private async initialize(): Promise<void> {
    const candidates = this.endpointCandidates();
    let lastError: unknown = null;
    for (const candidate of candidates) {
      try {
        await this.performInitialize(candidate);
        this.endpoint = candidate;
        this.logger?.debug?.(`memory-lancedb-pro: graphiti endpoint selected: ${candidate}`);
        return;
      } catch (err) {
        lastError = err;
      }
    }

    throw new Error(
      `Graphiti MCP endpoint unavailable under ${this.config.baseUrl}. Last error: ${String(lastError)}`,
    );
  }

  private async performInitialize(endpoint: string): Promise<void> {
    const initializePayload = this.buildRequest("initialize", {
      protocolVersion: "2025-03-26",
      capabilities: {},
      clientInfo: {
        name: "memory-lancedb-pro",
        version: "1.1.0",
      },
    });

    const initializeResponse = await this.postJsonRpc(
      endpoint,
      initializePayload,
      this.config.timeoutMs,
      { allowEmptyBody: false },
    );

    if (initializeResponse.body.error) {
      throw new Error(
        initializeResponse.body.error.message || "initialize failed",
      );
    }

    const nextSessionId = initializeResponse.headers.get("mcp-session-id");
    this.sessionId = nextSessionId && nextSessionId.trim().length > 0
      ? nextSessionId.trim()
      : null;

    const initializedPayload = this.buildRequest("notifications/initialized", {});
    try {
      const initializedResponse = await this.postJsonRpc(
        endpoint,
        initializedPayload,
        this.config.timeoutMs,
        {
          sessionId: this.sessionId || undefined,
          allowEmptyBody: true,
        },
      );

      if (initializedResponse.body.error) {
        this.logger?.debug?.(
          `memory-lancedb-pro: graphiti notifications/initialized ignored: ${
            initializedResponse.body.error.message || "unknown_error"
          }`,
        );
      }
    } catch (err) {
      this.logger?.debug?.(
        `memory-lancedb-pro: graphiti notifications/initialized unsupported: ${String(err)}`,
      );
    }
  }

  private buildRequest(method: string, params: Record<string, unknown>) {
    const id = `graphiti-${Date.now()}-${this.requestCounter++}`;
    return {
      jsonrpc: "2.0",
      id,
      method,
      params,
    };
  }

  private endpointCandidates(): string[] {
    const normalized = this.config.baseUrl.replace(/\/+$/, "");
    if (this.config.transport === "mcp") {
      return [`${normalized}/mcp`, `${normalized}/mcp/`];
    }
    return [`${normalized}/mcp`, `${normalized}/mcp/`];
  }

  private async postJsonRpc(
    url: string,
    payload: Record<string, unknown>,
    timeoutMs: number,
    options: PostJsonRpcOptions = {},
  ): Promise<JsonRpcHttpResult> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    const headers: Record<string, string> = {
      "content-type": "application/json",
      accept: "application/json, text/event-stream",
      ...this.resolveAuthHeaders(),
    };
    if (options.sessionId) {
      headers["mcp-session-id"] = options.sessionId;
    }

    try {
      const response = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText}`);
      }

      const bodyText = await response.text();
      if (!bodyText.trim()) {
        if (options.allowEmptyBody) {
          return { body: {}, headers: response.headers };
        }
        throw new Error("Empty JSON-RPC response body");
      }

      let parsed: unknown;
      try {
        parsed = JSON.parse(bodyText);
      } catch (err) {
        const parsedSse = parseSseJsonRpc(bodyText);
        if (parsedSse) {
          return {
            body: parsedSse,
            headers: response.headers,
          };
        }
        throw new Error(
          `Invalid JSON-RPC response body: ${String(err)}`,
        );
      }

      if (Array.isArray(parsed)) {
        const first = parsed.find((item) => item && typeof item === "object");
        if (!first) {
          if (options.allowEmptyBody) return { body: {}, headers: response.headers };
          throw new Error("Invalid JSON-RPC batch response: empty array");
        }
        return {
          body: first as JsonRpcResponse,
          headers: response.headers,
        };
      }

      if (!parsed || typeof parsed !== "object") {
        throw new Error("Invalid JSON-RPC response payload type");
      }

      return {
        body: parsed as JsonRpcResponse,
        headers: response.headers,
      };
    } finally {
      clearTimeout(timer);
    }
  }

  private resolveAuthHeaders(): Record<string, string> {
    const auth = this.config.auth;
    if (!auth) return {};

    const tokenFromConfig =
      typeof auth.token === "string" && auth.token.trim().length > 0
        ? auth.token.trim()
        : undefined;
    const tokenFromEnv =
      typeof auth.tokenEnv === "string" && auth.tokenEnv.trim().length > 0
        ? process.env[auth.tokenEnv.trim()]
        : undefined;
    const token = tokenFromConfig ||
      (typeof tokenFromEnv === "string" && tokenFromEnv.trim().length > 0
        ? tokenFromEnv.trim()
        : undefined);
    if (!token) return {};

    const headerName =
      typeof auth.headerName === "string" && auth.headerName.trim().length > 0
        ? auth.headerName.trim()
        : "authorization";
    const value =
      headerName.toLowerCase() === "authorization" && !/^(bearer|token)\s+/i.test(token)
        ? `Bearer ${token}`
        : token;
    return {
      [headerName]: value,
    };
  }
}
