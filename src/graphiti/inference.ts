import type { Embedder } from "../embedder.js";
import type { MdMirrorWriter } from "../tools.js";
import type { MemoryStore } from "../store.js";
import type { GraphitiBridge } from "./bridge.js";
import type { GraphitiSyncService } from "./sync.js";
import type { GraphitiPluginConfig } from "./types.js";
import { inferCandidatesFromFacts } from "./reflection.js";

interface LoggerLike {
  info?: (message: string) => void;
  warn?: (message: string) => void;
}

interface InferenceJobDependencies {
  store: MemoryStore;
  embedder: Embedder;
  graphitiBridge?: GraphitiBridge;
  graphitiSync: GraphitiSyncService;
  graphitiConfig?: GraphitiPluginConfig;
  mdMirror?: MdMirrorWriter | null;
  logger?: LoggerLike;
}

interface InferenceJobRunOptions {
  reason: string;
  dryRun?: boolean;
  includeScopes?: string[];
  excludeScopes?: string[];
  forceRun?: boolean;
}

interface InferenceJobSummary {
  reason: string;
  dryRun: boolean;
  scopesScanned: number;
  scopeFilterApplied: string[];
  candidates: number;
  stored: number;
  skippedDuplicate: number;
}

export function createGraphInferenceJob(deps: InferenceJobDependencies) {
  let running = false;

  return async (options: InferenceJobRunOptions): Promise<InferenceJobSummary> => {
    const cfg = deps.graphitiConfig;
    const dryRun = options.dryRun === true;
    const forceRun = options.forceRun === true;

    if (!cfg?.enabled || !deps.graphitiBridge || (!forceRun && !cfg.inference.enabled)) {
      return {
        reason: options.reason,
        dryRun,
        scopesScanned: 0,
        scopeFilterApplied: [],
        candidates: 0,
        stored: 0,
        skippedDuplicate: 0,
      };
    }

    if (running) {
      deps.logger?.info?.("graph-inference: previous run still active, skipping");
      return {
        reason: `${options.reason}:skipped_running`,
        dryRun,
        scopesScanned: 0,
        scopeFilterApplied: [],
        candidates: 0,
        stored: 0,
        skippedDuplicate: 0,
      };
    }

    running = true;
    try {
      const entries = await deps.store.list(undefined, undefined, cfg.inference.maxMemories, 0);
      const eligible = entries.filter((entry) => {
        if (entry.category === "reflection") return false;
        if (/^\[graph-inferred\]/i.test(entry.text.trim())) return false;
        return entry.text.trim().length > 0;
      });

      const scopeMap = new Map<string, typeof eligible>();
      for (const entry of eligible) {
        const bucket = scopeMap.get(entry.scope) ?? [];
        bucket.push(entry);
        scopeMap.set(entry.scope, bucket);
      }

      let scopesScanned = 0;
      let candidates = 0;
      let stored = 0;
      let skippedDuplicate = 0;

      const scopes = resolveEffectiveScopes({
        allScopes: [...scopeMap.keys()],
        includeFromConfig: cfg.inference.includeScopes,
        excludeFromConfig: cfg.inference.excludeScopes,
        includeFromRun: options.includeScopes,
        excludeFromRun: options.excludeScopes,
      }).slice(0, cfg.inference.maxScopes);

      for (const scope of scopes) {
        const rows = (scopeMap.get(scope) || []).slice(0, 10);
        if (rows.length === 0) continue;
        scopesScanned++;

        const query = rows
          .map((row) => row.text.replace(/\s+/g, " ").trim())
          .join("; ")
          .slice(0, 420);
        if (!query) continue;

        const recall = await deps.graphitiBridge.recall({
          scope,
          query,
          limitNodes: cfg.read.topKNodes,
          limitFacts: cfg.read.topKFacts,
        });

        const inferred = inferCandidatesFromFacts(recall.facts.map((fact) => fact.text))
          .filter((candidate) => candidate.confidence >= cfg.inference.minConfidence)
          .slice(0, 8);
        candidates += inferred.length;

        for (const candidate of inferred) {
          const text = `[graph-inferred] ${candidate.text}`;
          const vector = await deps.embedder.embedPassage(text);
          const existing = await deps.store.vectorSearch(vector, 1, 0.1, [scope]);
          if (existing.length > 0 && existing[0].score > 0.97) {
            skippedDuplicate++;
            continue;
          }

          if (dryRun) {
            stored++;
            continue;
          }

          const metadata = JSON.stringify({
            source: "graphiti_inference_job",
            assertionKind: "inferred",
            confidence: candidate.confidence,
            evidenceFact: candidate.evidenceFact,
            scope,
            reason: options.reason,
            runAt: Date.now(),
          });
          const entry = await deps.store.store({
            text,
            vector,
            importance: Math.max(0.45, Math.min(0.75, candidate.confidence)),
            category: "fact",
            scope,
            metadata,
          });

          await deps.graphitiSync.syncMemory(
            {
              id: entry.id,
              text,
              scope,
              category: entry.category,
              metadata: entry.metadata,
            },
            {
              mode: "memoryStore",
              source: "graphiti:inference-job",
              mutation: "graph_inference_job",
              extraMetadata: {
                confidence: candidate.confidence,
                evidenceFact: candidate.evidenceFact,
              },
            },
          );

          if (deps.mdMirror) {
            await deps.mdMirror(
              {
                text,
                category: entry.category,
                scope,
                timestamp: entry.timestamp,
              },
              { source: "graphiti:inference-job" },
            );
          }

          stored++;
        }
      }

      deps.logger?.info?.(
        `graph-inference: reason=${options.reason} dryRun=${dryRun ? "yes" : "no"} scopes=${scopesScanned} candidates=${candidates} stored=${stored} skippedDuplicate=${skippedDuplicate}`,
      );
      return {
        reason: options.reason,
        dryRun,
        scopesScanned,
        scopeFilterApplied: scopes,
        candidates,
        stored,
        skippedDuplicate,
      };
    } catch (err) {
      deps.logger?.warn?.(`graph-inference: run failed (${options.reason}): ${String(err)}`);
      return {
        reason: `${options.reason}:failed`,
        dryRun,
        scopesScanned: 0,
        scopeFilterApplied: [],
        candidates: 0,
        stored: 0,
        skippedDuplicate: 0,
      };
    } finally {
      running = false;
    }
  };
}

function resolveEffectiveScopes(input: {
  allScopes: string[];
  includeFromConfig?: string[];
  excludeFromConfig?: string[];
  includeFromRun?: string[];
  excludeFromRun?: string[];
}): string[] {
  const fromConfigInclude = normalizeScopeList(input.includeFromConfig);
  const fromConfigExclude = normalizeScopeList(input.excludeFromConfig);
  const fromRunInclude = normalizeScopeList(input.includeFromRun);
  const fromRunExclude = normalizeScopeList(input.excludeFromRun);

  const include = fromRunInclude.length > 0 ? fromRunInclude : fromConfigInclude;
  const exclude = new Set([...fromConfigExclude, ...fromRunExclude]);

  const base = include.length > 0
    ? input.allScopes.filter((scope) => include.includes(scope))
    : [...input.allScopes];
  return base.filter((scope) => !exclude.has(scope));
}

function normalizeScopeList(value: string[] | undefined): string[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter((item) => item.length > 0))];
}
