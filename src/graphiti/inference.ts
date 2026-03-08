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
}

interface InferenceJobSummary {
  reason: string;
  scopesScanned: number;
  candidates: number;
  stored: number;
  skippedDuplicate: number;
}

export function createGraphInferenceJob(deps: InferenceJobDependencies) {
  let running = false;

  return async (options: InferenceJobRunOptions): Promise<InferenceJobSummary> => {
    const cfg = deps.graphitiConfig;
    if (!cfg?.enabled || !cfg.inference.enabled || !deps.graphitiBridge) {
      return {
        reason: options.reason,
        scopesScanned: 0,
        candidates: 0,
        stored: 0,
        skippedDuplicate: 0,
      };
    }

    if (running) {
      deps.logger?.info?.("graph-inference: previous run still active, skipping");
      return {
        reason: `${options.reason}:skipped_running`,
        scopesScanned: 0,
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

      const scopes = [...scopeMap.keys()].slice(0, cfg.inference.maxScopes);
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
        `graph-inference: reason=${options.reason} scopes=${scopesScanned} candidates=${candidates} stored=${stored} skippedDuplicate=${skippedDuplicate}`,
      );
      return {
        reason: options.reason,
        scopesScanned,
        candidates,
        stored,
        skippedDuplicate,
      };
    } catch (err) {
      deps.logger?.warn?.(`graph-inference: run failed (${options.reason}): ${String(err)}`);
      return {
        reason: `${options.reason}:failed`,
        scopesScanned: 0,
        candidates: 0,
        stored: 0,
        skippedDuplicate: 0,
      };
    } finally {
      running = false;
    }
  };
}
