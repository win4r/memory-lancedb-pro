/**
 * Tier Manager — Three-tier memory promotion/demotion system
 *
 * Tiers:
 * - Core (decay floor 0.9): Identity-level facts, almost never forgotten
 * - Working (decay floor 0.7): Active context, ages out without reinforcement
 * - Peripheral (decay floor 0.5): Low-priority or aging memories
 *
 * Promotion: Peripheral → Working → Core (based on access, composite score, importance)
 * Demotion: Core → Working → Peripheral (based on decay, age)
 */

import type { MemoryTier } from "./memory-categories.js";
import type { DecayScore } from "./decay-engine.js";

// ============================================================================
// Types
// ============================================================================

export interface TierConfig {
  /** Minimum access count for Core promotion (default: 10) */
  coreAccessThreshold: number;
  /** Minimum composite decay score for Core promotion (default: 0.7) */
  coreCompositeThreshold: number;
  /** Minimum importance for Core promotion (default: 0.8) */
  coreImportanceThreshold: number;
  /** Composite threshold below which to demote to Peripheral (default: 0.15) */
  peripheralCompositeThreshold: number;
  /** Age in days after which infrequent memories demote to Peripheral (default: 60) */
  peripheralAgeDays: number;
  /** Minimum access count for Working promotion from Peripheral (default: 3) */
  workingAccessThreshold: number;
  /** Minimum composite for Working promotion from Peripheral (default: 0.4) */
  workingCompositeThreshold: number;
}

export const DEFAULT_TIER_CONFIG: TierConfig = {
  coreAccessThreshold: 10,
  coreCompositeThreshold: 0.7,
  coreImportanceThreshold: 0.8,
  peripheralCompositeThreshold: 0.15,
  peripheralAgeDays: 60,
  workingAccessThreshold: 3,
  workingCompositeThreshold: 0.4,
};

export interface TierTransition {
  memoryId: string;
  fromTier: MemoryTier;
  toTier: MemoryTier;
  reason: string;
}

/** Minimal memory fields needed for tier evaluation. */
export interface TierableMemory {
  id: string;
  tier: MemoryTier;
  importance: number;
  accessCount: number;
  createdAt: number;
}

export interface TierManager {
  /**
   * Evaluate whether a memory should change tiers.
   * Returns the transition if a change is needed, null otherwise.
   */
  evaluate(
    memory: TierableMemory,
    decayScore: DecayScore,
    now?: number,
  ): TierTransition | null;

  /**
   * Evaluate multiple memories and return all transitions.
   */
  evaluateAll(
    memories: TierableMemory[],
    decayScores: DecayScore[],
    now?: number,
  ): TierTransition[];
}

// ============================================================================
// Factory
// ============================================================================

const MS_PER_DAY = 86_400_000;

export function createTierManager(
  config: TierConfig = DEFAULT_TIER_CONFIG,
): TierManager {
  function evaluate(
    memory: TierableMemory,
    decayScore: DecayScore,
    now: number = Date.now(),
  ): TierTransition | null {
    const ageDays = (now - memory.createdAt) / MS_PER_DAY;

    switch (memory.tier) {
      case "peripheral": {
        // Promote to Working?
        if (
          memory.accessCount >= config.workingAccessThreshold &&
          decayScore.composite >= config.workingCompositeThreshold
        ) {
          return {
            memoryId: memory.id,
            fromTier: "peripheral",
            toTier: "working",
            reason: `Access count (${memory.accessCount}) >= ${config.workingAccessThreshold} and composite (${decayScore.composite.toFixed(2)}) >= ${config.workingCompositeThreshold}`,
          };
        }
        break;
      }

      case "working": {
        // Promote to Core?
        if (
          memory.accessCount >= config.coreAccessThreshold &&
          decayScore.composite >= config.coreCompositeThreshold &&
          memory.importance >= config.coreImportanceThreshold
        ) {
          return {
            memoryId: memory.id,
            fromTier: "working",
            toTier: "core",
            reason: `High access (${memory.accessCount}), composite (${decayScore.composite.toFixed(2)}), importance (${memory.importance})`,
          };
        }

        // Demote to Peripheral?
        if (
          decayScore.composite < config.peripheralCompositeThreshold ||
          (ageDays > config.peripheralAgeDays &&
            memory.accessCount < config.workingAccessThreshold)
        ) {
          return {
            memoryId: memory.id,
            fromTier: "working",
            toTier: "peripheral",
            reason: `Low composite (${decayScore.composite.toFixed(2)}) or aged ${ageDays.toFixed(0)} days with low access (${memory.accessCount})`,
          };
        }
        break;
      }

      case "core": {
        // Demote to Working? (Core rarely demotes, but it can)
        if (
          decayScore.composite < config.peripheralCompositeThreshold &&
          memory.accessCount < config.workingAccessThreshold
        ) {
          return {
            memoryId: memory.id,
            fromTier: "core",
            toTier: "working",
            reason: `Severely low composite (${decayScore.composite.toFixed(2)}) and access (${memory.accessCount})`,
          };
        }
        break;
      }
    }

    return null;
  }

  return {
    evaluate,

    evaluateAll(memories, decayScores, now = Date.now()) {
      const scoreMap = new Map(decayScores.map((s) => [s.memoryId, s]));
      const transitions: TierTransition[] = [];

      for (const memory of memories) {
        const score = scoreMap.get(memory.id);
        if (!score) continue;

        const transition = evaluate(memory, score, now);
        if (transition) {
          transitions.push(transition);
        }
      }

      return transitions;
    },
  };
}
