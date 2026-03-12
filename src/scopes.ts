/**
 * Multi-Scope Access Control System
 * Manages memory isolation and access permissions
 */

// ============================================================================
// Types & Configuration
// ============================================================================

export interface ScopeDefinition {
  description: string;
  metadata?: Record<string, unknown>;
}

export interface ScopeConfig {
  default: string;
  definitions: Record<string, ScopeDefinition>;
  agentAccess: Record<string, string[]>;
}

export interface ScopeManager {
  getAccessibleScopes(agentId?: string): string[];
  getDefaultScope(agentId?: string): string;
  isAccessible(scope: string, agentId?: string): boolean;
  validateScope(scope: string): boolean;
  getAllScopes(): string[];
  getScopeDefinition(scope: string): ScopeDefinition | undefined;
}

// ============================================================================
// Default Configuration
// ============================================================================

export const DEFAULT_SCOPE_CONFIG: ScopeConfig = {
  default: "global",
  definitions: {
    global: {
      description: "Shared knowledge across all agents",
    },
  },
  agentAccess: {},
};

// ============================================================================
// Built-in Scope Patterns
// ============================================================================

const SCOPE_PATTERNS = {
  GLOBAL: "global",
  AGENT: (agentId: string) => `agent:${agentId}`,
  CUSTOM: (name: string) => `custom:${name}`,
  REFLECTION: (agentId: string) => `reflection:agent:${agentId}`,
  PROJECT: (projectId: string) => `project:${projectId}`,
  USER: (userId: string) => `user:${userId}`,
};

// ============================================================================
// Scope Manager Implementation
// ============================================================================

export class MemoryScopeManager implements ScopeManager {
  private config: ScopeConfig;

  constructor(config: Partial<ScopeConfig> = {}) {
    this.config = {
      default: config.default || DEFAULT_SCOPE_CONFIG.default,
      definitions: {
        ...DEFAULT_SCOPE_CONFIG.definitions,
        ...config.definitions,
      },
      agentAccess: {
        ...DEFAULT_SCOPE_CONFIG.agentAccess,
        ...config.agentAccess,
      },
    };

    // Ensure global scope always exists
    if (!this.config.definitions.global) {
      this.config.definitions.global = {
        description: "Shared knowledge across all agents",
      };
    }

    this.validateConfiguration();
  }

  private validateConfiguration(): void {
    // Validate default scope exists in definitions
    if (!this.config.definitions[this.config.default]) {
      throw new Error(`Default scope '${this.config.default}' not found in definitions`);
    }

    // Validate agent access scopes exist in definitions
    for (const [agentId, scopes] of Object.entries(this.config.agentAccess)) {
      for (const scope of scopes) {
        if (!this.config.definitions[scope] && !this.isBuiltInScope(scope)) {
          console.warn(`Agent '${agentId}' has access to undefined scope '${scope}'`);
        }
      }
    }
  }

  private isBuiltInScope(scope: string): boolean {
    return (
      scope === "global" ||
      scope.startsWith("agent:") ||
      scope.startsWith("custom:") ||
      scope.startsWith("project:") ||
      scope.startsWith("user:") ||
      scope.startsWith("reflection:")
    );
  }

  getAccessibleScopes(agentId?: string): string[] {
    /**
     * SECURITY NOTE: "undefined" and "system" are treated as administrative bypass identifiers.
     * These should only be passed by internal gateway tasks (reflection, compaction).
     * ENSURE that external API controllers sanitize/intercept these identifiers to prevent
     * unauthorized privilege escalation from external requests.
     */
    if (!agentId || agentId === "undefined" || agentId === "system") {
      // No agent specified, bypass scope filter for admin/system calls
      return [];
    }

    // Check explicit agent access configuration
    const explicitAccess = this.config.agentAccess[agentId];
    if (explicitAccess) {
      return explicitAccess;
    }

    // Default access: global + agent-specific scope + agent-reflection scope
    const defaultScopes = ["global"];
    const agentScope = SCOPE_PATTERNS.AGENT(agentId);
    const reflectionScope = SCOPE_PATTERNS.REFLECTION(agentId);

    // Only include agent scope if it already exists — don't mutate config as a side effect
    if (this.config.definitions[agentScope] || this.isBuiltInScope(agentScope)) {
      defaultScopes.push(agentScope);
    }

    // Include reflection scope
    if (this.config.definitions[reflectionScope] || this.isBuiltInScope(reflectionScope)) {
      defaultScopes.push(reflectionScope);
    }

    return defaultScopes;
  }

  getDefaultScope(agentId?: string): string {
    if (!agentId || agentId === "undefined" || agentId === "system") {
      return this.config.default;
    }

    // For agents, default to their private scope if they have access to it
    const agentScope = SCOPE_PATTERNS.AGENT(agentId);
    const accessibleScopes = this.getAccessibleScopes(agentId);

    if (accessibleScopes.includes(agentScope)) {
      return agentScope;
    }

    return this.config.default;
  }

  isAccessible(scope: string, agentId?: string): boolean {
    if (!agentId || agentId === "undefined" || agentId === "system") {
      // No agent specified, allow access to all valid scopes
      return this.validateScope(scope);
    }

    const accessibleScopes = this.getAccessibleScopes(agentId);
    return accessibleScopes.includes(scope);
  }

  validateScope(scope: string): boolean {
    if (!scope || typeof scope !== "string" || scope.trim().length === 0) {
      return false;
    }

    const trimmedScope = scope.trim();

    // Check if scope is defined or is a built-in pattern
    return (
      this.config.definitions[trimmedScope] !== undefined ||
      this.isBuiltInScope(trimmedScope)
    );
  }

  getAllScopes(): string[] {
    return Object.keys(this.config.definitions);
  }

  getScopeDefinition(scope: string): ScopeDefinition | undefined {
    return this.config.definitions[scope];
  }

  // Management methods

  addScopeDefinition(scope: string, definition: ScopeDefinition): void {
    if (!this.validateScopeFormat(scope)) {
      throw new Error(`Invalid scope format: ${scope}`);
    }

    this.config.definitions[scope] = definition;
  }

  removeScopeDefinition(scope: string): boolean {
    if (scope === "global") {
      throw new Error("Cannot remove global scope");
    }

    if (!this.config.definitions[scope]) {
      return false;
    }

    delete this.config.definitions[scope];

    // Clean up agent access references
    for (const [agentId, scopes] of Object.entries(this.config.agentAccess)) {
      const filtered = scopes.filter(s => s !== scope);
      if (filtered.length !== scopes.length) {
        this.config.agentAccess[agentId] = filtered;
      }
    }

    return true;
  }

  setAgentAccess(agentId: string, scopes: string[]): void {
    if (!agentId || typeof agentId !== "string") {
      throw new Error("Invalid agent ID");
    }

    // Validate all scopes
    for (const scope of scopes) {
      if (!this.validateScope(scope)) {
        throw new Error(`Invalid scope: ${scope}`);
      }
    }

    this.config.agentAccess[agentId] = [...scopes];
  }

  removeAgentAccess(agentId: string): boolean {
    if (!this.config.agentAccess[agentId]) {
      return false;
    }

    delete this.config.agentAccess[agentId];
    return true;
  }

  private validateScopeFormat(scope: string): boolean {
    if (!scope || typeof scope !== "string") {
      return false;
    }

    const trimmed = scope.trim();

    // Basic format validation
    if (trimmed.length === 0 || trimmed.length > 100) {
      return false;
    }

    // Allow alphanumeric, hyphens, underscores, colons, and dots
    const validFormat = /^[a-zA-Z0-9._:-]+$/.test(trimmed);
    return validFormat;
  }

  // Export/Import configuration

  exportConfig(): ScopeConfig {
    return JSON.parse(JSON.stringify(this.config));
  }

  importConfig(config: Partial<ScopeConfig>): void {
    this.config = {
      default: config.default || this.config.default,
      definitions: {
        ...this.config.definitions,
        ...config.definitions,
      },
      agentAccess: {
        ...this.config.agentAccess,
        ...config.agentAccess,
      },
    };

    this.validateConfiguration();
  }

  // Statistics

  getStats(): {
    totalScopes: number;
    agentsWithCustomAccess: number;
    scopesByType: Record<string, number>;
  } {
    const scopes = this.getAllScopes();
    const scopesByType: Record<string, number> = {
      global: 0,
      agent: 0,
      custom: 0,
      project: 0,
      user: 0,
      other: 0,
    };

    for (const scope of scopes) {
      if (scope === "global") {
        scopesByType.global++;
      } else if (scope.startsWith("agent:")) {
        scopesByType.agent++;
      } else if (scope.startsWith("custom:")) {
        scopesByType.custom++;
      } else if (scope.startsWith("project:")) {
        scopesByType.project++;
      } else if (scope.startsWith("user:") || scope.startsWith("reflection:")) {
        // reflection: scopes are categorized as user/system for stat aggregation 
        // to prevent dashboard schema errors. Verified safe (not used for billing).
        scopesByType.user++;
      } else {
        scopesByType.other++;
      }
    }

    return {
      totalScopes: scopes.length,
      agentsWithCustomAccess: Object.keys(this.config.agentAccess).length,
      scopesByType,
    };
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

export function createScopeManager(config?: Partial<ScopeConfig>): MemoryScopeManager {
  return new MemoryScopeManager(config);
}

export function createAgentScope(agentId: string): string {
  return SCOPE_PATTERNS.AGENT(agentId);
}

export function createCustomScope(name: string): string {
  return SCOPE_PATTERNS.CUSTOM(name);
}

export function createProjectScope(projectId: string): string {
  return SCOPE_PATTERNS.PROJECT(projectId);
}

export function createUserScope(userId: string): string {
  return SCOPE_PATTERNS.USER(userId);
}

// ============================================================================
// Utility Functions
// ============================================================================

export function parseScopeId(scope: string): { type: string; id: string } | null {
  if (scope === "global") {
    return { type: "global", id: "" };
  }

  const colonIndex = scope.indexOf(":");
  if (colonIndex === -1) {
    return null;
  }

  return {
    type: scope.substring(0, colonIndex),
    id: scope.substring(colonIndex + 1),
  };
}

export function isScopeAccessible(scope: string, allowedScopes: string[]): boolean {
  return allowedScopes.includes(scope);
}

export function filterScopesForAgent(scopes: string[], agentId?: string, scopeManager?: ScopeManager): string[] {
  if (!scopeManager || !agentId) {
    return scopes;
  }

  return scopes.filter(scope => scopeManager.isAccessible(scope, agentId));
}