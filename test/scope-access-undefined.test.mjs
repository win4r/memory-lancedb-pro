import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const { MemoryScopeManager } = jiti("../src/scopes.ts");

describe("MemoryScopeManager - System & Reflection Scopes", () => {
  let manager;
  const config = {
    default: "global",
    agentAccess: {}
  };

  beforeEach(() => {
    manager = new MemoryScopeManager(config);
  });

  describe("isBuiltInScope", () => {
    it("should recognize reflection: prefix as built-in", () => {
      assert.strictEqual(manager.isBuiltInScope("reflection:agent:main"), true);
      assert.strictEqual(manager.isBuiltInScope("global"), true);
    });
  });

  describe("System/Admin Bypass (agentId identifier)", () => {
    const bypassIds = ["undefined", "system", "", undefined];
    
    bypassIds.forEach(id => {
      it(`should allow any valid scope when agentId is '${id}'`, () => {
        assert.strictEqual(manager.isAccessible("global", id), true);
        assert.strictEqual(manager.isAccessible("reflection:agent:main", id), true);
        assert.strictEqual(manager.isAccessible("agent:any-agent", id), true);
      });

      it(`should return empty accessible scopes for '${id}' to bypass filter`, () => {
        assert.deepStrictEqual(manager.getAccessibleScopes(id), []);
      });

      it(`should return default config scope when agentId is '${id}'`, () => {
        assert.strictEqual(manager.getDefaultScope(id), "global");
      });
    });
  });

  describe("Reflection scope access for specific agents", () => {
    it("should allow an agent to access its own reflection scope automatically", () => {
      assert.strictEqual(manager.isAccessible("reflection:agent:main", "main"), true);
      assert.strictEqual(manager.isAccessible("reflection:agent:sub-agent", "sub-agent"), true);
    });

    it("should not allow an agent to access another agent's reflection scope by default", () => {
      // Note: Current implementation treats reflection: as a general built-in but does it restrict to ID?
      // Re-checking getAccessibleScopes: it includes reflectionScope = SCOPE_PATTERNS.REFLECTION(agentId)
      // So reflection:agent:other should NOT be in accessibleScopes for agent "main"
      assert.strictEqual(manager.isAccessible("reflection:agent:other", "main"), false);
    });
  });

  describe("validateScope", () => {
    it("should validate reflection scope format", () => {
      assert.strictEqual(manager.validateScope("reflection:agent:main"), true);
      assert.strictEqual(manager.validateScope("reflection:anything"), true);
    });
  });
});
