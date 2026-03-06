import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import jitiFactory from "jiti";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const pluginSdkStubPath = path.resolve(testDir, "helpers", "openclaw-plugin-sdk-stub.mjs");
const jiti = jitiFactory(import.meta.url, {
  interopDefault: true,
  alias: {
    "openclaw/plugin-sdk": pluginSdkStubPath,
  },
});
const {
  registerSelfImprovementLogTool,
  registerSelfImprovementExtractSkillTool,
} = jiti("../src/tools.ts");

function createToolHarness(workspaceDir) {
  const factories = new Map();
  const api = {
    registerTool(factory, meta) {
      factories.set(meta?.name || "", factory);
    },
  };

  const context = {
    workspaceDir,
    retriever: {},
    store: {},
    scopeManager: {},
    embedder: {},
    mdMirror: null,
  };

  registerSelfImprovementLogTool(api, context);
  registerSelfImprovementExtractSkillTool(api, context);

  return {
    tool(name, toolCtx = {}) {
      const factory = factories.get(name);
      assert.ok(factory, `tool not registered: ${name}`);
      return factory(toolCtx);
    },
  };
}

describe("self-improvement tool file-write flow", () => {
  let workspaceDir;

  beforeEach(() => {
    workspaceDir = mkdtempSync(path.join(tmpdir(), "self-improvement-test-"));
  });

  afterEach(() => {
    rmSync(workspaceDir, { recursive: true, force: true });
  });

  it("handles learning id validation and writes promoted skill scaffold with sanitized outputDir", async () => {
    const harness = createToolHarness(workspaceDir);
    const logTool = harness.tool("self_improvement_log");
    const extractTool = harness.tool("self_improvement_extract_skill");

    const logged = await logTool.execute("tc-1", {
      type: "learning",
      summary: "Use deterministic temp fixtures in tests.",
      details: "Nondeterministic fixture paths caused flaky assertions.",
      suggestedAction: "Always bind fixtures to test-local temp dirs.",
      category: "best_practice",
      area: "tests",
      priority: "high",
    });

    const learningId = logged?.details?.id;
    assert.match(learningId, /^LRN-\d{8}-001$/);

    const invalid = await extractTool.execute("tc-2", {
      learningId: "LRN-INVALID",
      skillName: "deterministic-fixtures",
    });
    assert.equal(invalid?.details?.error, "invalid_learning_id");

    const extracted = await extractTool.execute("tc-3", {
      learningId,
      skillName: "deterministic-fixtures",
      outputDir: "../../outside//skills",
    });

    assert.equal(extracted?.details?.action, "skill_extracted");
    const skillPath = extracted?.details?.skillPath;
    assert.ok(typeof skillPath === "string" && skillPath.length > 0);
    assert.ok(!skillPath.includes(".."), `skillPath must be sanitized: ${skillPath}`);
    assert.ok(!skillPath.startsWith("/"), `skillPath must stay relative: ${skillPath}`);

    const absSkillPath = path.resolve(workspaceDir, skillPath);
    assert.ok(
      absSkillPath.startsWith(path.resolve(workspaceDir) + path.sep),
      `skill file escaped workspace: ${absSkillPath}`
    );

    const skillContent = readFileSync(absSkillPath, "utf-8");
    assert.match(skillContent, /# Deterministic Fixtures/);
    assert.match(skillContent, new RegExp(`Learning ID: ${learningId}`));

    const learningsPath = path.join(workspaceDir, ".learnings", "LEARNINGS.md");
    const learningsBody = readFileSync(learningsPath, "utf-8");
    assert.match(learningsBody, /\*\*Status\*\*:\s*promoted_to_skill/);
    assert.match(learningsBody, /Skill-Path:\s*outside\/skills\/deterministic-fixtures/);
  });
});
