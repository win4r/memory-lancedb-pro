/**
 * B6: Context-aware SupportInfo E2E tests.
 * 
 * 3 scenarios testing that context_label flows from LLM through the full pipeline
 * into actual stored metadata.
 *
 * Run: node test/context-support-e2e.mjs
 */
import assert from "node:assert/strict";
import http from "node:http";
import { mkdtempSync, rmSync } from "node:fs";
import Module from "node:module";
import { tmpdir } from "node:os";
import path from "node:path";

import jitiFactory from "jiti";

process.env.NODE_PATH = [
    process.env.NODE_PATH,
    "/opt/homebrew/lib/node_modules/openclaw/node_modules",
    "/opt/homebrew/lib/node_modules",
].filter(Boolean).join(":");
Module._initPaths();

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const plugin = jiti("../index.ts");
const { MemoryStore } = jiti("../src/store.ts");
const { createEmbedder } = jiti("../src/embedder.ts");
const {
    buildSmartMetadata,
    stringifySmartMetadata,
    parseSmartMetadata,
    buildInitialProvenance,
    buildInitialDecision,
    createSourceRecord,
    updateSupportStats,
} = jiti("../src/smart-metadata.ts");

const EMBEDDING_DIMENSIONS = 2560;

function createDeterministicEmbedding(text, dimensions = EMBEDDING_DIMENSIONS) {
    // All texts → same unit vector. Guarantees cosine sim ~1.0 so dedup matches seed.
    void text;
    const value = 1 / Math.sqrt(dimensions);
    return new Array(dimensions).fill(value);
}

function createEmbeddingServer() {
    return http.createServer(async (req, res) => {
        if (req.method !== "POST" || req.url !== "/v1/embeddings") {
            res.writeHead(404);
            res.end();
            return;
        }
        const chunks = [];
        for await (const chunk of req) chunks.push(chunk);
        const payload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
        const inputs = Array.isArray(payload.input) ? payload.input : [payload.input];
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
            object: "list",
            data: inputs.map((input, index) => ({
                object: "embedding",
                index,
                embedding: createDeterministicEmbedding(String(input)),
            })),
            model: "mock-embedding-model",
            usage: { prompt_tokens: 0, total_tokens: 0 },
        }));
    });
}

function createMockApi(dbPath, embeddingBaseURL, llmBaseURL, logs) {
    return {
        pluginConfig: {
            dbPath,
            autoCapture: true,
            autoRecall: false,
            smartExtraction: true,
            extractMinMessages: 2,
            embedding: {
                apiKey: "dummy",
                model: "qwen3-embedding-4b",
                baseURL: embeddingBaseURL,
                dimensions: EMBEDDING_DIMENSIONS,
            },
            llm: {
                apiKey: "dummy",
                model: "mock-memory-model",
                baseURL: llmBaseURL,
            },
            retrieval: {
                mode: "hybrid",
                minScore: 0.1,
                hardMinScore: 0.1,
                candidatePoolSize: 12,
            },
            scopes: {
                default: "global",
                definitions: {
                    global: { description: "shared" },
                    "agent:life": { description: "life private" },
                },
                agentAccess: {
                    life: ["global", "agent:life"],
                },
            },
        },
        hooks: {},
        toolFactories: {},
        services: [],
        logger: {
            info(...args) { logs.push(["info", args.join(" ")]); },
            warn(...args) { logs.push(["warn", args.join(" ")]); },
            error(...args) { logs.push(["error", args.join(" ")]); },
            debug(...args) { logs.push(["debug", args.join(" ")]); },
        },
        resolvePath(value) { return value; },
        registerTool(toolOrFactory, meta) {
            this.toolFactories[meta.name] =
                typeof toolOrFactory === "function" ? toolOrFactory : () => toolOrFactory;
        },
        registerCli() { },
        registerService(service) { this.services.push(service); },
        on(name, handler) { this.hooks[name] = handler; },
        registerHook(name, handler) { this.hooks[name] = handler; },
    };
}

async function seedMemory(dbPath, text, seedMeta = {}) {
    const store = new MemoryStore({ dbPath, vectorDim: EMBEDDING_DIMENSIONS });
    const embedder = createEmbedder({
        provider: "openai-compatible",
        apiKey: "dummy",
        model: "qwen3-embedding-4b",
        baseURL: process.env.TEST_EMBEDDING_BASE_URL,
        dimensions: EMBEDDING_DIMENSIONS,
    });

    const vector = await embedder.embedPassage(text);
    const src = createSourceRecord({ type: "manual", excerpt: text.slice(0, 200) });
    const metadata = stringifySmartMetadata(
        buildSmartMetadata(
            { text, category: "preference", importance: 0.8 },
            {
                schema_version: 2,
                l0_abstract: text,
                l1_overview: `- ${text}`,
                l2_content: text,
                memory_category: "preferences",
                tier: "working",
                confidence: 0.8,
                provenance: buildInitialProvenance(src),
                decision: buildInitialDecision({
                    actor: "user",
                    reason: "Seeded for test",
                    sourceIds: [src.source_id],
                }),
                support: { global_strength: 0.5, total_observations: 0, slices: [] },
                ...seedMeta,
            },
        ),
    );
    return store.store({
        text,
        vector,
        category: "preference",
        scope: "agent:life",
        importance: 0.8,
        metadata,
    });
}

async function runContextScenario(scenario) {
    const workDir = mkdtempSync(path.join(tmpdir(), `memory-ctx-${scenario.name}-`));
    const dbPath = path.join(workDir, "db");
    const logs = [];
    const embeddingServer = createEmbeddingServer();

    const server = http.createServer(async (req, res) => {
        if (req.method !== "POST" || req.url !== "/chat/completions") {
            res.writeHead(404);
            res.end();
            return;
        }
        const chunks = [];
        for await (const chunk of req) chunks.push(chunk);
        const payload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
        const prompt = payload.messages?.[1]?.content || payload.messages?.[0]?.content || "";

        let content;
        if (prompt.includes("Analyze the following session context")) {
            // Extraction prompt → return the scenario's candidate
            content = JSON.stringify({
                memories: [scenario.candidate],
            });
        } else if (prompt.includes("Determine how to handle this candidate memory")) {
            // Dedup prompt → return the scenario's decision
            content = JSON.stringify(scenario.dedupResponse);
        } else if (prompt.includes("Merge the following memory")) {
            // Merge prompt (for contextualize/refine)
            content = JSON.stringify(scenario.mergeResponse || {
                abstract: scenario.candidate.abstract,
                overview: scenario.candidate.overview || "",
                content: scenario.candidate.content || "",
            });
        } else {
            content = JSON.stringify({ memories: [] });
        }

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
            id: "chatcmpl-test",
            object: "chat.completion",
            created: Math.floor(Date.now() / 1000),
            model: "mock-memory-model",
            choices: [{
                index: 0,
                message: { role: "assistant", content },
                finish_reason: "stop",
            }],
        }));
    });

    await new Promise(r => embeddingServer.listen(0, "127.0.0.1", r));
    await new Promise(r => server.listen(0, "127.0.0.1", r));
    const embeddingPort = embeddingServer.address().port;
    const port = server.address().port;
    process.env.TEST_EMBEDDING_BASE_URL = `http://127.0.0.1:${embeddingPort}/v1`;

    try {
        const api = createMockApi(
            dbPath,
            `http://127.0.0.1:${embeddingPort}/v1`,
            `http://127.0.0.1:${port}`,
            logs,
        );
        plugin.register(api);

        // Seed the existing memory
        await seedMemory(dbPath, scenario.seedText, scenario.seedMeta || {});

        // Trigger extraction via agent_end
        await api.hooks.agent_end(
            {
                success: true,
                sessionKey: "agent:life:test",
                messages: scenario.messages,
            },
            { agentId: "life", sessionKey: "agent:life:test" },
        );

        // Read back all entries
        const freshStore = new MemoryStore({ dbPath, vectorDim: EMBEDDING_DIMENSIONS });
        const entries = await freshStore.list(["agent:life"], undefined, 20, 0);

        return { entries, logs };
    } finally {
        delete process.env.TEST_EMBEDDING_BASE_URL;
        await new Promise(r => embeddingServer.close(r));
        await new Promise(r => server.close(r));
        rmSync(workDir, { recursive: true, force: true });
    }
}

console.log("=== B6: Complex Preference E2E Tests ===\n");

// ---------------------------------------------------------------
// Scenario 1: SUPPORT + context_label=evening
// Existing: "喜欢乌龙茶"
// Evidence: "晚上还是喜欢乌龙茶" → support + context_label=evening
// Expected: Same memory, support.slices has "evening" with 1 confirmation
// ---------------------------------------------------------------
{
    const result = await runContextScenario({
        name: "support-evening",
        seedText: "饮品偏好：喜欢乌龙茶",
        messages: [
            { role: "user", content: "我晚上还是喜欢喝乌龙茶。" },
            { role: "user", content: "这个偏好一直没变。" },
            { role: "user", content: "请记住这个。" },
            { role: "user", content: "好的。" },
        ],
        candidate: {
            category: "preferences",
            abstract: "饮品偏好：晚上喜欢乌龙茶",
            overview: "- 晚上喜欢乌龙茶",
            content: "用户晚上还是喜欢喝乌龙茶。",
        },
        dedupResponse: {
            decision: "support",
            match_index: 1,
            reason: "Same preference reconfirmed in evening context",
            context_label: "evening",
        },
    });

    // Should still be 1 entry (support doesn't create new)
    assert.equal(result.entries.length, 1, "Support should not create new entry");

    const meta = parseSmartMetadata(result.entries[0].metadata, result.entries[0]);
    assert.ok(meta.support, "Support info should exist");
    assert.ok(meta.support.slices.length >= 1, "Should have at least 1 support slice");

    const eveningSlice = meta.support.slices.find(s => s.context === "evening");
    assert.ok(eveningSlice, "Should have an 'evening' context slice");
    assert.equal(eveningSlice.confirmations, 1, "Evening slice should have 1 confirmation");
    assert.equal(eveningSlice.contradictions, 0, "Evening slice should have 0 contradictions");

    // Provenance should have 2 sources now (seed + support)
    assert.ok(meta.provenance, "Provenance should exist");
    assert.ok(meta.provenance.evidence_count >= 2, "Should have at least 2 evidence sources");

    // Decision trail should have support entry
    assert.ok(meta.decision, "Decision should exist");
    const supportAction = meta.decision.history.find(h => h.action === "supported");
    assert.ok(supportAction, "Should have a 'supported' action in decision trail");

    console.log("✅ Scenario 1: SUPPORT + evening context — passed");
}

// ---------------------------------------------------------------
// Scenario 2: CONTEXTUALIZE + context_label=night
// Existing: "通常喜欢拿铁"
// Evidence: "晚上更偏茶" → contextualize + context_label=night
// Expected: Old claim preserved, new claim created with contextualizes relation
//           and claim.contexts = ["night"]
// ---------------------------------------------------------------
{
    const result = await runContextScenario({
        name: "contextualize-night",
        seedText: "饮品偏好：通常喜欢拿铁",
        messages: [
            { role: "user", content: "不过晚上的话我更偏茶。" },
            { role: "user", content: "拿铁白天喝就行。" },
            { role: "user", content: "记一下。" },
            { role: "user", content: "好的。" },
        ],
        candidate: {
            category: "preferences",
            abstract: "饮品偏好：晚上更偏茶",
            overview: "- 晚上偏好茶而非拿铁",
            content: "用户晚上更偏向喝茶，拿铁保留在白天。",
        },
        dedupResponse: {
            decision: "contextualize",
            match_index: 1,
            reason: "Adds night-time context to existing coffee preference",
            context_label: "night",
        },
    });

    // Should be 2 entries: original + contextualized new claim
    assert.equal(result.entries.length, 2, "Contextualize should create a new entry (total 2)");

    // Find the new entry (the one about tea)
    const newEntry = result.entries.find(e => e.text.includes("茶"));
    assert.ok(newEntry, "New contextualizing entry should exist");

    const newMeta = parseSmartMetadata(newEntry.metadata, newEntry);

    // Claim should have contexts: ["night"]
    assert.ok(newMeta.claim, "New entry should have a claim");
    assert.equal(newMeta.claim.stability, "situational", "Contextualized claim should be situational");
    assert.ok(newMeta.claim.contexts, "Claim should have contexts");
    assert.ok(newMeta.claim.contexts.includes("night"), "Claim contexts should include 'night'");

    // Relations should point to original
    assert.ok(newMeta.relations, "Should have relations");
    const ctxRelation = newMeta.relations.find(r => r.relation === "contextualizes");
    assert.ok(ctxRelation, "Should have a 'contextualizes' relation");

    // Original entry should still exist with its text
    const originalEntry = result.entries.find(e => e.text.includes("拿铁"));
    assert.ok(originalEntry, "Original entry should still exist");
    assert.ok(originalEntry.text.includes("拿铁"), "Original text should be preserved");

    console.log("✅ Scenario 2: CONTEXTUALIZE + night context — passed");
}

// ---------------------------------------------------------------
// Scenario 3: CONTRADICT + context_label=weekend
// Existing: "周末喜欢跑步" (with general support)
// Evidence: "最近周末不跑步了" → contradict + context_label=weekend
// Expected: New contradicting claim created, old memory's weekend slice
//           gets 1 contradiction, other slices unaffected
// ---------------------------------------------------------------
{
    const result = await runContextScenario({
        name: "contradict-weekend",
        seedText: "运动偏好：周末喜欢跑步",
        seedMeta: {
            support: updateSupportStats(undefined, "support", "general"),
        },
        messages: [
            { role: "user", content: "最近周末不怎么跑步了。" },
            { role: "user", content: "改打羽毛球了。" },
            { role: "user", content: "记一下。" },
            { role: "user", content: "没问题。" },
        ],
        candidate: {
            category: "preferences",
            abstract: "运动偏好：最近周末不跑步，改打羽毛球",
            overview: "- 周末不跑步了\n- 改打羽毛球",
            content: "用户最近周末不再跑步，改为打羽毛球。",
        },
        dedupResponse: {
            decision: "contradict",
            match_index: 1,
            reason: "Weekend exercise preference changed from running to badminton",
            context_label: "weekend",
        },
    });

    // Should be 2 entries: original + contradicting new claim
    assert.equal(result.entries.length, 2, "Contradict should create a new entry (total 2)");

    // Find entries: new is the one about 羽毛球, original is the other
    const newEntry = result.entries.find(e => e.text.includes("羽毛球"));
    assert.ok(newEntry, "New contradicting entry should exist");

    const originalEntry = result.entries.find(e => e !== newEntry);
    assert.ok(originalEntry, "Original entry should still exist");

    const originalMeta = parseSmartMetadata(originalEntry.metadata, originalEntry);

    // Original should have weekend contradiction
    assert.ok(originalMeta.support, "Original should have support info");

    // Check the general slice is unchanged
    const generalSlice = originalMeta.support.slices.find(s => s.context === "general");
    assert.ok(generalSlice, "Original should still have general slice");
    assert.equal(generalSlice.confirmations, 1, "General confirmations should be unchanged");
    assert.equal(generalSlice.contradictions, 0, "General contradictions should be unchanged");

    // Check the weekend slice has the contradiction
    const weekendSlice = originalMeta.support.slices.find(s => s.context === "weekend");
    assert.ok(weekendSlice, "Original should have weekend slice from contradiction");
    assert.equal(weekendSlice.contradictions, 1, "Weekend slice should have 1 contradiction");

    // Decision trail should show contradicted action
    assert.ok(originalMeta.decision, "Original should have decision info");
    const contradictAction = originalMeta.decision.history.find(h => h.action === "contradicted");
    assert.ok(contradictAction, "Should have 'contradicted' action in original's decision trail");

    // New entry (already found above) should have claim.contexts = ["weekend"]
    const newMeta = parseSmartMetadata(newEntry.metadata, newEntry);
    assert.ok(newMeta.claim, "New entry should have a claim");
    assert.ok(newMeta.claim.contexts, "New claim should have contexts");
    assert.ok(
        newMeta.claim.contexts.includes("weekend"),
        "New claim contexts should include 'weekend'",
    );

    // New entry should have a contradicts relation
    assert.ok(newMeta.relations, "New entry should have relations");
    const contradictRelation = newMeta.relations.find(r => r.relation === "contradicts");
    assert.ok(contradictRelation, "Should have a 'contradicts' relation");

    console.log("✅ Scenario 3: CONTRADICT + weekend context — passed");
}

console.log("\n=== All B6 Complex Preference E2E tests passed! ===");
