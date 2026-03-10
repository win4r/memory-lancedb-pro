/**
 * Tests for V2 Bionic Memory metadata: schema extension, parsing, building.
 *
 * Run:  node test/smart-metadata-v2.mjs
 */
import assert from "node:assert/strict";
import Module from "node:module";
import jitiFactory from "jiti";

process.env.NODE_PATH = [
    process.env.NODE_PATH,
    "/opt/homebrew/lib/node_modules/openclaw/node_modules",
    "/opt/homebrew/lib/node_modules",
].filter(Boolean).join(":");
Module._initPaths();

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const meta = jiti("../src/smart-metadata.ts");

const {
    parseSmartMetadata,
    buildSmartMetadata,
    stringifySmartMetadata,
    toLifecycleMemory,
    getDecayableFromEntry,
    createSourceRecord,
    appendDecisionEntry,
    updateSupportStats,
    buildInitialProvenance,
    buildInitialDecision,
    inferClaimKind,
} = meta;

console.log("=== V2 Smart Metadata Tests ===\n");

// -------------------------------------------------------------------------
// Test 1: Parse V1 metadata → auto-upgrade with defaults
// -------------------------------------------------------------------------
{
    const v1Json = JSON.stringify({
        l0_abstract: "User prefers dark mode",
        l1_overview: "- User prefers dark mode",
        l2_content: "User said they prefer dark mode in all IDEs.",
        memory_category: "preferences",
        tier: "working",
        access_count: 3,
        confidence: 0.85,
        last_accessed_at: 1700000000,
        source_session: "session-abc123",
    });

    const entry = { text: "User prefers dark mode", timestamp: 1700000000 };
    const parsed = parseSmartMetadata(v1Json, entry);

    assert.equal(parsed.schema_version, 1, "V1 should default to schema_version=1");
    assert.equal(parsed.l0_abstract, "User prefers dark mode");
    assert.equal(parsed.confidence, 0.85);
    assert.equal(parsed.memory_category, "preferences");

    // V2 fields should be generated from source_session
    assert.ok(parsed.provenance, "V1 with source_session should generate provenance");
    assert.equal(parsed.provenance.sources.length, 1);
    assert.equal(parsed.provenance.sources[0].type, "auto-capture");
    assert.equal(parsed.provenance.sources[0].session_key, "session-abc123");
    assert.equal(parsed.provenance.evidence_count, 1);

    // No claim, decision, support, or relations from pure V1
    assert.equal(parsed.claim, undefined);
    assert.equal(parsed.decision, undefined);
    assert.equal(parsed.support, undefined);
    assert.equal(parsed.relations, undefined);

    console.log("✅ Test 1: V1 → V2 auto-upgrade passed");
}

// -------------------------------------------------------------------------
// Test 2: Parse V2 metadata → all fields preserved
// -------------------------------------------------------------------------
{
    const v2Json = JSON.stringify({
        schema_version: 2,
        l0_abstract: "User likes lattes",
        l1_overview: "- User likes lattes",
        l2_content: "User mentioned they enjoy lattes.",
        memory_category: "preferences",
        tier: "core",
        access_count: 5,
        confidence: 0.9,
        last_accessed_at: 1700000000,
        source_session: "sess-1",
        claim: {
            kind: "preference",
            subject: "User",
            attribute: "coffee",
            value_summary: "User likes lattes",
            stability: "stable",
            polarity: "positive",
            valid_time: "ongoing",
            contexts: ["general"],
        },
        provenance: {
            sources: [
                {
                    source_id: "src-001",
                    type: "smart-extraction",
                    session_key: "sess-1",
                    timestamp: 1700000000,
                    excerpt: "I love lattes",
                    confidence_hint: 0.8,
                },
            ],
            evidence_count: 1,
            first_observed_at: 1700000000,
            last_observed_at: 1700000000,
        },
        decision: {
            current_reason: "Extracted from conversation",
            history: [
                {
                    action: "created",
                    actor: "llm",
                    timestamp: 1700000000,
                    reason: "Extracted from conversation",
                    source_ids: ["src-001"],
                },
            ],
        },
        support: {
            confirmations: 2,
            contradictions: 0,
            support_strength: 1.0,
        },
        relations: [
            {
                relation: "contextualizes",
                target_id: "mem-old-123",
                strength: 0.8,
                reason: "Adds context about preference timing",
            },
        ],
    });

    const parsed = parseSmartMetadata(v2Json, { text: "User likes lattes", timestamp: 1700000000 });

    assert.equal(parsed.schema_version, 2);
    assert.ok(parsed.claim, "Claim should be parsed");
    assert.equal(parsed.claim.kind, "preference");
    assert.equal(parsed.claim.subject, "User");
    assert.equal(parsed.claim.stability, "stable");
    assert.equal(parsed.claim.polarity, "positive");
    assert.deepStrictEqual(parsed.claim.contexts, ["general"]);

    assert.ok(parsed.provenance, "Provenance should be parsed");
    assert.equal(parsed.provenance.sources.length, 1);
    assert.equal(parsed.provenance.sources[0].source_id, "src-001");
    assert.equal(parsed.provenance.sources[0].type, "smart-extraction");
    assert.equal(parsed.provenance.sources[0].excerpt, "I love lattes");

    assert.ok(parsed.decision, "Decision should be parsed");
    assert.equal(parsed.decision.current_reason, "Extracted from conversation");
    assert.equal(parsed.decision.history.length, 1);
    assert.equal(parsed.decision.history[0].action, "created");

    // V1 support format → auto-migrated to general slice
    assert.ok(parsed.support, "Support should be parsed");
    assert.equal(parsed.support.slices.length, 1, "V1 support should migrate to 1 slice");
    assert.equal(parsed.support.slices[0].context, "general");
    assert.equal(parsed.support.slices[0].confirmations, 2);
    assert.equal(parsed.support.slices[0].contradictions, 0);
    assert.equal(parsed.support.global_strength, 1.0);
    assert.equal(parsed.support.total_observations, 2);

    assert.ok(parsed.relations, "Relations should be parsed");
    assert.equal(parsed.relations.length, 1);
    assert.equal(parsed.relations[0].relation, "contextualizes");
    assert.equal(parsed.relations[0].target_id, "mem-old-123");

    console.log("✅ Test 2: V2 full parse passed");
}

// -------------------------------------------------------------------------
// Test 3: Parse empty / corrupted metadata → no crash
// -------------------------------------------------------------------------
{
    const parsed1 = parseSmartMetadata(undefined, { text: "hello" });
    assert.equal(parsed1.l0_abstract, "hello");
    assert.equal(parsed1.schema_version, 1);
    assert.equal(parsed1.claim, undefined);

    const parsed2 = parseSmartMetadata("not valid json", { text: "world" });
    assert.equal(parsed2.l0_abstract, "world");
    assert.equal(parsed2.schema_version, 1);

    const parsed3 = parseSmartMetadata("null", { text: "test" });
    assert.equal(parsed3.l0_abstract, "test");

    const parsed4 = parseSmartMetadata("{}", { text: "empty" });
    assert.equal(parsed4.l0_abstract, "empty");
    assert.equal(parsed4.schema_version, 1);

    console.log("✅ Test 3: Empty/corrupt metadata → no crash passed");
}

// -------------------------------------------------------------------------
// Test 4: buildSmartMetadata deep-merges V2 structures
// -------------------------------------------------------------------------
{
    const existingMeta = JSON.stringify({
        schema_version: 2,
        l0_abstract: "User likes coffee",
        l1_overview: "- Coffee preference",
        l2_content: "User likes coffee.",
        memory_category: "preferences",
        tier: "working",
        access_count: 1,
        confidence: 0.7,
        last_accessed_at: 1700000000,
        provenance: {
            sources: [
                { source_id: "src-A", type: "smart-extraction", timestamp: 1700000000, session_key: "sess-1" },
            ],
            evidence_count: 1,
            first_observed_at: 1700000000,
            last_observed_at: 1700000000,
        },
        decision: {
            current_reason: "Created",
            history: [
                { action: "created", actor: "llm", timestamp: 1700000000, reason: "Created" },
            ],
        },
    });

    const entry = { text: "User likes coffee", metadata: existingMeta, timestamp: 1700000000 };

    const result = buildSmartMetadata(entry, {
        provenance: {
            sources: [
                { source_id: "src-B", type: "smart-extraction", timestamp: 1700100000, session_key: "sess-2", excerpt: "Still like coffee" },
            ],
            evidence_count: 1,
            first_observed_at: 1700100000,
            last_observed_at: 1700100000,
        },
        decision: {
            current_reason: "Confirmed again",
            history: [
                { action: "supported", actor: "llm", timestamp: 1700100000, reason: "Re-observed" },
            ],
        },
    });

    // Provenance should be merged (both sources)
    assert.ok(result.provenance, "Provenance should exist after merge");
    assert.equal(result.provenance.sources.length, 2, "Should have 2 sources after merge");
    assert.equal(result.provenance.sources[0].source_id, "src-A");
    assert.equal(result.provenance.sources[1].source_id, "src-B");
    assert.equal(result.provenance.evidence_count, 2);
    assert.equal(result.provenance.first_observed_at, 1700000000);
    assert.equal(result.provenance.last_observed_at, 1700100000);

    // Decision should be merged (both history entries)
    assert.ok(result.decision, "Decision should exist after merge");
    assert.equal(result.decision.current_reason, "Confirmed again");
    assert.equal(result.decision.history.length, 2, "Should have 2 decision entries after merge");
    assert.equal(result.decision.history[0].action, "created");
    assert.equal(result.decision.history[1].action, "supported");

    console.log("✅ Test 4: buildSmartMetadata deep-merge passed");
}

// -------------------------------------------------------------------------
// Test 5: V2 helpers — createSourceRecord, appendDecisionEntry, updateSupportStats
// -------------------------------------------------------------------------
{
    // createSourceRecord
    const src = createSourceRecord({
        type: "smart-extraction",
        agentId: "agent-main",
        sessionKey: "sess-xyz",
        excerpt: "I prefer tea",
        confidenceHint: 0.9,
    });
    assert.ok(src.source_id, "Should have a source_id");
    assert.equal(src.type, "smart-extraction");
    assert.equal(src.agent_id, "agent-main");
    assert.equal(src.session_key, "sess-xyz");
    assert.equal(src.excerpt, "I prefer tea");
    assert.equal(src.confidence_hint, 0.9);
    assert.ok(src.timestamp > 0, "Should have a timestamp");

    // appendDecisionEntry
    const decision = appendDecisionEntry(undefined, {
        action: "created",
        actor: "llm",
        timestamp: 1700000000,
        reason: "New claim",
    });
    assert.equal(decision.history.length, 1);
    assert.equal(decision.current_reason, "New claim");

    const decision2 = appendDecisionEntry(decision, {
        action: "supported",
        actor: "llm",
        timestamp: 1700100000,
        reason: "Re-observed",
    });
    assert.equal(decision2.history.length, 2);
    assert.equal(decision2.current_reason, "Re-observed");

    // updateSupportStats — now context-aware
    const stats1 = updateSupportStats(undefined, "support");
    assert.equal(stats1.slices.length, 1);
    assert.equal(stats1.slices[0].context, "general");
    assert.equal(stats1.slices[0].confirmations, 1);
    assert.equal(stats1.slices[0].contradictions, 0);
    assert.equal(stats1.global_strength, 1.0);

    const stats2 = updateSupportStats(stats1, "contradict");
    assert.equal(stats2.slices[0].confirmations, 1);
    assert.equal(stats2.slices[0].contradictions, 1);
    assert.equal(stats2.global_strength, 0.5);

    const stats3 = updateSupportStats(stats2, "support");
    assert.equal(stats3.slices[0].confirmations, 2);
    assert.equal(stats3.slices[0].contradictions, 1);
    assert.ok(Math.abs(stats3.global_strength - 2 / 3) < 0.01);

    console.log("✅ Test 5: V2 helpers passed");
}

// -------------------------------------------------------------------------
// Test 6: Backward compatibility — toLifecycleMemory & getDecayableFromEntry on V2
// -------------------------------------------------------------------------
{
    const v2Meta = JSON.stringify({
        schema_version: 2,
        l0_abstract: "Test memory",
        l1_overview: "- Test",
        l2_content: "Test content",
        memory_category: "patterns",
        tier: "core",
        access_count: 10,
        confidence: 0.95,
        last_accessed_at: 1700000000,
        claim: { kind: "procedure", value_summary: "Test", stability: "stable" },
        provenance: {
            sources: [{ source_id: "x", type: "manual", timestamp: 1700000000 }],
            evidence_count: 1,
            first_observed_at: 1700000000,
            last_observed_at: 1700000000,
        },
        support: { confirmations: 5, contradictions: 0, support_strength: 1.0 },  // V1 format
    });

    const entry = { text: "Test memory", metadata: v2Meta, timestamp: 1700000000, importance: 0.9 };

    const lm = toLifecycleMemory("test-id-123", entry);
    assert.equal(lm.id, "test-id-123");
    assert.equal(lm.importance, 0.9);
    assert.equal(lm.confidence, 0.95);
    assert.equal(lm.tier, "core");
    assert.equal(lm.accessCount, 10);

    const { memory, meta: parsedMeta } = getDecayableFromEntry({ ...entry, id: "test-id-123" });
    assert.equal(memory.id, "test-id-123");
    assert.equal(memory.confidence, 0.95);
    assert.equal(parsedMeta.schema_version, 2);
    assert.ok(parsedMeta.claim, "V2 claim preserved");
    assert.ok(parsedMeta.provenance, "V2 provenance preserved");

    console.log("✅ Test 6: Backward compatibility passed");
}

// -------------------------------------------------------------------------
// Test 7: inferClaimKind
// -------------------------------------------------------------------------
{
    assert.equal(inferClaimKind("profile"), "semantic");
    assert.equal(inferClaimKind("preferences"), "preference");
    assert.equal(inferClaimKind("entities"), "semantic");
    assert.equal(inferClaimKind("events"), "episodic");
    assert.equal(inferClaimKind("cases"), "procedure");
    assert.equal(inferClaimKind("patterns"), "procedure");

    console.log("✅ Test 7: inferClaimKind passed");
}

// -------------------------------------------------------------------------
// Test 8: buildInitialProvenance and buildInitialDecision
// -------------------------------------------------------------------------
{
    const src = createSourceRecord({ type: "manual", timestamp: 1700000000 });
    const prov = buildInitialProvenance(src);
    assert.equal(prov.sources.length, 1);
    assert.equal(prov.evidence_count, 1);
    assert.equal(prov.first_observed_at, 1700000000);

    const dec = buildInitialDecision({
        actor: "user",
        reason: "Manually added",
        sourceIds: [src.source_id],
    });
    assert.equal(dec.history.length, 1);
    assert.equal(dec.history[0].action, "created");
    assert.equal(dec.history[0].actor, "user");
    assert.deepStrictEqual(dec.history[0].source_ids, [src.source_id]);

    console.log("✅ Test 8: buildInitialProvenance/Decision passed");
}

// -------------------------------------------------------------------------
// Test 9: V1 without source_session → no provenance synthesized
// -------------------------------------------------------------------------
{
    const v1Json = JSON.stringify({
        l0_abstract: "Old memory",
        l1_overview: "- Old",
        l2_content: "Old content",
        memory_category: "events",
        tier: "peripheral",
        access_count: 0,
        confidence: 0.5,
        last_accessed_at: 1600000000,
    });

    const parsed = parseSmartMetadata(v1Json, { text: "Old memory", timestamp: 1600000000 });
    assert.equal(parsed.schema_version, 1);
    assert.equal(parsed.provenance, undefined, "No provenance without source_session");

    console.log("✅ Test 9: V1 without source_session → no provenance passed");
}

// -------------------------------------------------------------------------
// Test 10: Provenance dedup on merge
// -------------------------------------------------------------------------
{
    const existingMeta = JSON.stringify({
        provenance: {
            sources: [{ source_id: "dup-1", type: "manual", timestamp: 1700000000 }],
            evidence_count: 1,
            first_observed_at: 1700000000,
            last_observed_at: 1700000000,
        },
    });

    const result = buildSmartMetadata(
        { text: "test", metadata: existingMeta, timestamp: 1700000000 },
        {
            provenance: {
                sources: [
                    { source_id: "dup-1", type: "manual", timestamp: 1700000000 }, // duplicate
                    { source_id: "new-2", type: "smart-extraction", timestamp: 1700100000 },
                ],
                evidence_count: 2,
                first_observed_at: 1700000000,
                last_observed_at: 1700100000,
            },
        },
    );

    assert.ok(result.provenance);
    assert.equal(result.provenance.sources.length, 2, "Duplicate source_id deduped");
    const ids = result.provenance.sources.map((s) => s.source_id);
    assert.ok(ids.includes("dup-1"));
    assert.ok(ids.includes("new-2"));

    console.log("✅ Test 10: Provenance dedup on merge passed");
}

// -------------------------------------------------------------------------
// Test 11: Context-aware SupportInfo — multi-context slices
// -------------------------------------------------------------------------
{
    // Different context creates separate slices
    let stats = updateSupportStats(undefined, "support", "general");
    stats = updateSupportStats(stats, "support", "evening");
    stats = updateSupportStats(stats, "contradict", "evening");

    assert.equal(stats.slices.length, 2, "Should have 2 context slices");

    const general = stats.slices.find(s => s.context === "general");
    const evening = stats.slices.find(s => s.context === "evening");

    assert.ok(general, "Should have general slice");
    assert.equal(general.confirmations, 1);
    assert.equal(general.contradictions, 0);
    assert.equal(general.strength, 1.0);

    assert.ok(evening, "Should have evening slice");
    assert.equal(evening.confirmations, 1);
    assert.equal(evening.contradictions, 1);
    assert.equal(evening.strength, 0.5);

    // Global strength = 2 conf / (2 conf + 1 contra) = 2/3
    assert.ok(Math.abs(stats.global_strength - 2 / 3) < 0.01, `Global strength should be ~0.67, got ${stats.global_strength}`);
    assert.equal(stats.total_observations, 3);

    // Unknown context normalizes to general
    const stats2 = updateSupportStats(undefined, "support", "unknown_context");
    assert.equal(stats2.slices[0].context, "general", "Unknown context should normalize to general");

    console.log("✅ Test 11: Context-aware SupportInfo passed");
}

console.log("\n=== All V2 metadata tests passed! ===");
