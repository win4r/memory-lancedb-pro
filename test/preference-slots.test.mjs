import assert from "node:assert/strict";
import { test } from "node:test";
import Module from "node:module";

process.env.NODE_PATH = [
  process.env.NODE_PATH,
  "/opt/homebrew/lib/node_modules/openclaw/node_modules",
  "/opt/homebrew/lib/node_modules",
].filter(Boolean).join(":");
Module._initPaths();

import jitiFactory from "jiti";
const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const {
  parseBrandItemPreference,
  inferAtomicBrandItemPreferenceSlot,
  normalizePreferenceToken,
} = jiti("../src/preference-slots.ts");

// ---------------------------------------------------------------------------
// parseBrandItemPreference
// ---------------------------------------------------------------------------

test("parseBrandItemPreference: Chinese single-item", () => {
  const result = parseBrandItemPreference("喜欢麦当劳的麦辣鸡翅");
  assert.ok(result);
  assert.equal(result.brand, "麦当劳");
  assert.deepEqual(result.items, ["麦辣鸡翅"]);
  assert.equal(result.aggregate, false);
});

test("parseBrandItemPreference: Chinese aggregate (multiple items)", () => {
  const result = parseBrandItemPreference("喜欢麦当劳的麦旋风、鸡翅和鸡腿堡");
  assert.ok(result);
  assert.equal(result.brand, "麦当劳");
  assert.ok(result.items.length > 1);
  assert.equal(result.aggregate, true);
});

test("parseBrandItemPreference: Chinese verb variants", () => {
  for (const verb of ["爱吃", "偏爱", "常吃", "想吃"]) {
    const result = parseBrandItemPreference(`${verb}麦当劳的薯条`);
    assert.ok(result, `should parse with verb "${verb}"`);
    assert.equal(result.brand, "麦当劳");
  }
});

test("parseBrandItemPreference: English pattern", () => {
  const result = parseBrandItemPreference("I love fries from McDonald's");
  assert.ok(result);
  assert.equal(result.brand, "mcdonald's");
  assert.ok(result.items.length >= 1);
});

test("parseBrandItemPreference: non-preference text returns null", () => {
  assert.equal(parseBrandItemPreference("今天天气不错"), null);
  assert.equal(parseBrandItemPreference("Hello world"), null);
  assert.equal(parseBrandItemPreference("记住我的地址是北京"), null);
});

test("parseBrandItemPreference: stops at reason clause", () => {
  const result = parseBrandItemPreference("喜欢麦当劳的薯条因为很好吃");
  assert.ok(result);
  assert.deepEqual(result.items, ["薯条"]);
  assert.equal(result.aggregate, false);
});

// ---------------------------------------------------------------------------
// inferAtomicBrandItemPreferenceSlot
// ---------------------------------------------------------------------------

test("inferAtomicBrandItemPreferenceSlot: single item returns slot", () => {
  const slot = inferAtomicBrandItemPreferenceSlot("喜欢麦当劳的麦辣鸡翅");
  assert.ok(slot);
  assert.equal(slot.type, "brand-item");
  assert.equal(slot.brand, "麦当劳");
  assert.equal(slot.item, "麦辣鸡翅");
});

test("inferAtomicBrandItemPreferenceSlot: aggregate returns null", () => {
  const slot = inferAtomicBrandItemPreferenceSlot("喜欢麦当劳的麦旋风、鸡翅和鸡腿堡");
  assert.equal(slot, null);
});

test("inferAtomicBrandItemPreferenceSlot: non-preference returns null", () => {
  assert.equal(inferAtomicBrandItemPreferenceSlot("今天天气不错"), null);
});

// ---------------------------------------------------------------------------
// normalizePreferenceToken
// ---------------------------------------------------------------------------

test("normalizePreferenceToken: strips punctuation and lowercases", () => {
  assert.equal(normalizePreferenceToken("  McDonald's!  "), "mcdonald's");
  assert.equal(normalizePreferenceToken("\u201C麦辣鸡翅\u201D"), "麦辣鸡翅");
});

// ---------------------------------------------------------------------------
// category guard integration (verify "preferences" plural is used)
// ---------------------------------------------------------------------------

test("smart-extractor uses 'preferences' (plural) for category check", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(
    new URL("../src/smart-extractor.ts", import.meta.url),
    "utf8",
  );
  // Must contain the plural form in the preference-slot guard
  assert.ok(
    src.includes('candidate.category === "preferences"'),
    'Guard should check for "preferences" (plural), not "preference" (singular)',
  );
  // Must NOT contain the singular typo
  assert.ok(
    !src.includes('candidate.category === "preference"'),
    'Should not have the singular "preference" typo',
  );
});
