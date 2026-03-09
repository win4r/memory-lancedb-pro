/**
 * Test: Chinese noise filter patterns
 *
 * Verifies that Chinese greetings, denials, meta-questions, and boilerplate
 * are correctly identified as noise, while meaningful Chinese content passes through.
 *
 * Usage: node test/noise-filter-chinese.mjs
 */

import assert from "node:assert/strict";
import jitiFactory from "jiti";

const jiti = jitiFactory(import.meta.url, { interopDefault: true });
const { isNoise, filterNoise } = jiti("../src/noise-filter.ts");

// ============================================================================
// Test cases
// ============================================================================

const SHOULD_BE_NOISE = [
  // Chinese denials
  { text: "我没有相关信息", category: "denial" },
  { text: "我没有任何相关记录", category: "denial" },
  { text: "我不确定", category: "denial" },
  { text: "我不太确定这件事", category: "denial" },
  { text: "我不记得了", category: "denial" },
  { text: "我想不起来了", category: "denial" },
  { text: "我没找到相关内容", category: "denial" },
  { text: "找不到相关记忆", category: "denial" },
  { text: "没有相关记忆", category: "denial" },

  // Chinese meta-questions
  { text: "你还记得吗", category: "meta-question" },
  { text: "你记得吗？", category: "meta-question" },
  { text: "你记不记得上次说的", category: "meta-question" },
  { text: "我有没有说过这件事", category: "meta-question" },
  { text: "我之前提到过什么", category: "meta-question" },
  { text: "我是不是告诉过你", category: "meta-question" },
  { text: "我跟你说过这个吗", category: "meta-question" },
  { text: "你知道我说过什么吗", category: "meta-question" },
  { text: "你知道我提过这件事吗", category: "meta-question" },

  // Chinese boilerplate
  { text: "你好", category: "boilerplate" },
  { text: "早上好", category: "boilerplate" },
  { text: "晚上好！", category: "boilerplate" },
  { text: "早安", category: "boilerplate" },
  { text: "晚安", category: "boilerplate" },
  { text: "哈喽", category: "boilerplate" },
  { text: "好的", category: "boilerplate" },
  { text: "行", category: "boilerplate" },  // length < 5, caught by min-length
  { text: "可以", category: "boilerplate" }, // length < 5, caught by min-length
  { text: "没问题", category: "boilerplate" },
  { text: "收到", category: "boilerplate" }, // length < 5, caught by min-length
  { text: "谢谢", category: "boilerplate" }, // length < 5, caught by min-length
  { text: "谢谢你的帮助", category: "boilerplate" },
  { text: "好吧我知道了", category: "boilerplate" },
  { text: "OK", category: "boilerplate" },   // length < 5, caught by min-length
  { text: "明白了", category: "boilerplate" },
  { text: "了解", category: "boilerplate" }, // length < 5, caught by min-length
  { text: "知道了", category: "boilerplate" },
  { text: "新会话开始", category: "boilerplate" },

  // English patterns (should still work)
  { text: "I don't have any information about that", category: "denial" },
  { text: "hello there", category: "boilerplate" },
  { text: "do you remember what I said", category: "meta-question" },
];

const SHOULD_NOT_BE_NOISE = [
  // Meaningful Chinese content that must NOT be filtered
  "我的MacBook Pro电池续航大概8小时",
  "项目部署在阿里云的ECS实例上，用的是2核4G配置",
  "SDDP算法的Level Bundle方法收敛速度比较慢",
  "记得明天下午3点开会讨论需求",
  "API Key是sk-xxx，配置在.env文件里",
  "你好厉害，这个方案解决了我的问题",  // starts with 你好 but has meaningful content after
  "好的方案是使用Redis做缓存层",        // starts with 好的 but has meaningful content
  "谢谢分享，我觉得这个思路很好，可以用在我们的EVRP项目里",
  "你知道Redis的持久化策略吗",            // legitimate question, must NOT be filtered
  "你知道怎么配置Nginx反向代理吗",         // legitimate question, must NOT be filtered
];

// ============================================================================
// Run tests
// ============================================================================

let passed = 0;
let failed = 0;

console.log("## Chinese Noise Filter Test Results\n");

// Test noise detection
console.log("### Should be detected as noise\n");
console.log("| Text | Category | Result |");
console.log("|------|----------|--------|");

for (const { text, category } of SHOULD_BE_NOISE) {
  const result = isNoise(text);
  const status = result ? "✅ filtered" : "❌ MISSED";
  console.log(`| ${text} | ${category} | ${status} |`);
  if (result) {
    passed++;
  } else {
    failed++;
  }
}

// Test false-positive protection
console.log("\n### Should NOT be detected as noise\n");
console.log("| Text | Result |");
console.log("|------|--------|");

for (const text of SHOULD_NOT_BE_NOISE) {
  const result = isNoise(text);
  const status = result ? "❌ FALSE POSITIVE" : "✅ kept";
  console.log(`| ${text} | ${status} |`);
  if (!result) {
    passed++;
  } else {
    failed++;
  }
}

// Test filterNoise function with mixed content
console.log("\n### filterNoise() integration test\n");

const mixedItems = [
  { id: 1, text: "你好" },
  { id: 2, text: "MacBook Pro电池续航很好" },
  { id: 3, text: "我不记得了" },
  { id: 4, text: "SDDP算法收敛速度提升了30%" },
  { id: 5, text: "好的" },
  { id: 6, text: "谢谢你的帮助" },
];

const filtered = filterNoise(mixedItems, (item) => item.text);
const keptIds = filtered.map((item) => item.id);

console.log(`Input: ${mixedItems.length} items`);
console.log(`Output: ${filtered.length} items (kept IDs: ${keptIds.join(", ")})`);

assert.ok(keptIds.includes(2), "Should keep: MacBook Pro电池续航很好");
assert.ok(keptIds.includes(4), "Should keep: SDDP算法收敛速度提升了30%");
assert.ok(!keptIds.includes(1), "Should filter: 你好");
assert.ok(!keptIds.includes(3), "Should filter: 我不记得了");
assert.ok(!keptIds.includes(5), "Should filter: 好的");
assert.ok(!keptIds.includes(6), "Should filter: 谢谢你的帮助");
passed += 6;

// Summary
console.log(`\n### Summary\n`);
console.log(`Total: ${passed + failed} | Passed: ${passed} | Failed: ${failed}`);

if (failed > 0) {
  console.error(`\n❌ ${failed} test(s) failed`);
  process.exit(1);
} else {
  console.log(`\n✅ All tests passed`);
}
