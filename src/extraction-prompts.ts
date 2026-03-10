/**
 * Prompt templates for intelligent memory extraction.
 * Three mandatory prompts:
 * - buildExtractionPrompt: 6-category L0/L1/L2 extraction with few-shot
 * - buildDedupPrompt: CREATE/MERGE/SKIP dedup decision
 * - buildMergePrompt: Memory merge with three-level structure
 */

export function buildExtractionPrompt(
  conversationText: string,
  user: string,
): string {
  return `Analyze the following session context and extract memories worth long-term preservation.

User: ${user}

Target Output Language: auto (detect from recent messages)

## Recent Conversation
${conversationText}

# Memory Extraction Criteria

## What is worth remembering?
- Personalized information: Information specific to this user, not general domain knowledge
- Long-term validity: Information that will still be useful in future sessions
- Specific and clear: Has concrete details, not vague generalizations

## What is NOT worth remembering?
- General knowledge that anyone would know
- Temporary information: One-time questions or conversations
- Vague information: "User has questions about a feature" (no specific details)
- Tool output, error logs, or boilerplate

# Memory Classification

## Core Decision Logic

| Question | Answer | Category |
|----------|--------|----------|
| Who is the user? | Identity, attributes | profile |
| What does the user prefer? | Preferences, habits | preferences |
| What is this thing? | Person, project, organization | entities |
| What happened? | Decision, milestone | events |
| How was it solved? | Problem + solution | cases |
| What is the process? | Reusable steps | patterns |

## Precise Definition

**profile** - User identity (static attributes). Test: "User is..."
**preferences** - User preferences (tendencies). Test: "User prefers/likes..."
**entities** - Continuously existing nouns. Test: "XXX's state is..."
**events** - Things that happened. Test: "XXX did/completed..."
**cases** - Problem + solution pairs. Test: Contains "problem -> solution"
**patterns** - Reusable processes. Test: Can be used in "similar situations"

## Common Confusion
- "Plan to do X" -> events (action, not entity)
- "Project X status: Y" -> entities (describes entity)
- "User prefers X" -> preferences (not profile)
- "Encountered problem A, used solution B" -> cases (not events)
- "General process for handling certain problems" -> patterns (not cases)

# Three-Level Structure

Each memory contains three levels:

**abstract (L0)**: One-liner index
- Merge types (preferences/entities/profile/patterns): \`[Merge key]: [Description]\`
- Independent types (events/cases): Specific description

**overview (L1)**: Structured Markdown summary with category-specific headings

**content (L2)**: Full narrative with background and details

# Few-shot Examples

## profile
\`\`\`json
{
  "category": "profile",
  "abstract": "User basic info: AI development engineer, 3 years LLM experience",
  "overview": "## Background\\n- Occupation: AI development engineer\\n- Experience: 3 years LLM development\\n- Tech stack: Python, LangChain",
  "content": "User is an AI development engineer with 3 years of LLM application development experience."
}
\`\`\`

## preferences
\`\`\`json
{
  "category": "preferences",
  "abstract": "Python code style: No type hints, concise and direct",
  "overview": "## Preference Domain\\n- Language: Python\\n- Topic: Code style\\n\\n## Details\\n- No type hints\\n- Concise function comments\\n- Direct implementation",
  "content": "User prefers Python code without type hints, with concise function comments."
}
\`\`\`

## cases
\`\`\`json
{
  "category": "cases",
  "abstract": "LanceDB BigInt error -> Use Number() coercion before arithmetic",
  "overview": "## Problem\\nLanceDB 0.26+ returns BigInt for numeric columns\\n\\n## Solution\\nCoerce values with Number(...) before arithmetic",
  "content": "When LanceDB returns BigInt values, wrap them with Number() before doing arithmetic operations."
}
\`\`\`

# Output Format

Return JSON:
{
  "memories": [
    {
      "category": "profile|preferences|entities|events|cases|patterns",
      "abstract": "One-line index",
      "overview": "Structured Markdown summary",
      "content": "Full narrative"
    }
  ]
}

Notes:
- Output language should match the dominant language in the conversation
- Only extract truly valuable personalized information
- If nothing worth recording, return {"memories": []}
- Maximum 5 memories per extraction
- Preferences should be aggregated by topic`;
}

export function buildDedupPrompt(
  candidateAbstract: string,
  candidateOverview: string,
  candidateContent: string,
  existingMemories: string,
): string {
  return `Determine how to handle this candidate memory.

**Candidate Memory**:
Abstract: ${candidateAbstract}
Overview: ${candidateOverview}
Content: ${candidateContent}

**Existing Similar Memories**:
${existingMemories}

Please decide one of the following actions:

- **SKIP**: Candidate is a near-exact duplicate with no new information.
- **CREATE**: Completely new information, no overlap with existing memories.
- **SUPPORT**: Same claim confirmed again. The existing memory stays unchanged; we only record that it was re-observed.
- **MERGE**: Candidate should be merged with an existing memory (legacy alias for REFINE — rewrites text).
- **REFINE**: Candidate adds precision or detail to an existing memory (e.g. "likes tea" → "likes oolong tea").
- **CONTEXTUALIZE**: Candidate adds situational/temporal context to an existing memory (e.g. "prefers tea at night"), but the original claim remains valid in its own context.
- **CONTRADICT**: Candidate conflicts with an existing memory. Both should be kept; the conflict is logged but not auto-resolved.

IMPORTANT:
- "events" and "cases" categories are independent records — only use SKIP or CREATE for them.
- Default to SUPPORT (not SKIP) when the same preference/fact is re-stated, so evidence accumulates.
- Use CONTEXTUALIZE (not MERGE) when info adds time/place/condition constraints without invalidating the original.
- Use CONTRADICT only when the candidate clearly negates the existing claim.

Return JSON format:
{
  "decision": "skip|create|support|merge|refine|contextualize|contradict",
  "match_index": 1,
  "reason": "Decision reason",
  "context_label": "general"
}

Rules for context_label:
- Required when decision is support, contextualize, or contradict.
- Must be one of: general, morning, evening, night, weekday, weekend, work, leisure, travel, recent, historical.
- Use "general" when no specific situational/temporal context applies.
- For CONTEXTUALIZE decisions, the label should describe the NEW context being added (e.g. "evening" for "prefers tea at night").
- For CONTRADICT decisions, the label should describe the context in which the contradiction applies.
- For SUPPORT decisions, the label should match the context of the reconfirming evidence.

If decision involves an existing memory, set "match_index" to the number of the existing memory (1-based).`;
}

export function buildMergePrompt(
  existingAbstract: string,
  existingOverview: string,
  existingContent: string,
  newAbstract: string,
  newOverview: string,
  newContent: string,
  category: string,
  relationType?: string,
): string {
  const relationGuidance = relationType === "contextualize"
    ? `\n\nRelation type: CONTEXTUALIZE\nThe new information adds situational context (time/place/condition) to the existing memory. Preserve the original claim as-is and integrate the contextual nuance. Both the general claim and the specific context should coexist in the merged result.`
    : relationType === "refine"
      ? `\n\nRelation type: REFINE\nThe new information adds precision or detail to the existing claim. Update the claim to be more specific while preserving all accurate existing information.`
      : "";

  return `Merge the following memory into a single coherent record with all three levels.

**Category**: ${category}${relationGuidance}

**Existing Memory:**
Abstract: ${existingAbstract}
Overview:
${existingOverview}
Content:
${existingContent}

**New Information:**
Abstract: ${newAbstract}
Overview:
${newOverview}
Content:
${newContent}

Requirements:
- Remove duplicate information
- Keep the most up-to-date details
- Maintain a coherent narrative
- Keep code identifiers / URIs / model names unchanged when they are proper nouns

Return JSON:
{
  "abstract": "Merged one-line abstract",
  "overview": "Merged structured Markdown overview",
  "content": "Merged full content"
}`;
}
