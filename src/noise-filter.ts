/**
 * Noise Filter
 * Filters out low-quality memories (meta-questions, agent denials, session boilerplate)
 * Inspired by openclaw-plugin-continuity's noise filtering approach.
 *
 * Supports both English and Chinese patterns.
 */

// Agent-side denial patterns
const DENIAL_PATTERNS = [
  /i don'?t have (any )?(information|data|memory|record)/i,
  /i'?m not sure about/i,
  /i don'?t recall/i,
  /i don'?t remember/i,
  /it looks like i don'?t/i,
  /i wasn'?t able to find/i,
  /no (relevant )?memories found/i,
  /i don'?t have access to/i,
  // Chinese denial patterns (agent-side)
  /我没有(任何)?(相关)?(信息|数据|记忆|记录)/,
  /我不(太)?确定/,
  /我不记得/,
  /我想不起来/,
  /我没(有)?找到/,
  /找不到(相关)?记忆/,
  /没有(相关)?记忆/,
  /我无法(访问|获取)/,
];

// User-side meta-question patterns (about memory itself, not content)
const META_QUESTION_PATTERNS = [
  /\bdo you (remember|recall|know about)\b/i,
  /\bcan you (remember|recall)\b/i,
  /\bdid i (tell|mention|say|share)\b/i,
  /\bhave i (told|mentioned|said)\b/i,
  /\bwhat did i (tell|say|mention)\b/i,
  // Chinese meta-question patterns (user-side)
  /你(还)?记得吗/,
  /你(还)?记不记得/,
  /你知道我(说过|提过|告诉|提到).*吗/,
  /我(有没有|是不是)(说过|提过|告诉|提到)/,
  /我之前(说过|提过|提到|告诉)/,
  /我(跟你)?说过.*吗/,
];

// Session boilerplate — safe patterns that won't cause false positives
const BOILERPLATE_PATTERNS = [
  /^(hi|hello|hey|good morning|good evening|greetings)/i,
  /^fresh session/i,
  /^new session/i,
  /^HEARTBEAT/i,
  // Chinese boilerplate — anchored to start AND end to prevent false positives
  // e.g. "你好厉害" should NOT match, but "你好" or "你好！" should
  /^你好[!！\s,.，。]?$/,
  /^(早上好|早安|午安|晚上好|晚安)[!！\s,.，。]?$/,
  /^(嗨|哈[喽啰]|哈[喽啰]呀)[!！\s,.，。]?$/,
  /^(嗯|哦|噢|呃)[!！\s,.，。]?$/,
  /^新(会话|对话|聊天)/,
];

/**
 * Short boilerplate patterns — these are prefix patterns that are only
 * treated as noise when the total text length is short (≤ BOILERPLATE_MAX_LENGTH).
 *
 * This prevents false positives on longer messages that happen to start
 * with acknowledgment words:
 *   "好的" → noise (2 chars)
 *   "好的方案是使用Redis做缓存层" → NOT noise (meaningful content after)
 *   "谢谢你的帮助" → noise (7 chars, still just thanks)
 *   "谢谢分享，我觉得这个思路很好" → NOT noise (substantive content)
 */
const SHORT_BOILERPLATE_PATTERNS = [
  /^(好的|好吧|行|可以|没问题|OK|ok|收到|明白|了解|知道了)/i,
  /^(谢谢|感谢|多谢|谢啦|3Q|thx)/i,
];

/** Max character length for short boilerplate to be considered noise */
const BOILERPLATE_MAX_LENGTH = 10;

export interface NoiseFilterOptions {
  /** Filter agent denial responses (default: true) */
  filterDenials?: boolean;
  /** Filter meta-questions about memory (default: true) */
  filterMetaQuestions?: boolean;
  /** Filter session boilerplate (default: true) */
  filterBoilerplate?: boolean;
}

const DEFAULT_OPTIONS: Required<NoiseFilterOptions> = {
  filterDenials: true,
  filterMetaQuestions: true,
  filterBoilerplate: true,
};

/**
 * Check if a memory text is noise that should be filtered out.
 * Returns true if the text is noise.
 */
export function isNoise(
  text: string,
  options: NoiseFilterOptions = {},
): boolean {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const trimmed = text.trim();

  if (trimmed.length < 5) return true;

  if (opts.filterDenials && DENIAL_PATTERNS.some((p) => p.test(trimmed)))
    return true;
  if (
    opts.filterMetaQuestions &&
    META_QUESTION_PATTERNS.some((p) => p.test(trimmed))
  )
    return true;
  if (opts.filterBoilerplate) {
    if (BOILERPLATE_PATTERNS.some((p) => p.test(trimmed))) return true;
    // Short boilerplate: only filter when text is short enough to be pure filler
    if (
      trimmed.length <= BOILERPLATE_MAX_LENGTH &&
      SHORT_BOILERPLATE_PATTERNS.some((p) => p.test(trimmed))
    )
      return true;
  }

  return false;
}

/**
 * Filter an array of items, removing noise entries.
 */
export function filterNoise<T>(
  items: T[],
  getText: (item: T) => string,
  options?: NoiseFilterOptions,
): T[] {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  return items.filter((item) => !isNoise(getText(item), opts));
}
