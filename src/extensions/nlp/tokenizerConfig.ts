/**
 * The subset of a Hugging Face `tokenizer_config.json` needed to drive chat:
 * the jinja chat template plus the special tokens it references.
 */
export type TokenizerChatConfig = {
  readonly chatTemplate: string;
  readonly bosToken?: string;
  readonly eosToken?: string;
};

// A special token may be a bare string or an `AddedToken` dict (`{ content }`).
function resolveToken(token: unknown): string | undefined {
  if (typeof token === 'string') return token;
  if (token && typeof token === 'object' && typeof (token as any).content === 'string') {
    return (token as any).content;
  }
  return undefined;
}

/**
 * Extracts the chat template and special tokens from a parsed
 * `tokenizer_config.json`. Throws if no string `chat_template` is present.
 */
export function parseTokenizerConfig(config: any): TokenizerChatConfig {
  let chatTemplate = config?.chat_template;

  // Some models ship multiple named templates as `[{ name, template }]`.
  if (Array.isArray(chatTemplate)) {
    const entry = chatTemplate.find((t) => t?.name === 'default') ?? chatTemplate[0];
    chatTemplate = entry?.template;
  }

  if (typeof chatTemplate !== 'string') {
    throw new Error('tokenizer_config.json does not contain a string `chat_template`');
  }

  return {
    chatTemplate,
    bosToken: resolveToken(config?.bos_token),
    eosToken: resolveToken(config?.eos_token),
  };
}
