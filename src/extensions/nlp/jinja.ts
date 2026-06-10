import { Template } from '@huggingface/jinja';

import type { ChatFormatter } from './tasks/llm';

export type JinjaFormatterOptions = {
  /** The model's BOS token, emitted once at the start of the conversation. */
  readonly bosToken?: string;
  /** Extra variables exposed to the template (e.g. `tools`, custom flags). */
  readonly extraContext?: Record<string, unknown>;
};

/**
 * Builds a {@link ChatFormatter} from a Hugging Face jinja `chat_template`
 * (as found in a model's `tokenizer_config.json`).
 *
 * The chat session formats one message at a time and relies on incremental
 * prefill, whereas chat templates render a whole conversation. We bridge the
 * two by rendering a single-message conversation per call and gating the BOS
 * token on `isFirst`, so it is emitted exactly once. An empty assistant
 * message is treated as the generation prompt and rendered via
 * `add_generation_prompt`.
 */
export function createJinjaChatFormatter(
  chatTemplate: string,
  options: JinjaFormatterOptions = {},
): ChatFormatter {
  const { bosToken = '', extraContext } = options;
  const template = new Template(chatTemplate);

  return (message, { isFirst }) => {
    const isGenerationPrompt = message.role === 'assistant' && message.content === '';
    return template.render({
      // Only the first prefill of a conversation should carry the BOS token;
      // later turns append to the model's existing KV cache.
      bos_token: isFirst ? bosToken : '',
      add_generation_prompt: isGenerationPrompt,
      messages: isGenerationPrompt
        ? []
        : [{ role: message.role, content: message.content }],
      ...extraContext,
    });
  };
}
