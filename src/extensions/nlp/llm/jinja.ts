import { Template } from '@huggingface/jinja';

import type { ChatFormatter } from '../tasks/llmChat';

export type JinjaFormatterOptions = {
  readonly bosToken?: string;
  readonly extraContext?: Record<string, unknown>;
};

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
      messages: isGenerationPrompt ? [] : [{ role: message.role, content: message.content }],
      ...extraContext,
    });
  };
}
