export type ChatHistory = {
  role: 'assistant' | 'user';
  content: string;
};

export type ChatResponse = {
  answer: string;
  sources: string[];
};

type StreamEvent = { event: string; data: Record<string, unknown> };

function readEvent(block: string): StreamEvent {
  const lines = block.split('\n');
  const event = lines.find((line) => line.startsWith('event: '))?.slice(7);
  const data = lines.find((line) => line.startsWith('data: '))?.slice(6);
  if (!event || !data) throw new Error('Invalid stream event');
  return { event, data: JSON.parse(data) as Record<string, unknown> };
}

export async function streamChat(
  apiUrl: string,
  question: string,
  history: ChatHistory[],
  onToken: (text: string) => void,
): Promise<ChatResponse> {
  const response = await fetch(`${apiUrl}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, history }),
  });
  if (!response.ok || !response.body) {
    throw new Error(`Request failed with ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result: ChatResponse | null = null;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split('\n\n');
    buffer = blocks.pop() ?? '';

    for (const block of blocks) {
      if (!block.trim()) continue;
      const message = readEvent(block);
      if (message.event === 'token' && typeof message.data.text === 'string') {
        onToken(message.data.text);
      }
      if (message.event === 'error')
        throw new Error('The assistant could not respond');
      if (
        message.event === 'done' &&
        typeof message.data.answer === 'string' &&
        Array.isArray(message.data.sources) &&
        message.data.sources.every((source) => typeof source === 'string')
      ) {
        result = {
          answer: message.data.answer,
          sources: message.data.sources,
        };
      }
    }
    if (done) break;
  }

  if (!result) throw new Error('The stream ended without a response');
  return result;
}
