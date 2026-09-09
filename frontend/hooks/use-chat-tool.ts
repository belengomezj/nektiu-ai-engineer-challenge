'use client';

import { useEffect, useRef } from 'react';

type ToolResult = { answer: string; sources: string[] };
type AskQuestion = (question: string) => Promise<ToolResult>;

export function useChatTool(askQuestion: AskQuestion) {
  const actionRef = useRef(askQuestion);

  useEffect(() => {
    actionRef.current = askQuestion;
  }, [askQuestion]);

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;

    const lifecycle = new AbortController();
    const registration = context.registerTool(
      {
        name: 'ask_nektibot',
        title: 'Preguntar a NektiBot',
        description:
          'Consulta el manual de producto y muestra la respuesta y sus fuentes en el chat.',
        inputSchema: {
          type: 'object',
          properties: {
            question: { type: 'string', minLength: 1, maxLength: 500 },
          },
          required: ['question'],
          additionalProperties: false,
        },
        annotations: { readOnlyHint: false, untrustedContentHint: false },
        async execute(input) {
          return actionRef.current(readQuestion(input));
        },
      },
      { signal: lifecycle.signal },
    );

    void Promise.resolve(registration).catch((error) => {
      console.error('Unable to register the NektiBot browser tool', error);
    });
    return () => lifecycle.abort();
  }, []);
}

function readQuestion(input: unknown): string {
  if (!input || typeof input !== 'object' || !('question' in input)) {
    throw new Error('A question is required');
  }
  const question = (input as { question: unknown }).question;
  if (
    typeof question !== 'string' ||
    !question.trim() ||
    question.length > 500
  ) {
    throw new Error('Question must contain between 1 and 500 characters');
  }
  return question.trim();
}
