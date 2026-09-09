'use client';

import {
  type BaseSyntheticEvent,
  type KeyboardEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';
import {
  BookOpenText,
  ChevronDown,
  CircleCheck,
  Send,
  Sparkles,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { Textarea } from '@/components/ui/textarea';
import { useChatTool } from '@/hooks/use-chat-tool';

type Message = {
  id: string;
  role: 'assistant' | 'user';
  text: string;
  sources?: string[];
};

type ChatResponse = {
  answer: string;
  sources: string[];
};

const API_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
).replace(/\/$/, '');

const EXAMPLES = [
  '¿Cuánto cuesta el plan Starter?',
  '¿Cómo protege NektiBot mis datos?',
  '¿Puede enviar correos por mí?',
];

const INITIAL_MESSAGE: Message = {
  id: 'welcome',
  role: 'assistant',
  text: 'Hola, soy NektiBot. Pregúntame sobre planes, integraciones, privacidad o cualquier dato del manual de producto.',
};

export function ChatApp() {
  const [messages, setMessages] = useState<Message[]>([INITIAL_MESSAGE]);
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const ask = useCallback(
    async (rawQuestion: string): Promise<ChatResponse> => {
      const nextQuestion = rawQuestion.trim();
      if (!nextQuestion) throw new Error('A question is required');
      if (isLoading) throw new Error('A request is already in progress');

      setQuestion('');
      setError(null);
      setIsLoading(true);
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: 'user', text: nextQuestion },
      ]);

      try {
        const response = await fetch(`${API_URL}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: nextQuestion }),
        });
        if (!response.ok)
          throw new Error(`Request failed with ${response.status}`);

        const data = (await response.json()) as ChatResponse;
        setMessages((current) => [
          ...current,
          {
            id: crypto.randomUUID(),
            role: 'assistant',
            text: data.answer,
            sources: data.sources,
          },
        ]);
        return data;
      } catch (requestError) {
        setError(
          'No he podido conectar con el asistente. Comprueba el backend e inténtalo de nuevo.',
        );
        setQuestion(nextQuestion);
        throw requestError;
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading],
  );

  useChatTool(ask);

  function handleSubmit(event: BaseSyntheticEvent) {
    event.preventDefault();
    void ask(question).catch(() => undefined);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void ask(question).catch(() => undefined);
    }
  }

  return (
    <main className="app-shell">
      <aside className="context-panel">
        <div>
          <div className="brand-mark" aria-hidden="true">
            <Sparkles size={21} strokeWidth={2.2} />
          </div>
          <p className="eyebrow">Asistente documental</p>
          <h1>NektiBot</h1>
          <p className="context-copy">
            Respuestas fundamentadas en el manual de producto, con las fuentes
            siempre a la vista.
          </p>
        </div>

        <section className="document-card" aria-label="Documento conectado">
          <BookOpenText size={19} aria-hidden="true" />
          <div>
            <span>Documento conectado</span>
            <strong>Manual de producto</strong>
          </div>
          <CircleCheck
            className="document-check"
            size={18}
            aria-label="Disponible"
          />
        </section>

        <div className="trust-note">
          <span className="status-dot" />
          <p>
            Si el manual no contiene la respuesta, NektiBot dirá{' '}
            <strong>“No lo sé”</strong>.
          </p>
        </div>
      </aside>

      <section className="chat-panel" aria-label="Chat con NektiBot">
        <header className="chat-header">
          <div>
            <p className="eyebrow">Manual de producto</p>
            <h2>Pregunta lo que necesites</h2>
          </div>
          <span className="online-label">
            <span /> Disponible
          </span>
        </header>

        <div className="messages" aria-live="polite">
          {messages.map((message) => (
            <article
              key={message.id}
              className={`message message-${message.role}`}
            >
              <span className="message-author">
                {message.role === 'assistant' ? 'NektiBot' : 'Tú'}
              </span>
              <div className="message-bubble">
                <p>{message.text}</p>
              </div>
              {message.sources?.length ? (
                <Sources sources={message.sources} />
              ) : null}
            </article>
          ))}

          {isLoading ? (
            <output className="thinking">
              <span />
              <span />
              <span />
              <span className="sr-only">Buscando en el documento</span>
            </output>
          ) : null}
          <div ref={endRef} />
        </div>

        <div className="composer-area">
          {messages.length === 1 ? (
            <div className="suggestions" aria-label="Preguntas de ejemplo">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  onClick={() => void ask(example).catch(() => undefined)}
                >
                  {example}
                </button>
              ))}
            </div>
          ) : null}
          {error ? (
            <p className="error-message" role="alert">
              {error}
            </p>
          ) : null}
          <form className="composer" onSubmit={handleSubmit}>
            <Textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Escribe una pregunta sobre NektiBot…"
              aria-label="Pregunta"
              maxLength={500}
              disabled={isLoading}
              rows={1}
            />
            <Button
              type="submit"
              size="icon-lg"
              disabled={!question.trim() || isLoading}
              aria-label="Enviar pregunta"
            >
              <Send size={18} />
            </Button>
          </form>
          <p className="composer-hint">
            Enter para enviar · Mayús + Enter para nueva línea
          </p>
        </div>
      </section>
    </main>
  );
}

function Sources({ sources }: { sources: string[] }) {
  return (
    <Collapsible className="sources">
      <CollapsibleTrigger className="sources-trigger">
        <BookOpenText size={15} />
        {sources.length === 1
          ? '1 fuente consultada'
          : `${sources.length} fuentes consultadas`}
        <ChevronDown className="sources-chevron" size={15} />
      </CollapsibleTrigger>
      <CollapsibleContent className="sources-content">
        {sources.map((source) => {
          const [title, ...body] = source.split('\n');
          return (
            <blockquote key={source}>
              <strong>{title}</strong>
              <p>{body.join(' ')}</p>
            </blockquote>
          );
        })}
      </CollapsibleContent>
    </Collapsible>
  );
}
