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
  RotateCcw,
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
import {
  type ChatHistory,
  type ChatResponse,
  streamChat,
} from '@/lib/chat-stream';

type Message = {
  id: string;
  role: 'assistant' | 'user';
  text: string;
  sources?: string[];
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

const STORAGE_KEY = 'nektibot-history';

function isMessage(value: unknown): value is Message {
  if (!value || typeof value !== 'object') return false;
  const message = value as Partial<Message>;
  return (
    typeof message.id === 'string' &&
    typeof message.text === 'string' &&
    (message.role === 'assistant' || message.role === 'user')
  );
}

export function ChatApp() {
  const [messages, setMessages] = useState<Message[]>([INITIAL_MESSAGE]);
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;
    queueMicrotask(() => {
      if (!active) return;
      try {
        const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]');
        if (Array.isArray(stored)) {
          setMessages([
            INITIAL_MESSAGE,
            ...stored.filter(isMessage).slice(-40),
          ]);
        }
      } catch {
        localStorage.removeItem(STORAGE_KEY);
      } finally {
        setHistoryLoaded(true);
      }
    });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (historyLoaded && !isLoading) {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(messages.slice(1).slice(-40)),
      );
    }
  }, [historyLoaded, isLoading, messages]);

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
      const history: ChatHistory[] = messages
        .slice(1)
        .slice(-6)
        .map((message) => ({ role: message.role, content: message.text }));
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: 'user', text: nextQuestion },
      ]);

      try {
        const responseId = crypto.randomUUID();
        const data = await streamChat(API_URL, nextQuestion, history, (text) =>
          setMessages((current) => {
            const existing = current.some(
              (message) => message.id === responseId,
            );
            if (!existing) {
              return [...current, { id: responseId, role: 'assistant', text }];
            }
            return current.map((message) =>
              message.id === responseId
                ? { ...message, text: message.text + text }
                : message,
            );
          }),
        );
        setMessages((current) => {
          const complete = {
            id: responseId,
            role: 'assistant' as const,
            text: data.answer,
            sources: data.sources,
          };
          return current.some((message) => message.id === responseId)
            ? current.map((message) =>
                message.id === responseId ? complete : message,
              )
            : [...current, complete];
        });
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
    [isLoading, messages],
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

  function clearHistory() {
    localStorage.removeItem(STORAGE_KEY);
    setMessages([INITIAL_MESSAGE]);
    setError(null);
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
          <div className="header-actions">
            {messages.length > 1 ? (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                disabled={isLoading}
                onClick={clearHistory}
              >
                <RotateCcw size={14} /> Limpiar
              </Button>
            ) : null}
            <span className="online-label">
              <span /> Disponible
            </span>
          </div>
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
