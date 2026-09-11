# NektiBot — asistente RAG

Chat documental que responde únicamente desde `data/sample.md`, cita los fragmentos utilizados y
contesta `No lo sé` cuando no encuentra evidencia suficiente.

> Enlace público: se añadirá después del despliegue.

## Arquitectura

```text
Pregunta ─┬─ BM25 (términos exactos) ─────┐
          └─ embeddings + coseno ─────────┴─ ranking híbrido
                                                   │
                                      ¿hay evidencia suficiente?
                                         ├─ no → "No lo sé"
                                         └─ sí → LLM → respuesta + fuentes
```

El documento se divide por secciones Markdown. BM25 y los embeddings del corpus se calculan una vez,
de forma diferida, y permanecen en memoria. Cada pregunta requiere un embedding y solo llama al
modelo de chat cuando supera el umbral de recuperación.

## Ejecución

Requisitos: Python 3.10+, [uv](https://docs.astral.sh/uv/) y Node.js 22+.

### Local

Backend:

```bash
uv sync
cp api/.env.example .env
# Añade OPENAI_API_KEY a .env
uv run uvicorn api.app:app --reload --port 8000
```

Frontend, en otra terminal:

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Abre `http://localhost:3000`. La API expone su healthcheck en
`http://localhost:8000/api/health` y OpenAPI en `http://localhost:8000/docs`.

### Docker

API y frontend son targets independientes del mismo Dockerfile y Compose los inicia por separado:

```bash
cp api/.env.example api/.env
# Añade OPENAI_API_KEY a api/.env
docker compose up --build
```

Servicios: frontend en `http://localhost:3000` y API en `http://localhost:8000`.

## API

`POST /api/chat`

```json
{ "question": "¿Cuánto cuesta Starter?" }
```

```json
{
  "answer": "El plan Starter cuesta 49 € al mes.",
  "sources": ["Planes y precios\n- **Starter**: 49 €/mes. ..."]
}
```

Sin evidencia devuelve `{"answer": "No lo sé", "sources": []}`.

## Calidad

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest

cd frontend
npm run lint
npx oxfmt --check .
npm run build
```

La mini-evaluación contiene cuatro preguntas respondibles y una fuera del documento; los cinco casos
se han validado con llamadas reales a OpenAI:

```bash
uv run python evals/run.py
# Para un despliegue: EVAL_API_URL=https://api.example.com uv run python evals/run.py
```

Opcionalmente, Ragas evalúa fidelidad, relevancia y calidad del contexto sobre los cuatro casos
respondibles. Ejecuta el pipeline directamente y realiza llamadas adicionales a OpenAI:

```bash
uv run --with 'ragas>=0.4,<0.5' --with 'langchain-community>=0.3,<0.4' \
  python evals/ragas_eval.py
```

## Decisiones técnicas

- **Retrieval híbrido en memoria:** BM25 cubre nombres, precios y términos exactos; los embeddings,
  paráfrasis. Para siete secciones, una base vectorial añadiría infraestructura sin beneficio real.
- **Dos barreras contra alucinaciones:** un umbral evita llamar al LLM sin evidencia y el prompt
  prohíbe utilizar conocimiento externo.
- **Fuentes literales:** la API devuelve el contexto entregado al modelo y oculta las fuentes cuando
  la respuesta es `No lo sé`.
- **OpenAI aislado:** retrieval no depende del SDK, usa embeddings deterministas en tests y crea el
  cliente real solo cuando hace falta.
- **Estado local:** el historial vive en el frontend; no hay persistencia ni cuentas de usuario.

## Despliegue y configuración

- Contenedores: `Dockerfile` multi-stage con targets `api` y `frontend`.
- Backend: Blueprint `render.yaml` para Render.
- Frontend: configuración de Sites/Cloudflare.
- Desarrollo conjunto: `compose.yaml`.

| Variable | Servicio | Uso |
|---|---|---|
| `OPENAI_API_KEY` | Backend | Credencial privada; nunca se expone al frontend. |
| `OPENAI_MODEL` | Backend | Modelo de chat. |
| `OPENAI_EMBEDDING_MODEL` | Backend | Modelo de embeddings. |
| `CORS_ORIGINS` | Backend | URLs frontend permitidas, separadas por comas. |
| `NEXT_PUBLIC_API_URL` | Frontend | URL pública de la API. |
| `NEXT_PUBLIC_SITE_URL` | Frontend | URL usada en metadatos sociales. |

## Próximos pasos

- Calibrar pesos y umbrales con un conjunto de evaluación mayor.
- Medir latencia y scores de recuperación sin registrar información sensible.
- Añadir streaming cuando el tamaño de las respuestas lo justifique.
- Para colecciones grandes o actualizables, usar Qdrant y una base persistente para metadatos.
