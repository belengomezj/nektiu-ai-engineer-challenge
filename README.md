# NektiBot — asistente RAG

**Aplicación pública:** https://nektibot-web.onrender.com/

## Instalación

### Requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Node.js 22+

### Ejecución local

Backend:

```bash
uv sync
cp api/.env.example .env
# Añade OPENAI_API_KEY a .env
uv run uvicorn api.app:app --reload --port 8000
```

- API: `http://localhost:8000`
- Health check: `http://localhost:8000/api/health`
- OpenAPI: `http://localhost:8000/docs`

Frontend, en otra terminal:

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Abre `http://localhost:3000`.

### Docker

También se puede levantar todo el proyecto con Docker Compose:

```bash
cp api/.env.example api/.env
# Añade OPENAI_API_KEY a api/.env
docker compose up --build
```

- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`

### Despliegue en Render

[`render.yaml`](render.yaml) crea dos servicios públicos e independientes:

| Servicio | Runtime | URL |
|---|---|---|
| `nektibot-web` | Node | https://nektibot-web.onrender.com/ |
| `nektibot-api` | Docker/FastAPI | https://nektibot-api.onrender.com/ |

1. Crea un Blueprint de Render desde este repositorio y la rama `main`.
2. Introduce `OPENAI_API_KEY` como secreto y pulsa **Deploy Blueprint**.
3. Render conecta automáticamente la URL de la API con el frontend y configura CORS con el origen
   real del frontend.

Los dos servicios usan el plan gratuito. Es suficiente para esta prueba, pero los servicios web se
duermen tras un periodo sin tráfico, pueden tardar alrededor de un minuto en arrancar y comparten
las horas gratuitas del workspace. El uso de la API de OpenAI se factura por separado.

### Comprobaciones

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest

cd frontend
npm run lint
npx oxfmt --check .
npm run build
```

## Decisiones

### Repositorio

| Decisión | Motivo |
|---|---|
| `uv`, `pyproject.toml` y `uv.lock` | Instalación reproducible y una única definición de dependencias. |
| Backend y frontend separados | Cada servicio puede desarrollarse y desplegarse de forma independiente. |
| Un Dockerfile multi-stage | Evita duplicar configuración y genera imágenes específicas para cada servicio. |
| Dos servicios en Render | Mantiene esa separación también en producción y evita exponer la clave de OpenAI en el navegador. |
| CI con Ruff, formato y pytest | Cubre las comprobaciones esenciales. |
| Ragas como herramienta opcional | La evaluación usa red y modelos, por lo que no forma parte de la aplicación ni del CI. |

No se utiliza un framework de agentes: descarto una capa de orquestación por el tamaño de la tarea.

### Arquitectura

| Componente | Implementación |
|---|---|
| Interfaz | Next/Vinext, historial local y diseño minimalista. |
| API | FastAPI con endpoints JSON, streaming SSE y validación Pydantic. |
| Estado | El backend no almacena usuarios ni conversaciones. El navegador conserva hasta 40 mensajes y envía los 6 últimos. |
| Inicialización | El retriever se construye al recibir el primer health check o la primera consulta y queda en memoria. |
| Disponibilidad | `/api/health` devuelve `200` si el índice está preparado y `503` si no puede construirse. |
| Operación | `Server-Timing` y logs de latencia para HTTP, retrieval y generación, sin registrar preguntas. |
| Contenedores | API y frontend se ejecutan como servicios independientes. |

La API expone `POST /api/chat` y `POST /api/chat/stream`. Ambos aceptan una pregunta y un historial
breve; la variante streaming entrega eventos SSE `token`, `done` y `error`.

### RAG

Se utilizan los modelos sugeridos en `api/.env.example`.

| Etapa | Modelo | Uso |
|---|---|---|
| Embeddings del documento | `text-embedding-3-small` | Una llamada por lotes al construir el índice. |
| Embedding de la pregunta | `text-embedding-3-small` | Una llamada en cada consulta. |
| Generación de la respuesta | `gpt-4.1-mini` | Solo cuando el retrieval encuentra evidencia. |
| Evaluación Ragas | `gpt-4.1-mini` y `text-embedding-3-small` | Ejecución manual, fuera de la aplicación. |

**Flujo:** documento → secciones Markdown → búsqueda BM25 y semántica → ranking ponderado →
filtro de evidencia → modelo → respuesta con fuentes.

- Cada encabezado `##` forma un fragmento con significado y un título útil para citar.
- BM25 recupera coincidencias literales; los embeddings cubren preguntas formuladas con otras
  palabras.
- Las dos señales se normalizan y combinan mediante pesos calibrados.
- Si ninguna señal alcanza sus umbrales, la API devuelve `No lo sé` sin llamar al modelo de chat.
- El prompt usa el historial breve para interpretar la conversación, limita los hechos al contexto
  recuperado y hace que el modelo identifique los fragmentos realmente usados.
- Las variantes `No lo sé` y `No lo sé.` se normalizan y nunca muestran fuentes.
- Los embeddings del documento se calculan una vez; cada consulta solo vectoriza la pregunta.

La evaluación funcional contiene 25 escenarios propios: 20 respondibles y 5 fuera del documento.
La última ejecución real obtuvo `23/25`; los dos fallos fueron abstenciones conservadoras.

La calibración explora 512 combinaciones reutilizando un único lote de embeddings:

La configuración base se eligió al azar como punto de partida, sin apoyarse en resultados de
evaluación:

| Parámetro | Base | Configuración elegida |
|---|---:|---:|
| Peso BM25 | `0.45` | **`0.15`** |
| Peso semántico | `0.55` | **`0.85`** |
| Umbral BM25 | `0.8` | **`1.1`** |
| Umbral de similitud semántica | `0.35` | **`0.40`** |

| Métrica de retrieval | Base | Configuración elegida |
|---|---:|---:|
| Puntuación equilibrada | 0,73 | **0,85** |
| Recall de la sección esperada | 0,85 | **0,90** |
| Abstención correcta | 0,60 | **0,80** |

El resultado completo está en `evals/calibration_result.json`.

Ragas se ejecuta sobre los nueve casos respondibles del subconjunto de validación:

| Métrica Ragas | Base | Actual |
|---|---:|---:|
| Fidelidad | 0,75 | **0,84** |
| Relevancia de la respuesta | 0,40 | **0,46** |
| Precisión del contexto | 0,78 | **0,83** |
| Recall del contexto | 0,74 | **0,85** |

```bash
uv run python evals/run.py
uv run python evals/calibrate.py
uv run --with 'ragas>=0.4,<0.5' --with 'langchain-community>=0.3,<0.4' \
  python evals/ragas_eval.py
```

## Mejoras

Estas mejoras no son necesarias para el alcance actual, pero tendrían sentido si el proyecto creciera:

| Escenario posible | Mejora |
| --- | --- |
| Hay muchos documentos o cambian con frecuencia | Persistiría documentos, metadatos y vectores para no depender de un índice construido en memoria. |
| Se necesitan cuentas de usuario | Añadiría autenticación y persistencia de conversaciones y permisos; actualmente el historial solo vive en el navegador. |
| La API recibe tráfico público significativo | Añadiría rate limiting para evitar abuso y controlar el coste de las llamadas al modelo. |
| Aumentan mucho el tráfico o los costes | Además de reutilizar los embeddings del documento, estudiaría cachear consultas repetidas y separar el procesamiento de documentos. |
| Los logs actuales dejan de ser suficientes para diagnosticar problemas | Centralizaría logs y añadiría trazas para seguir una petición completa entre servicios. |
| Quiero tener más confianza en las métricas del RAG | Separaría claramente los casos usados para calibrar de los usados para validar y ampliaría el conjunto de evaluación con preguntas externas. |
| La búsqueda falla con nombres concretos de productos, organizaciones, lugares o normas | Añadiría NER para detectar esas entidades y utilizarlas como metadatos o filtros de búsqueda. |
| Las respuestas requieren relacionar información repartida entre varios documentos | Plantearía una ontología y un grafo para representar entidades y relaciones de forma explícita. |
| El proyecto crece y el README deja de ser suficiente | Ampliaría la documentación con MkDocs, separando instalación, arquitectura, API y decisiones técnicas en páginas específicas. |
