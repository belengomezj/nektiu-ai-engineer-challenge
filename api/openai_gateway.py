"""Small OpenAI boundary used by the application."""

import os
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from api.messages import UNKNOWN_ANSWER

load_dotenv()
load_dotenv(Path(__file__).with_name(".env"))

CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


@lru_cache
def get_client() -> OpenAI:
    """Create the client only when the API needs it."""
    return OpenAI()


def create_embeddings(texts: list[str]) -> list[list[float]]:
    response = get_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _messages(question: str, context: list[str], history: list[dict]) -> list[dict]:
    formatted_context = "\n\n---\n\n".join(
        f"[{index}] {text}" for index, text in enumerate(context, start=1)
    )
    return [
        {
            "role": "system",
            "content": (
                "Contesta la pregunta usando exclusivamente hechos presentes en el contexto. "
                "Comprueba todas las condiciones cuando la consulta reúna varias y conserva "
                "los límites o matices del documento. Si falta la información necesaria, "
                f"contesta exactamente «{UNKNOWN_ANSWER}». No completes huecos con conocimiento "
                "externo. Responde de forma directa, breve y en el idioma de la consulta. "
                "El historial sirve para entender referencias, pero no aporta hechos nuevos. "
                "Acaba con una línea FUENTES: [1, 2] que enumere únicamente los fragmentos "
                "que sostienen la respuesta; usa FUENTES: [] al contestar No lo sé."
            ),
        },
        *history,
        {
            "role": "user",
            "content": f"Contexto:\n{formatted_context}\n\nPregunta: {question}",
        },
    ]


def create_answer(question: str, context: list[str], history: list[dict] | None = None) -> str:
    response = get_client().chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=_messages(question, context, history or []),
    )
    return response.choices[0].message.content or UNKNOWN_ANSWER


def create_answer_stream(
    question: str, context: list[str], history: list[dict] | None = None
) -> Iterator[str]:
    """Yield answer fragments as OpenAI produces them."""
    stream = get_client().chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=_messages(question, context, history or []),
        stream=True,
    )
    for chunk in stream:
        if text := chunk.choices[0].delta.content:
            yield text
