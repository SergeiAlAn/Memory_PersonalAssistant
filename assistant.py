"""
Умный персональный помощник на Haystack Agent + PineconeDocumentStore.

В Pinecone хранятся только извлечённые факты о пользователе (не вопросы).
Перед ответом факты подтягиваются по косинусному сходству.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from haystack import Document
from haystack.components.agents import Agent
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from tools import build_tools

load_dotenv(Path(__file__).resolve().parent / ".env")

# Порог: похожий факт → обновляем/пропускаем, а не плодим дубликаты
FACT_DEDUP_THRESHOLD = 0.74

EXTRACT_PROMPT = """Извлеки из сообщения пользователя факты о НЁМ для долговременной памяти.

Что НУЖНО сохранять (это факты):
- интересы и увлечения («меня интересует Плутон», «люблю прогулки»);
- желания и цели («хочу на Марс», «хочу выучить Python»);
- имя, город, работа, предпочтения, важные события;
- формулируй от третьего лица: «Интересуется Плутоном», «Хочет на Марс».

Что НЕ сохранять:
- чистые вопросы без новых фактов («Где я живу?», «Что я люблю?», «Что ты обо мне помнишь?»);
- приветствия/команды без содержания;
- просьбы к боту (погода, факт о кошках/собаках, картинка собаки) без личных фактов;
- просьбы вспомнить уже сказанное без новой информации.

Если в одном сообщении есть и вопрос, и факт — сохрани только факт.
Верни ТОЛЬКО JSON-массив строк. Если фактов нет — [].

Примеры:
"Меня зовут Анна, живу в Казани" → ["Пользователя зовут Анна", "Живёт в Казани"]
"Привет! Я хочу на Марс!" → ["Хочет на Марс"]
"Я живу в Москве" → ["Живёт в Москве"]
"А где я живу?" → []
"Что ты обо мне помнишь?" → []
"Какая погода в Москве?" → []
"Расскажи факт о кошках" → []
"Расскажи факт о собаках" → []
"Запомни: я живу на Венере" → ["Живёт на Венере"]
"""

SYSTEM_PROMPT = """Ты умный персональный помощник в Telegram.

## Стиль
- Общайся как реальный помощник: тепло, по делу, на языке пользователя.
- Блок «Контекст из памяти» — факты о пользователе из Pinecone (cosine).
  Считай их достоверными (имя, город, учёба, интересы и т.д.).
- Если спрашивают «что ты помнишь / где я живу / как меня зовут» —
  отвечай по этому контексту явно; не говори, что ничего не помнишь,
  пока в контексте есть факты.
- Не выдумывай факты о пользователе, которых нет в контексте.

## Инструменты
- `get_random_cat_fact` — случайный факт о кошках (catfact.ninja).
  Вызывай, если просят факт/интересное про кошек или котов.
- `get_random_dog_fact` — случайный факт о собаках (dogapi.dog).
  Вызывай, если просят факт/интересное про собак без картинки.
- `get_random_dog_image_and_describe` — случайная картинка собаки + vision-описание
  породы и краткой истории породы. Вызывай по запросу про фото/породу собаки.
- `get_current_weather` — текущая погода в городе через Open-Meteo.
  Вызывай, если спрашивают про погоду, температуру, дождь, ветер. Передавай город явно.

Вызывай нужный tool по запросу пользователя. Для обычного разговора tools не нужны.

Важно: картинку бот отправит в Telegram сам. В ответе не вставляй markdown-изображения
и не дублируй длинный URL — достаточно краткого описания породы и истории.
"""


@dataclass
class AssistantReply:
    text: str
    image_url: Optional[str] = None


class PersonalAssistant:
    """Haystack Agent с долговременным контекстом в Pinecone (cosine)."""

    def __init__(
        self,
        *,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        top_k: int = 8,
        # Короткие факты («Я живу в Москве») слабо похожи на «что ты помнишь?»
        # по cosine (~0.20), поэтому порог держим низким.
        similarity_threshold: float = 0.12,
    ) -> None:
        self.openai_api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")
        self.pinecone_api_key = (
            os.getenv("PINICONE_TOKEN")
            or os.getenv("PINECONE_API_KEY")
            or os.getenv("PINECONE_TOKEN")
        )

        if not self.openai_api_key:
            raise ValueError("Не найден OPENAI_KEY / OPENAI_API_KEY в .env")
        if not self.pinecone_api_key:
            raise ValueError("Не найден PINICONE_TOKEN / PINECONE_API_KEY в .env")

        # Haystack читает стандартные имена переменных
        os.environ.setdefault("OPENAI_API_KEY", self.openai_api_key)
        os.environ.setdefault("PINECONE_API_KEY", self.pinecone_api_key)

        self.index_name = index_name or os.getenv(
            "PINECONE_INDEX_NAME", "personal-assistant"
        )
        self.namespace = namespace or os.getenv(
            "PINECONE_NAMESPACE", "personal_assistant"
        )
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.last_image_url: Optional[str] = None

        openai_secret = Secret.from_token(self.openai_api_key)
        pinecone_secret = Secret.from_token(self.pinecone_api_key)

        # Прямая интеграция Haystack ↔ Pinecone, metric=cosine
        self.document_store = PineconeDocumentStore(
            api_key=pinecone_secret,
            index=self.index_name,
            namespace=self.namespace,
            dimension=self.embedding_dimension,
            metric="cosine",
            spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
            show_progress=False,
        )

        self.doc_embedder = OpenAIDocumentEmbedder(
            api_key=openai_secret,
            api_base_url=self.openai_base_url,
            model=self.embedding_model,
            progress_bar=False,
        )
        self.text_embedder = OpenAITextEmbedder(
            api_key=openai_secret,
            api_base_url=self.openai_base_url,
            model=self.embedding_model,
        )
        self.retriever = PineconeEmbeddingRetriever(
            document_store=self.document_store,
            top_k=self.top_k,
        )

        self.fact_llm = OpenAIChatGenerator(
            api_key=openai_secret,
            api_base_url=self.openai_base_url,
            model=self.chat_model,
            generation_kwargs={"temperature": 0, "max_tokens": 400},
        )
        self.tools = build_tools(on_image_ready=self._set_last_image_url)
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(
                api_key=openai_secret,
                api_base_url=self.openai_base_url,
                model=self.chat_model,
                generation_kwargs={"temperature": 0.5, "max_tokens": 1200},
            ),
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            max_agent_steps=8,
        )

    def _set_last_image_url(self, url: str) -> None:
        self.last_image_url = url

    @staticmethod
    def user_filter(user_id: str | int) -> dict[str, Any]:
        return {
            "field": "meta.user_id",
            "operator": "==",
            "value": str(user_id),
        }

    def retrieve_context(
        self,
        query: str,
        *,
        user_id: str | int,
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """Семантический поиск фактов о пользователе (cosine similarity)."""
        embedding = self.text_embedder.run(text=query)["embedding"]
        result = self.retriever.run(
            query_embedding=embedding,
            filters=self.user_filter(user_id),
            top_k=top_k or self.top_k,
        )
        documents = result.get("documents") or []
        return [
            doc
            for doc in documents
            if (doc.score is None) or (doc.score >= self.similarity_threshold)
        ]

    def load_all_facts(self, user_id: str | int) -> list[Document]:
        """Все сохранённые факты пользователя из Pinecone."""
        try:
            docs = self.document_store.filter_documents(filters=self.user_filter(user_id))
        except Exception:
            return []
        result: list[Document] = []
        for doc in docs:
            if not (doc.content or "").strip():
                continue
            kind = (doc.meta or {}).get("kind")
            role = (doc.meta or {}).get("role")
            if role == "assistant" or kind == "assistant":
                continue
            # Старые «сырые» реплики без фактов игнорируем в профиле
            if kind == "user_message":
                continue
            result.append(doc)
        result.sort(key=lambda d: (d.meta or {}).get("created_at") or "", reverse=True)
        return result

    def build_context_documents(
        self,
        query: str,
        *,
        user_id: str | int,
    ) -> list[Document]:
        """
        Контекст для ответа: факты по cosine + полный список фактов пользователя,
        чтобы короткие записи не терялись на вопросах вроде «что ты помнишь?».
        """
        semantic = self.retrieve_context(query, user_id=user_id)
        all_facts = self.load_all_facts(user_id)

        merged: list[Document] = []
        seen: set[str] = set()

        def _add(doc: Document) -> None:
            text = (doc.content or "").strip()
            if not text or text in seen:
                return
            kind = (doc.meta or {}).get("kind")
            if kind == "user_message":
                return
            seen.add(text)
            merged.append(doc)

        for doc in semantic:
            _add(doc)
        for doc in all_facts:
            _add(doc)

        return merged[:20]

    def format_context(self, documents: list[Document]) -> str:
        if not documents:
            return "Пока в памяти пусто."
        lines: list[str] = []
        for i, doc in enumerate(documents, start=1):
            kind = (doc.meta or {}).get("kind", "user_fact")
            score = doc.score
            score_txt = f"{score:.3f}" if isinstance(score, float) else "n/a"
            lines.append(f"[{i}] kind={kind} cosine={score_txt}\n{doc.content}")
        return "\n\n".join(lines)

    @staticmethod
    def _is_savable_fact(text: str) -> bool:
        cleaned = (text or "").strip()
        if not cleaned or cleaned.startswith("/"):
            return False
        if "?" in cleaned or "？" in cleaned:
            return False
        return True

    def extract_facts(self, user_text: str) -> list[str]:
        """Достаёт из реплики только факты о пользователе (как в прошлом уроке)."""
        cleaned = (user_text or "").strip()
        if not cleaned or cleaned.startswith("/"):
            return []

        result = self.fact_llm.run(
            messages=[
                ChatMessage.from_system(EXTRACT_PROMPT),
                ChatMessage.from_user(cleaned),
            ]
        )
        replies = result.get("replies") or []
        raw = (replies[0].text if replies else "") or ""
        raw = raw.strip()
        try:
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            data = json.loads(raw)
            if isinstance(data, list):
                facts = [str(item).strip() for item in data if str(item).strip()]
                return [fact for fact in facts if self._is_savable_fact(fact)]
            print(f"[MEMORY] extract: unexpected JSON type={type(data).__name__} raw={raw!r}")
        except json.JSONDecodeError:
            print(f"[MEMORY] extract: bad JSON raw={raw!r}")
        return []

    def remember_fact(
        self,
        fact: str,
        *,
        user_id: str | int,
    ) -> dict[str, Any]:
        """
        Пишет факт в Pinecone с дедупликацией по cosine similarity
        (как remember() в прошлом уроке).
        """
        cleaned = (fact or "").strip()
        if not cleaned:
            return {"action": "skipped", "reason": "empty"}

        nearest = self.retrieve_context(cleaned, user_id=user_id, top_k=1)
        score = float(nearest[0].score) if nearest and nearest[0].score is not None else None

        # Высокое сходство — обновляем существующий слот
        if nearest and score is not None and score >= FACT_DEDUP_THRESHOLD:
            existing = nearest[0]
            doc_id = existing.id or f"{user_id}-user_fact-{uuid.uuid4().hex[:12]}"
            meta = dict(existing.meta or {})
            meta.update(
                {
                    "user_id": str(user_id),
                    "kind": "user_fact",
                    "role": "user",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            document = Document(id=doc_id, content=cleaned[:3500], meta=meta)
            embedded = self.doc_embedder.run(documents=[document])["documents"]
            self.document_store.write_documents(
                embedded,
                policy=DuplicatePolicy.OVERWRITE,
            )
            action = "updated"
        else:
            doc_id = f"{user_id}-user_fact-{uuid.uuid4().hex[:12]}"
            meta = {
                "user_id": str(user_id),
                "kind": "user_fact",
                "role": "user",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            document = Document(id=doc_id, content=cleaned[:3500], meta=meta)
            embedded = self.doc_embedder.run(documents=[document])["documents"]
            self.document_store.write_documents(
                embedded,
                policy=DuplicatePolicy.OVERWRITE,
            )
            action = "created"

        print(f"[MEMORY] action={action} score={score} id={doc_id} text={cleaned!r}")
        return {"action": action, "id": doc_id, "score": score, "text": cleaned}

    def save_user_facts(self, user_text: str, *, user_id: str | int) -> list[dict[str, Any]]:
        """Сохраняет только извлечённые факты; вопросы и команды не пишет."""
        facts = self.extract_facts(user_text)
        if not facts:
            print(f"[MEMORY] nothing to save for user={user_id}: {user_text!r}")
            return []
        return [self.remember_fact(fact, user_id=user_id) for fact in facts]

    def forget_user(self, user_id: str | int) -> int:
        """Удаляет весь контекст пользователя из Pinecone."""
        before = self.document_store.count_documents()
        self.document_store.delete_by_filter(filters=self.user_filter(user_id))
        after = self.document_store.count_documents()
        return max(0, before - after)

    def list_memories(self, user_id: str | int, query: str = "обо мне") -> list[str]:
        docs = self.load_all_facts(user_id)
        if not docs:
            docs = [
                d
                for d in self.retrieve_context(query, user_id=user_id, top_k=15)
                if (d.meta or {}).get("kind") != "user_message"
            ]

        facts: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            text = (doc.content or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            facts.append(text)
        return facts

    def ask(self, message: str, *, user_id: str | int) -> AssistantReply:
        """
        Главный вход:
        1) достаём релевантные факты из Pinecone (cosine);
        2) запускаем Haystack Agent с tools;
        3) в Pinecone сохраняем только извлечённые факты о пользователе
           (не сырые вопросы вроде «какая погода?» / «что ты помнишь?»).
        """
        self.last_image_url = None
        user_text = (message or "").strip()
        if not user_text:
            return AssistantReply(text="Напиши сообщение — я на связи.")

        # 1. Контекст: факты по cosine + полный профиль
        context_docs = self.build_context_documents(user_text, user_id=user_id)
        context_block = self.format_context(context_docs)

        # 2. Ответ агента
        user_payload = (
            "Контекст из памяти (Pinecone) — факты о пользователе:\n"
            f"{context_block}\n\n"
            f"Сообщение пользователя:\n{user_text}"
        )
        result = self.agent.run(messages=[ChatMessage.from_user(user_payload)])
        last_message = result.get("last_message")
        answer = (getattr(last_message, "text", None) or "").strip()
        if not answer:
            answer = "Готово, но текст ответа пустой. Попробуй переформулировать."

        # 3. В Pinecone — только факты о пользователе (не вопросы/команды)
        self.save_user_facts(user_text, user_id=user_id)

        return AssistantReply(text=answer, image_url=self.last_image_url)

    def test_connection(self, query: str = "привет") -> dict[str, Any]:
        """Проверка: эмбеддинги + Pinecone отвечают."""
        docs = self.retrieve_context(query, user_id="connection-test", top_k=1)
        return {
            "ok": True,
            "index": self.index_name,
            "namespace": self.namespace,
            "metric": "cosine",
            "documents_in_store": self.document_store.count_documents(),
            "hits": len(docs),
        }


if __name__ == "__main__":
    print("Инициализация PersonalAssistant (Haystack + Pinecone)...", flush=True)
    assistant = PersonalAssistant()
    report = assistant.test_connection()
    print("Подключение OK:" if report["ok"] else "FAIL:", report, flush=True)

    print("\nТест tool: факт о кошках...", flush=True)
    from tools import fetch_random_cat_fact, fetch_weather

    print(fetch_random_cat_fact(), flush=True)
    print("\nТест tool: погода в Москве...", flush=True)
    print(fetch_weather("Москва"), flush=True)
