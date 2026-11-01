"""Telegram-интерфейс умного персонального помощника (PyTelegramBotAPI)."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import telebot
from dotenv import load_dotenv
from telebot import types

from assistant import PersonalAssistant

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("personal-assistant-bot")

BOT_TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("Не найден BOT_TOKEN в .env")

ASK_TIMEOUT_SEC = int(os.getenv("ASK_TIMEOUT_SEC", "150"))

HELP_TEXT = (
    "Я умный персональный помощник.\n\n"
    "Умею:\n"
    "• вести диалог с учётом твоего контекста (в Pinecone — только факты о тебе)\n"
    "• выдавать случайный факт о кошках или о собаках\n"
    "• присылать случайную картинку собаки и описывать породу через OpenAI vision\n"
    "• сообщать текущую погоду в городе (Open-Meteo)\n\n"
    "Примеры:\n"
    "• Меня зовут Алекс, я изучаю Python\n"
    "• Расскажи факт о кошках\n"
    "• Расскажи факт о собаках\n"
    "• Покажи случайную собаку и расскажи про породу\n"
    "• Какая погода в Москве?\n"
    "• Что ты обо мне помнишь?\n\n"
    "Команды: /start /help /memory /forget"
)

bot = telebot.TeleBot(BOT_TOKEN, threaded=True)
executor = ThreadPoolExecutor(max_workers=4)

logger.info("Инициализация PersonalAssistant...")
assistant = PersonalAssistant()
logger.info(
    "Готов: index=%s namespace=%s metric=cosine",
    assistant.index_name,
    assistant.namespace,
)


def _split_message(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks or [text[:limit]]


def _safe_send(chat_id: int, text: str, *, reply_to: int | None = None) -> None:
    for part in _split_message(text):
        try:
            bot.send_message(chat_id, part, reply_to_message_id=reply_to)
        except telebot.apihelper.ApiTelegramException:
            logger.exception("Не удалось отправить сообщение chat_id=%s", chat_id)
            try:
                bot.send_message(chat_id, part)
            except Exception:
                logger.exception("Повторная отправка тоже не удалась")


def _send_photo(chat_id: int, image_url: str, caption: str | None = None) -> None:
    try:
        bot.send_photo(chat_id, image_url, caption=(caption or "")[:1024] or None)
    except telebot.apihelper.ApiTelegramException:
        logger.exception("Не удалось отправить фото, шлю ссылку")
        _safe_send(chat_id, f"Картинка: {image_url}")


@bot.message_handler(commands=["start", "help"])
def handle_start(message: types.Message) -> None:
    bot.reply_to(message, f"Привет! {HELP_TEXT}")


@bot.message_handler(commands=["memory"])
def handle_memory(message: types.Message) -> None:
    user_id = message.from_user.id
    bot.send_chat_action(message.chat.id, "typing")
    try:
        facts = assistant.list_memories(user_id)
    except Exception:
        logger.exception("Ошибка /memory user=%s", user_id)
        bot.reply_to(message, "Не удалось прочитать память. Попробуй позже.")
        return

    if not facts:
        bot.reply_to(
            message,
            "Пока ничего не помню. Расскажи о себе или напиши «запомни …».",
        )
        return

    lines = ["Вот что нашлось в памяти (по cosine similarity):"]
    for i, fact in enumerate(facts[:10], start=1):
        lines.append(f"{i}. {fact}")
    bot.reply_to(message, "\n".join(lines))


@bot.message_handler(commands=["forget"])
def handle_forget(message: types.Message) -> None:
    user_id = message.from_user.id
    try:
        removed = assistant.forget_user(user_id)
        bot.reply_to(message, f"Память очищена (удалено записей ≈ {removed}).")
    except Exception:
        logger.exception("Ошибка /forget user=%s", user_id)
        bot.reply_to(message, "Не удалось очистить память. Попробуй позже.")


@bot.message_handler(content_types=["text"])
def handle_text(message: types.Message) -> None:
    text = (message.text or "").strip()
    if not text or text.startswith("/"):
        return

    user_id = message.from_user.id
    chat_id = message.chat.id
    logger.info("user=%s text=%r", user_id, text[:200])

    status = None
    try:
        status = bot.reply_to(message, "Думаю: смотрю контекст и готовлю ответ…")
    except telebot.apihelper.ApiTelegramException:
        logger.exception("Не удалось отправить статус")

    bot.send_chat_action(chat_id, "typing")

    future = executor.submit(assistant.ask, text, user_id=user_id)
    image_url = None
    try:
        reply = future.result(timeout=ASK_TIMEOUT_SEC)
        answer = reply.text
        image_url = reply.image_url
        logger.info("user=%s answer_len=%s image=%s", user_id, len(answer or ""), bool(image_url))
    except FuturesTimeout:
        logger.error("Таймаут ask() user=%s after %ss", user_id, ASK_TIMEOUT_SEC)
        answer = (
            "Слишком долго жду ответ от модели/памяти. "
            "Попробуй ещё раз чуть позже."
        )
    except Exception:
        logger.exception("Ошибка обработки сообщения user=%s", user_id)
        answer = "Не удалось обработать сообщение. Попробуй ещё раз."

    if not answer or not str(answer).strip():
        answer = "Получил пустой ответ. Попробуй переформулировать."

    # Сначала картинка (если tool её достал), потом текст
    if image_url:
        _send_photo(chat_id, image_url, caption="Случайная собака 🐶")

    parts = _split_message(str(answer))
    if status is not None:
        try:
            bot.edit_message_text(parts[0], chat_id=chat_id, message_id=status.message_id)
        except telebot.apihelper.ApiTelegramException:
            _safe_send(chat_id, parts[0], reply_to=message.message_id)
        for part in parts[1:]:
            _safe_send(chat_id, part)
    else:
        for part in parts:
            _safe_send(chat_id, part, reply_to=message.message_id)


if __name__ == "__main__":
    logger.info("Бот запущен (polling), timeout ask=%ss", ASK_TIMEOUT_SEC)
    try:
        bot.infinity_polling(skip_pending=True, timeout=60, long_polling_timeout=50)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
