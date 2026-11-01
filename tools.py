"""Инструменты агента: факты о кошках, картинка собаки + vision, погода."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Callable

import requests
from dotenv import load_dotenv
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ImageContent
from haystack.tools import tool
from haystack.utils import Secret

load_dotenv(Path(__file__).resolve().parent / ".env")

CAT_FACTS_URL = "https://catfact.ninja/fact"
DOG_FACTS_URL = "https://dogapi.dog/api/v2/facts"
DOG_IMAGE_URL = "https://dog.ceo/api/breeds/image/random"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

DOG_VISION_PROMPT = (
    "На картинке собака. Определи наиболее вероятную породу (или помесь) "
    "и коротко опиши:\n"
    "1) как выглядит порода;\n"
    "2) краткую предысторию: откуда порода, для чего выводилась;\n"
    "3) 2–3 характерные черты характера.\n"
    "Отвечай на русском, живым языком, без воды. Если породу нельзя "
    "уверенно определить — скажи об этом и опиши видимые признаки."
)

WEATHER_CODE_RU = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "изморозь / туман",
    51: "лёгкая морось",
    53: "морось",
    55: "сильная морось",
    61: "небольшой дождь",
    63: "дождь",
    65: "сильный дождь",
    71: "небольшой снег",
    73: "снег",
    75: "сильный снег",
    80: "ливневый дождь",
    81: "ливень",
    82: "сильный ливень",
    95: "гроза",
    96: "гроза с градом",
    99: "сильная гроза с градом",
}


def _openai_secret() -> Secret:
    return Secret.from_token(
        os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY") or ""
    )


def _vision_generator() -> OpenAIChatGenerator:
    return OpenAIChatGenerator(
        api_key=_openai_secret(),
        api_base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        generation_kwargs={"temperature": 0.4, "max_tokens": 700},
    )


def fetch_random_cat_fact() -> str:
    """GET к бесплатному API: случайный факт о кошках."""
    response = requests.get(CAT_FACTS_URL, timeout=20)
    response.raise_for_status()
    payload = response.json()
    fact = (payload.get("fact") or "").strip()
    if not fact:
        raise RuntimeError("API фактов о кошках вернул пустой ответ")
    return fact


def fetch_random_dog_fact() -> str:
    """GET к бесплатному API: случайный факт о собаках."""
    response = requests.get(DOG_FACTS_URL, params={"limit": 1}, timeout=20)
    response.raise_for_status()
    payload = response.json()
    items = payload.get("data") or []
    if not items:
        raise RuntimeError("API фактов о собаках вернул пустой ответ")
    body = ((items[0].get("attributes") or {}).get("body") or "").strip()
    if not body:
        raise RuntimeError("В ответе API нет текста факта")
    return body


def fetch_random_dog_image_url() -> str:
    """GET к бесплатному API: URL случайной картинки собаки."""
    response = requests.get(DOG_IMAGE_URL, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success" or not payload.get("message"):
        raise RuntimeError(f"Неожиданный ответ dog.ceo: {payload}")
    return str(payload["message"])


def describe_dog_image(image_url: str) -> str:
    """Загружает картинку в vision-модель OpenAI и описывает породу."""
    image = ImageContent.from_url(image_url, detail="low")
    message = ChatMessage.from_user(content_parts=[DOG_VISION_PROMPT, image])
    result = _vision_generator().run([message])
    replies = result.get("replies") or []
    if not replies or not replies[0].text:
        raise RuntimeError("Vision-модель вернула пустой ответ")
    return replies[0].text.strip()


def fetch_weather(city: str) -> str:
    """Текущая погода через Open-Meteo (геокодинг + forecast), без API-ключа."""
    city = (city or "").strip()
    if not city:
        raise ValueError("Не указан город")

    geo = requests.get(
        GEOCODING_URL,
        params={"name": city, "count": 1, "language": "ru", "format": "json"},
        timeout=20,
    )
    geo.raise_for_status()
    results = (geo.json() or {}).get("results") or []
    if not results:
        raise RuntimeError(f"Город не найден: {city}")

    place = results[0]
    lat = place["latitude"]
    lon = place["longitude"]
    label = place.get("name") or city
    country = place.get("country") or ""
    admin = place.get("admin1") or ""
    where = ", ".join(part for part in (label, admin, country) if part)

    weather = requests.get(
        WEATHER_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "auto",
            "wind_speed_unit": "ms",
        },
        timeout=20,
    )
    weather.raise_for_status()
    current = (weather.json() or {}).get("current") or {}
    if not current:
        raise RuntimeError("Open-Meteo не вернул текущую погоду")

    code = current.get("weather_code")
    condition = WEATHER_CODE_RU.get(code, f"код погоды {code}")
    temp = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    wind = current.get("wind_speed_10m")

    return (
        f"Погода сейчас в {where}:\n"
        f"• Условия: {condition}\n"
        f"• Температура: {temp} °C\n"
        f"• Влажность: {humidity}%\n"
        f"• Ветер: {wind} м/с\n"
        f"Источник: Open-Meteo"
    )


def build_tools(
    *,
    on_image_ready: Callable[[str], None] | None = None,
) -> list:
    """Собирает Haystack-tools для PersonalAssistant."""

    @tool
    def get_random_cat_fact() -> str:
        """Получает случайный факт о кошках с бесплатного внешнего API (catfact.ninja).
        Вызывай, когда пользователь просит факт/интересное про кошек или котов.
        """
        try:
            fact = fetch_random_cat_fact()
        except Exception as exc:  # noqa: BLE001
            return f"Не удалось получить факт о кошках: {exc}"
        return f"Случайный факт о кошках: {fact}\nИсточник: {CAT_FACTS_URL}"

    @tool
    def get_random_dog_fact() -> str:
        """Получает случайный факт о собаках с бесплатного внешнего API (dogapi.dog).
        Вызывай, когда пользователь просит факт/интересное про собак (без картинки).
        """
        try:
            fact = fetch_random_dog_fact()
        except Exception as exc:  # noqa: BLE001
            return f"Не удалось получить факт о собаках: {exc}"
        return f"Случайный факт о собаках: {fact}\nИсточник: {DOG_FACTS_URL}"

    @tool
    def get_random_dog_image_and_describe(
        note: Annotated[
            str,
            "Короткий комментарий, зачем нужна картинка (можно пустую строку)",
        ] = "",
    ) -> str:
        """Получает случайную картинку собаки с dog.ceo, отправляет её в OpenAI vision
        и возвращает описание породы + краткую предысторию породы.
        Вызывай, когда пользователь хочет картинку/фото собаки или узнать породу.
        """
        try:
            image_url = fetch_random_dog_image_url()
            description = describe_dog_image(image_url)
        except Exception as exc:  # noqa: BLE001
            return f"Не удалось получить/описать картинку собаки: {exc}"

        if on_image_ready is not None:
            on_image_ready(image_url)

        extra = f"\nЗаметка агента: {note}" if note.strip() else ""
        return (
            f"Картинка собаки: {image_url}\n\n"
            f"Описание породы от vision-модели:\n{description}"
            f"{extra}\n"
            f"Источник изображения: {DOG_IMAGE_URL}"
        )

    @tool
    def get_current_weather(
        city: Annotated[str, "Город, для которого нужна текущая погода"],
    ) -> str:
        """Новый инструмент: текущая погода в указанном городе через Open-Meteo.
        Вызывай, когда пользователь спрашивает про погоду, температуру, дождь, ветер.
        """
        try:
            return fetch_weather(city)
        except Exception as exc:  # noqa: BLE001
            return f"Не удалось получить погоду для «{city}»: {exc}"

    return [
        get_random_cat_fact,
        get_random_dog_fact,
        get_random_dog_image_and_describe,
        get_current_weather,
    ]
