# Personal Assistant — Haystack Agent + Pinecone + Telegram

Умный персональный помощник на [Haystack](https://docs.haystack.deepset.ai/docs/get-started):  
Telegram-бот (`PyTelegramBotAPI`) + Agent с tools + долговременный контекст в Pinecone  
через официальную интеграцию [`pinecone-haystack`](https://docs.haystack.deepset.ai/reference/integrations-pinecone)  
с метрикой **cosine similarity**.

## Что умеет

- вести диалог как реальный помощник, опираясь на персональный контекст;
- хранить в Pinecone **только извлечённые факты о пользователе** (не вопросы/команды);
- перед ответом доставать релевантный контекст через `PineconeEmbeddingRetriever`;
- tool: случайный факт о кошках (`https://catfact.ninja/fact`);
- tool: случайный факт о собаках (`https://dogapi.dog/api/v2/facts`);
- tool: случайная картинка собаки (`dog.ceo`) → OpenAI vision → описание породы;
- tool: текущая погода в городе через Open-Meteo (`get_current_weather`);
- команды `/start`, `/help`, `/memory`, `/forget`.

## Как используется Haystack

Haystack — «мозг» помощника: память в Pinecone, агент с tools и LLM.  
Telegram (`bot.py`) только принимает сообщения и вызывает `PersonalAssistant.ask()`.

```text
Telegram → bot.py → PersonalAssistant.ask()
                         │
         ┌───────────────┼──────────────────┐
         ▼               ▼                  ▼
   достать факты    Agent (+ tools)    извлечь факты
   из Pinecone      ответить юзеру     и записать в Pinecone
```

### Компоненты и зачем они

| Компонент | Где | Роль у нас |
|---|---|---|
| **`PineconeDocumentStore`** | `assistant.py` | Хранилище фактов о пользователе в Pinecone (`metric=cosine`, namespace `personal_assistant`). |
| **`Document`** | `assistant.py` | Единица памяти: текст факта + meta (`user_id`, `kind=user_fact`). |
| **`OpenAIDocumentEmbedder`** | запись | Делает эмбеддинг факта перед `write_documents`. |
| **`OpenAITextEmbedder`** | чтение | Эмбеддит текущий вопрос пользователя для поиска. |
| **`PineconeEmbeddingRetriever`** | чтение | Ищет ближайшие факты по cosine (с фильтром `user_id`). |
| **`DuplicatePolicy.OVERWRITE`** | запись | При дедупе перезаписывает похожий факт. |
| **`Agent`** | ответ | Главный tool-calling цикл: LLM решает, вызывать ли tool или ответить текстом. |
| **`OpenAIChatGenerator`** | агент + факты + vision | Чат-модель: (1) внутри `Agent`, (2) извлечение фактов, (3) описание картинки собаки. |
| **`ChatMessage`** | везде | Формат сообщений system/user для генераторов и агента. |
| **`@tool` / `Tool`** | `tools.py` | Обёртки функций для агента: кошки, собаки (факт), собака+vision, погода. |
| **`ImageContent`** | `tools.py` | Картинка собаки → base64 → vision-модель. |
| **`Secret`** | конфиг | Безопасная передача API-ключей в компоненты Haystack. |

### Два пути данных

**Чтение (перед ответом)**  
`TextEmbedder` → `PineconeEmbeddingRetriever` → факты в промпт агента (+ полный список фактов пользователя).

**Запись (после ответа)**  
`extract_facts` через `OpenAIChatGenerator` → только факты → `DocumentEmbedder` → `PineconeDocumentStore`.  
Вопросы («какая погода?», «факт о собаках») не пишутся.

**Агент**  
`Agent` + `OpenAIChatGenerator` + tools: сам выбирает  
`get_random_cat_fact` / `get_random_dog_fact` / `get_random_dog_image_and_describe` / `get_current_weather`  
или отвечает без tools, опираясь на контекст из Pinecone.

## Структура

```text
.
├── assistant.py       # Haystack Agent + Pinecone память (cosine)
├── tools.py           # Tools: cat/dog fact, dog image+vision, weather
├── bot.py             # Telegram-интерфейс (PyTelegramBotAPI)
├── requirements.txt
├── .env.example
├── .env               # локальные ключи (НЕ коммитится)
└── README.md
```

## Быстрый старт

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # затем заполнить ключи
python assistant.py      # проверка Pinecone + tools
python bot.py
```

Нужны: `OPENAI_KEY`, `OPENAI_BASE_URL`, `PINICONE_TOKEN`, `PINECONE_INDEX_NAME`, `BOT_TOKEN`.  
Индекс Pinecone: dimension `1536`, metric `cosine`.

## Как закрыто домашнее задание

| Пункт ДЗ | Реализация |
|---|---|
| 1. Новый инструмент в Haystack-агенте | `get_current_weather` в `tools.py`, подключён через `build_tools()` → `Agent(tools=...)` |
| 2. Обновлён системный промпт | `SYSTEM_PROMPT` в `assistant.py` описывает все tools и когда их вызывать |
| 3. В Pinecone только полезный текст от пользователя | `extract_facts()` + `save_user_facts()`: вопросы/команды не пишутся, только факты; ответы ассистента тоже не пишутся |
| 4. Ответы с опорой на контекст Pinecone | перед ответом `retrieve_context()` (cosine) + блок «Контекст из памяти» |
| 5. Проверки в боте | факт о кошках / факт о собаках / анализ картинки собаки / погода |

## Примеры в боте

- `Меня зовут Алекс, я изучаю Python`
- `Какой язык я изучаю?` — ответ из контекста Pinecone
- `Расскажи факт о кошках`
- `Расскажи факт о собаках`
- `Покажи случайную собаку и расскажи про породу`
- `Какая погода в Москве?`
- `/memory` · `/forget`

## Документация

- [Haystack Get Started](https://docs.haystack.deepset.ai/docs/get-started)
- [Pinecone integration](https://docs.haystack.deepset.ai/reference/integrations-pinecone)
