# Релевант-слой для блога — minimal version

Оценивает виральные посты других: «залетит ли у меня похожий?»

## Что нужно (всё бесплатное на старте)

1. **OpenAI API ключ** — https://platform.openai.com/api-keys (для embeddings, ~$0.02/M токенов)
2. **OpenRouter API ключ** — https://openrouter.ai/keys (доступ к Haiku 4.5, ~$1/M токенов)
3. Python 3.10+ с openai, requests, numpy

## 5 шагов

### 1. Создайте проект
```bash
mkdir my-trend-scorer && cd $_
python3 -m venv .venv && source .venv/bin/activate
pip install openai requests numpy
```

### 2. Скачайте файлы
- relevance_layer.py — основной код
- context-template.md → переименуйте в context.md и заполните под себя

### 3. Положите свои посты
В папку `my_posts/` положите .md файлы ваших опубликованных постов. Чем больше — тем точнее эмбеддинг-отпечаток.

### 4. Подготовьте кандидатов
Где взять виральные посты других — на ваш выбор:
- Reddit OAuth API + top.json по сабам
- Threads search через Scrapling/Playwright
- Twitter/X API
Сохраните в `candidates.json` как `[{"text": "...", "likes": 123, "comments": 45}, ...]`

### 5. Запустите
```bash
export OPENAI_API_KEY=sk-...
export OPENROUTER_API_KEY=sk-or-...
python relevance_layer.py
```

В выводе — посты со score, angle, reasoning.

## Стоимость
~$1/мес при ежедневном запуске на 30-50 кандидатов. Главный расход — Haiku 4.5 на оценку.

## Что подкручивать
- `CUTOFF` (5) — порог попадания в выдачу
- `MIN_SIM` (0.30) — порог embedding-отсева
- `TOP_N` (10) — сколько отдавать LLM

## Расширения
- Cron + Telegram-бот для ежедневного дайджеста
- Кэш embeddings для своих постов (чтобы не пересчитывать)
- Адаптеры для разных источников трендов
