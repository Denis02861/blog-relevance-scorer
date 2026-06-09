"""Релевант-слой для оценки трендов под ваш блог.

Что делает:
1. Берёт ваши опубликованные посты, превращает в embeddings (OpenAI)
2. На каждый кандидат-пост (виральный у других) считает близость
3. Топ-N близких прогоняет через LLM-оценщик (Haiku 4.5 через OpenRouter)
4. Возвращает посты со score 0-10 и готовым angle

Требования: openai, requests, numpy
Ключи: OPENAI_API_KEY, OPENROUTER_API_KEY
"""
import json, math, os, requests
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "anthropic/claude-haiku-4.5"
CUTOFF = 5  # минимальный llm_score для попадания в выдачу
TOP_N = 10  # сколько кандидатов отдавать LLM после embedding-отсева
MIN_SIM = 0.30  # минимальная similarity для попадания в LLM

def embed(texts: list[str]) -> list[list[float]]:
    cli = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = cli.embeddings.create(model=EMBED_MODEL, input=[t[:8000] for t in texts])
    return [d.embedding for d in r.data]

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

def llm_batch(candidates: list[dict], context: str) -> dict:
    """candidates: [{idx, text, likes, comments}, ...]"""
    user_lines = ["Оцени каждый пост. Может ли у меня залететь похожий пост на 50+ лайков?", "", "Посты:"]
    for c in candidates:
        user_lines.append(f"[{c['idx']}] ❤{c['likes']} 💬{c['comments']}")
        user_lines.append(c['text'][:1500])
        user_lines.append("")
    user_lines.append('Верни JSON: [{"id":N,"score":0-10,"angle":"...","reasoning":"..."}]')
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": context}, {"role": "user", "content": "\n".join(user_lines)}],
        "temperature": 0.3, "max_tokens": 2000,
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                      json=payload,
                      headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                      timeout=120)
    content = r.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.strip("`").lstrip("json").strip()
    return {int(x["id"]): x for x in json.loads(content)}

def rank(my_posts: list[str], context: str, candidates: list[dict]) -> list[dict]:
    """my_posts: список ваших опубликованных постов (тексты).
    context: контекст блога (см. context-template.md).
    candidates: [{text, likes, comments}, ...] виральные у других.
    Возвращает: отсортированные RelevanceScore."""
    if not my_posts or not candidates: return []
    my_vec = embed(my_posts)
    cand_vec = embed([c['text'] for c in candidates])
    scored = []
    for i, (c, v) in enumerate(zip(candidates, cand_vec)):
        sim = max(cosine(v, m) for m in my_vec)
        if sim >= MIN_SIM:
            scored.append({**c, "idx": i+1, "similarity": sim})
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top = scored[:TOP_N]
    if not top: return []
    verdicts = llm_batch(top, context)
    results = []
    for c in top:
        v = verdicts.get(c["idx"])
        if v and int(v.get("score", 0)) >= CUTOFF:
            results.append({**c, **v})
    results.sort(key=lambda x: int(x["score"]), reverse=True)
    return results

# Пример использования
if __name__ == "__main__":
    context = open("context.md").read()
    my_posts = [open(f).read() for f in os.listdir("my_posts/") if f.endswith(".md")]
    candidates = json.load(open("candidates.json"))  # [{text, likes, comments}, ...]
    for r in rank(my_posts, context, candidates):
        print(f"{r['score']}/10 — {r['angle']}")
        print(f"  reasoning: {r['reasoning']}\n")
