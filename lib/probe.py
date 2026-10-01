"""Измерение скорости на живом сервере, на реальной глубине контекста.

Зачем: тюнер экономно проверяет короткий контекст, а глубокая фаза на 114688
токенов стоит часов (холодный prefill). Но именно глубокий tg интересует на
практике. Поэтому замер делается на уже загруженной модели обычным запросом:
llama-server сам считает timings (prompt_n / predicted_n / *_ms), и их достаточно.

Тот же приём позволяет проверить зрение: картинка уходит в запрос, ответ
доказывает, что mmproj подхватился.
"""

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import server

LOREM = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "Sphinx of black quartz, judge my vow. "
    "The five boxing wizards jump quickly. "
)


def tokenize(text: str, model: str | None = None) -> int:
    """Длина текста в токенах этой модели.

    Роутер требует поле model в /tokenize, одиночный сервер — нет, поэтому
    передаём его всегда, когда он известен.
    """
    body: dict = {"content": text}
    if model:
        body["model"] = model
    j = server.http("POST", f"{server.server_url()}/tokenize", body, timeout=60)
    if isinstance(j, dict) and isinstance(j.get("tokens"), list):
        return len(j["tokens"])
    return 0


def prompt_from_file(path: str | Path, target_tokens: int,
                     model: str | None = None, tail: int = 0) -> str:
    """Промпт из реального файла — честная замена псевдослучайным словам.

    Проба «без повторов» из выдуманного словаря частично зацикливает модель,
    и результат зависит от того, как сборка обращается с зацикливанием, а не от
    скорости на реальной нагрузке. Реальный код или проза такого эффекта не дают.
    """
    text = Path(path).expanduser().read_text(encoding="utf-8", errors="replace")
    if target_tokens <= 0:
        return text
    have = tokenize(text, model)
    if have <= target_tokens:
        return text
    # режем по строкам, пока не попадём в объём
    lines = text.splitlines(keepends=True)
    while lines and tokenize("".join(lines), model) > target_tokens:
        lines = lines[: int(len(lines) * 0.8) or 1]
    # хвост нужен, иначе модель считает файл завершённым и выдаёт EOS.
    # Файл всегда режется по строкам, поэтому все хвосты безопасны и здесь.
    return "".join(lines) + CONTINUATIONS[tail % len(CONTINUATIONS)]


def build_prompt(target_tokens: int, varied: bool = False,
                 model: str | None = None, tail: int = 0) -> str:
    """Текст примерно на target_tokens токенов.

    По умолчанию — повторяющийся (lorem). Так удобно мерить prefill и удобно
    для ngram, но именно повторы дают ускорителю преимущество. Для честной
    оценки ускорителя нужен varied: текст без повторов, где ngram не может
    найти совпадений.
    """
    if target_tokens <= 0:
        return ""
    if varied:
        return build_varied_prompt(target_tokens, tail=tail)
    unit = LOREM
    per_unit = tokenize(unit, model)
    if per_unit <= 0:
        # эндпоинт не ответил — грубая оценка по символам (4 симв./токен)
        per_unit = max(1, len(unit) // 4)
    need = max(1, target_tokens // per_unit)
    text = (unit + "\n") * need
    # самопроверка: если оценка промахнулась, подрезаем по факту
    actual = tokenize(text, model)
    if actual > target_tokens * 1.2 and actual > 0:
        text = "\n".join(text.splitlines()[:int(len(text.splitlines())
                                               * target_tokens / actual)])
    return text


# словарь для неповторяющегося текста; детерминированный, чтобы замер был
# воспроизводим между прогонами
WORDS = (
    "серафим керамика пыль маршрут фонарь атмосфера вертикаль кедр люстра "
    "протокол турбина линза карта прилив снег гарнитура ветер керамика "
    "галерея прибор светильник орбита соль пыльца линия контур якорь "
    "парус туман пламя зеркало контур линза мост туман парус якорь орбита "
    "галерея светильник прибор снег карта прилив гарнитура ветер кедр "
    "протокол турбина люстра вертикаль атмосфера фонарь маршрут пыль серафим"
).split()


# Хвосты-продолжения перебираются по порядку.
#
# Первыми идут слова-подсказки: они уместнее всего, но годятся не везде. На ik
# Qwen3.8 MiniPlus любой явный хвост из слов («Продолжи», «# продолжение»)
# немедленно вызывал EOS — модель читала его как законченную реплику и
# завершала реплику. А отступ или голый перевод строки работают: они выглядят как
# незавершённый код, и модель дописывает следующую строку.
#
# Порядок не случаен: сначала осмысленные хвосты, потом чисто синтаксические.
CONTINUATIONS = (
    "\nПродолжи:\n",
    " и дальше",
    "\nСледующий абзац:",
    "\n   ",
    "\n ",
)


def build_varied_prompt(target_tokens: int, seed: int = 12345,
                        tail: int = 0) -> str:
    """Неповторяющийся текст: псевдослучайная последовательность слов."""
    import random
    rnd = random.Random(seed)
    per_line = 12
    lines = []
    total_words = max(1, target_tokens // 3)
    while len(lines) * per_line < total_words:
        chunk = [rnd.choice(WORDS) for _ in range(per_line)]
        lines.append(" ".join(chunk))
    # хвост обязателен: без него модель считает текст завершённым и выдаёт
    # EOS на первом же токене
    return "\n".join(lines) + CONTINUATIONS[tail % len(CONTINUATIONS)]


def _image_block(path: str | Path) -> dict:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"нет картинки: {p}")
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def completion(prompt: str, model: str | None, max_tokens: int,
               image: str | Path | None = None, temperature: float = 0.0,
               timeout: float = 3600.0) -> dict:
    body: dict = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        # Без этого сервер переиспользует KV предыдущего запроса с тем же
        # префиксом: вторая проба приходит с prompt_n = 4 вместо 110000, и
        # замер молча становится прогретым, а не холодным. Повторно замереть
        # ту же глубину не получится — это разные состояния, а не повтор.
        "cache_prompt": False,
    }
    if model:
        body["model"] = model
    if image is not None:
        body = {
            "model": model or "default",
            "messages": [
                {"role": "user",
                 "content": [_image_block(image),
                             {"type": "text", "text": prompt or "Опиши картинку."}]},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "cache_prompt": False,
        }
    # с картинкой запрос уходит в OpenAI-совместимый /v1/chat/completions:
    # у /completion поля prompt нет, и мультимодальность через него не
    # передаётся — сервер отвечает 400 «key 'prompt' not found»
    url = (f"{server.server_url()}/v1/chat/completions" if image is not None
           else f"{server.server_url()}/completion")
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.code}: "
                                       f"{exc.read().decode('utf-8', 'replace')[:400]}"}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    wall = time.monotonic() - t0
    try:
        j = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"не JSON: {exc}"}
    t = j.get("timings") or {}
    if image is not None:
        # OpenAI-совместимый ответ: текст в choices[0].message.content
        content = ""
        choices = j.get("choices") or []
        if choices:
            content = ((choices[0].get("message") or {}).get("content") or "")
        usage = j.get("usage") or {}
        j = {"timings": t, "content": content,
             "tokens_evaluated": usage.get("prompt_tokens"),
             "tokens_predicted": usage.get("completion_tokens")}
        t = t or {}
    out = {
        "ok": True,
        "wall_s": round(wall, 2),
        "prompt_n": t.get("prompt_n") or j.get("tokens_evaluated"),
        "predicted_n": t.get("predicted_n") or j.get("tokens_predicted"),
        "prompt_ms": t.get("prompt_ms"),
        "predicted_ms": t.get("predicted_ms"),
        "content": (j.get("content") or "")[:400],
    }
    # tg не считается, если сгенерировано меньше двух токенов: модель
    # выдала EOS или стоп-слово, и деление даёт абсурдные миллионы t/s
    try:
        pn, pm = t.get("predicted_n"), t.get("predicted_ms")
        if pn and pm and int(pn) >= 2:
            out["gen_tps"] = round(pn / (pm / 1000), 2)
        elif pn:
            out["gen_tps"] = None
            out["note"] = (f"сгенерировано только {pn} ток. — tg не считается "
                           f"(модель остановилась сразу)")
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    try:
        if t.get("prompt_n") and t.get("prompt_ms"):
            out["prefill_tps"] = round(t["prompt_n"] / (t["prompt_ms"] / 1000), 2)
    except (TypeError, ZeroDivisionError):
        pass
    return out


def looks_cached(res: dict, expected_tokens: int | None) -> bool:
    """Проба пришла из прогретого KV, а не разбирала промпт заново.

    Признак — prompt_n заметно меньше ожидаемого. Раньше такие пробы молча
    попадали в статистику как обычные: prefill у них в разы ниже, и на графике
    это выглядит как «внезапно ускорилось».
    """
    if not res.get("ok") or not expected_tokens:
        return False
    got = res.get("prompt_n")
    if not got:
        return False
    # у запроса с картинкой prompt_n — это только текст, он всегда мал
    if res.get("with_image"):
        return False
    return int(got) < max(8, int(expected_tokens * 0.5))


def probe(target_tokens: int = 8192, max_tokens: int = 64,
          image: str | Path | None = None, model: str | None = None,
          timeout: float = 3600.0, varied: bool = False,
          seed: int = 12345, from_file: str | Path | None = None,
          tail: int = 0) -> dict:
    """Один замер. Возвращает распакованные timings."""
    if image is None and from_file:
        prompt = prompt_from_file(from_file, target_tokens, model, tail=tail)
    else:
        prompt = (build_prompt(target_tokens, varied=varied, model=model,
                               tail=tail) if image is None else "")
    res = completion(prompt, model, max_tokens, image=image, timeout=timeout)
    res["target_tokens"] = target_tokens
    res["with_image"] = bool(image)
    res["varied"] = varied
    res["tail"] = tail
    res["from_file"] = str(from_file) if from_file else None
    return res


def _stopped_immediately(res: dict) -> bool:
    """Модель выдала EOS или стоп-слово на первом токене.

    Так бывает, когда хвост-продолжение не подошёл: на ik фраза «Продолжи»
    давала ровно один токен, и замер молча выпадал из статистики.
    """
    if not res.get("ok") or res.get("with_image"):
        return False
    try:
        return int(res.get("predicted_n") or 0) < 2
    except (TypeError, ValueError):
        return False


def probe_repeat(target_tokens: int = 8192, max_tokens: int = 64,
                 repeats: int = 3, image: str | Path | None = None,
                 model: str | None = None, timeout: float = 3600.0,
                 varied: bool = False, seeds: list[int] | None = None,
                 from_file: str | Path | None = None) -> dict:
    """Несколько замеров; худший tg уходит в отчёт — планировать надо по нему.

    Для varied каждая проба получает свой seed: одинаковый текст после первой
    пробы возьмётся из prompt cache и измерять будет нечего. Проб, пришедших
    из прогретого KV, в статистику не пускаем: их prompt_n в разы меньше, и
    средний prefill от этого занижается, а tg выглядит правдоподобно.
    """
    runs = []
    for i in range(max(1, repeats)):
        seed = seeds[i] if seeds and i < len(seeds) else 12345 + i
        r = probe(target_tokens, max_tokens, image=image, model=model,
                  timeout=timeout, varied=varied, seed=seed,
                  from_file=from_file, tail=i)
        # модель молчит на первом токене — хвост не подошёл, пробуем следующий.
        # Текст прыгает по объёму на пару токенов, но это дешевле потерянной
        # пробы, а tg от длины хвоста не зависит.
        tries = 0
        while _stopped_immediately(r) and tries < len(CONTINUATIONS) - 1:
            tries += 1
            r2 = probe(target_tokens, max_tokens, image=image, model=model,
                       timeout=timeout, varied=varied, seed=seed,
                       from_file=from_file, tail=i + tries)
            if _stopped_immediately(r2):
                continue
            r2["retried"] = tries
            r = r2
            break
        if tries and _stopped_immediately(r):
            r["note"] = (f"модель остановилась сразу на всех {len(CONTINUATIONS)} "
                         "хвостах-продолжениях")
        r["cached"] = looks_cached(r, target_tokens)
        runs.append(r)
        if not r.get("ok"):
            break
    ok = [r for r in runs if r.get("ok") and r.get("gen_tps")]
    for r in runs:
        if r.get("ok") and r.get("note"):
            print(f"  проба с tg пропущена: {r['note']}")
    cold = [r for r in ok if not r.get("cached")]
    out = {"runs": runs, "ok_count": len(ok)}
    if ok:
        tps = [r["gen_tps"] for r in ok]
        out["gen_tps_min"] = round(min(tps), 2)
        out["gen_tps_max"] = round(max(tps), 2)
        out["gen_tps_mean"] = round(sum(tps) / len(tps), 2)
        pre = [r["prefill_tps"] for r in cold if r.get("prefill_tps")]
        if pre:
            out["prefill_tps"] = round(sum(pre) / len(pre), 2)
        out["prefill_runs"] = len(pre)
        out["cached_runs"] = len(ok) - len(cold)
        if out["cached_runs"]:
            print(f"  из {len(ok)} проб в прогноз prefill попало {len(pre)}: "
                  f"{out['cached_runs']} пришли из прогретого KV и не учитываются")
        if not cold:
            out["prefill_tps"] = None
            out["note"] = ("все пробы пришли из прогретого KV — честного "
                           "холодного prefill в этой серии нет")
        out["prompt_n"] = (cold or ok)[0].get("prompt_n")
    return out