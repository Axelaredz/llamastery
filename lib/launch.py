"""Shared launch-option handling for the wizard and tuner (no shell execution)."""
import re
import shlex
from pathlib import Path

from . import schema

# Stable internal spellings used by budget/tuning code.
KEYS = {
    "model": "model", "m": "model", "ctx-size": "c", "c": "c",
    "batch-size": "b", "b": "b", "threads": "t", "t": "t",
    "flash-attn": "fa", "fa": "fa", "override-tensor": "ot", "ot": "ot",
    "ngl": "n-gpu-layers", "gpu-layers": "n-gpu-layers",
    "ncmoe": "n-cpu-moe", "ncffn": "n-cpu-ffn", "ub": "ubatch-size", "tb": "threads-batch",
    "ctk": "cache-type-k", "ctv": "cache-type-v",
}
CONTROL = {"host", "port", "alias", "api-key", "api-key-file", "models-dir",
           "models-max", "models-preset", "models-autoload", "help", "version"}


def option_key(key, flags):
    f = schema.resolve(flags, key)
    canon = f.canonical.lstrip("-") if f else key.lstrip("-")
    # Negative aliases are actions, not synonyms for the positive switch.
    if f and key.lstrip("-").startswith("no-"):
        canon = next((a.lstrip("-") for a in f.aliases if a.startswith("--no-")),
                     key.lstrip("-"))
    return KEYS.get(canon, canon)


def normalize(pairs, flags):
    out = {}
    for k, v in pairs.items():
        key = option_key(k, flags)
        if key in {"no-kv-unified", "no-mmproj-offload", "no-op-offload", "no-repack"}:
            key = key.removeprefix("no-")
            v = "true" if str(v).lower() in {"0", "false", "no", "off", "disabled"} else "false"
        out[key] = v
    return out


def effective_pairs(ini, name, flags=None):
    flags = flags or {}
    sec = ini.section(name)
    if sec is None:
        raise ValueError("Секция не найдена")
    # version is INI format metadata, not llama-server --version.
    defaults = {k: v for k, v in ini.globals().items() if k != "version"}
    return {**normalize(defaults, flags), **normalize(sec.pairs(), flags)}


def parse_cli(text, flags):
    """Parse server flags; reject unknown/control options rather than lose them.

    Values remain single argv entries, including spaces, equals and negatives.
    This intentionally accepts flags only, not an executable or shell command.
    """
    tokens = shlex.split(text)
    out = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        name, eq, value = token.partition("=")
        f = schema.resolve(flags, name) if name.startswith("-") else None
        if f is None:
            raise ValueError(f"неизвестный флаг выбранной сборки: {name}")
        key = option_key(name, flags)
        if key in CONTROL or key in {"model", "hf-repo", "hf-file", "model-url"}:
            raise ValueError(f"{name}: путь к модели и управление сервером задаются отдельно")
        if f.kind == "flag":
            if not eq:
                value = "true"
            if value.lower() not in {"true", "false", "1", "0", "yes", "no", "on", "off", "enabled", "disabled"}:
                raise ValueError(f"{name}: ожидается булево значение")
        elif not eq:
            i += 1
            if i == len(tokens) or (tokens[i].startswith("-") and
                                    not re.fullmatch(r"-\d+(?:\.\d+)?", tokens[i])):
                raise ValueError(f"{name}: не задано значение")
            value = tokens[i]
        if not value and f.kind != "flag":
            raise ValueError(f"{name}: пустое значение")
        if f.kind in {"int", "float"} and not (key == "n-gpu-layers" and value in {"auto", "all"}):
            try:
                float(value)  # N in --help is also used for floating-point options
            except ValueError:
                raise ValueError(f"{name}: ожидается число") from None
        # The INI parser treats whitespace + #/; as a comment. Do not silently
        # truncate an arbitrary CLI value when persisting it.
        if "\n" in value or "\r" in value or re.search(r"\s[;#]", value):
            raise ValueError(f"{name}: значение нельзя без потерь записать в INI")
        if key in out:
            raise ValueError(f"{name}: повторный флаг; INI хранит одно значение на ключ")
        if f.kind == "path":
            value = str(Path(value).expanduser().resolve())
        out[key] = value
        i += 1
    return out


def resolve_option(flags, key):
    f = schema.resolve(flags, key)
    if f is None:
        f = next((schema.resolve(flags, alias) for alias, internal in KEYS.items()
                  if internal == key and schema.resolve(flags, alias)), None)
    return f


def flag_argv(key, value, flags):
    """Serialize one schema-known option, preserving negative switches."""
    f = resolve_option(flags, key)
    if f is None:
        raise ValueError(f"неизвестный флаг выбранной сборки: {key}")
    name = f.canonical
    if key.lstrip("-").startswith("no-"):
        name = "--" + key.lstrip("-")
    value = str(value)
    if f.kind != "flag":
        return [name, value]
    if value.lower() in {"0", "false", "no", "off", "disabled"}:
        if name.startswith("--no-"):
            return [f.canonical] if not f.canonical.startswith("--no-") else []
        negative = "--no-" + name.lstrip("-")
        if schema.resolve(flags, negative):
            return [negative]
        # A false opt-in switch must not accidentally become true.
        return []
    return [name]
