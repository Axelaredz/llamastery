"""Минимальный ридер метаданных GGUF (только stdlib).

Читает шапку и KV-блок, не читая тензоры. Этого хватает для оценки
размера весов, числа слоёв, голов и контекста — то есть для расчёта
потребления VRAM.

Спецификация: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

import struct
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# GGUF value types
(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING,
 ARRAY, UINT64, INT64, FLOAT64) = range(13)

_SCALARS = {
    UINT8: ("<B", 1), INT8: ("<b", 1), UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4), FLOAT32: ("<f", 4), BOOL: ("<?", 1),
    UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8),
}

MAX_ARRAY_ELEMS = 1 << 24   # абсолютный предохранитель от битого файла
MAX_ARRAY_STORED = 8192      # длиннее — читаем и выбрасываем (словарь токенов)


class GGUFError(Exception):
    pass


_FIXED_SKIP = {
    UINT8: 1, INT8: 1, UINT16: 2, INT16: 2, UINT32: 4, INT32: 4,
    FLOAT32: 4, BOOL: 1, UINT64: 8, INT64: 8, FLOAT64: 8,
}


@dataclass
class ModelMeta:
    path: Path
    size_bytes: int = 0
    arch: str = ""
    kv: dict = field(default_factory=dict)
    # таблица тензоров: (имя, ggml-тип, offset в секции данных, число элементов).
    # Читается из шапки, сами веса не трогаем. Пусто, если в файле нет
    # тензоров или таблицу не удалось разобрать.
    tensors: list = field(default_factory=list)
    # файловый offset начала секции данных (после выравнивания)
    data_start: int = 0

    # ── производные величины ──
    @property
    def n_layer(self) -> int:
        return int(self.kv.get(f"{self.arch}.block_count", 0) or 0)

    @property
    def n_embd(self) -> int:
        return int(self.kv.get(f"{self.arch}.embedding_length", 0) or 0)

    @property
    def n_head(self) -> int:
        return int(self.kv.get(f"{self.arch}.attention.head_count", 0) or 0)

    @property
    def n_head_kv(self) -> int:
        return int(self.kv.get(f"{self.arch}.attention.head_count_kv", 0) or 0)

    @property
    def head_dim(self) -> int:
        """Размерность одной головы внимания.

        Не равна n_embd/n_head, если у модели есть qk head-dim != v head-dim
        (например 192/128 у Qwen3). Тогда приоритет у attention.key_length.
        """
        kd = int(self.kv.get(f"{self.arch}.attention.key_length", 0) or 0)
        if kd:
            return kd
        if self.n_embd and self.n_head:
            return self.n_embd // self.n_head
        return 0

    @property
    def v_head_dim(self) -> int:
        vd = int(self.kv.get(f"{self.arch}.attention.value_length", 0) or 0)
        return vd or self.head_dim

    @property
    def n_ctx_trained(self) -> int:
        return int(self.kv.get(f"{self.arch}.context_length", 0) or 0)

    @property
    def n_expert(self) -> int:
        return int(self.kv.get(f"{self.arch}.expert_count", 0) or 0)

    @property
    def n_expert_used(self) -> int:
        return int(self.kv.get("llama.expert_used", 0) or 0)

    @property
    def n_expert_shared(self) -> int:
        return int(self.kv.get("llama.expert_shared_count", 0) or 0)

    @property
    def is_moe(self) -> bool:
        return self.n_expert > 0

    @property
    def n_params(self) -> int:
        return int(self.kv.get("general.parameter_count", 0) or 0)

    @property
    def quant(self) -> str:
        return str(self.kv.get("general.file_type", ""))

    @property
    def name(self) -> str:
        return str(self.kv.get("general.name", self.path.stem))

    @property
    def n_vocab(self) -> int:
        return int(self.kv.get("llama.vocab_size", 0) or 0)

    # ── MLA и MTP ──
    @property
    def kv_lora_rank(self) -> int:
        """Ранг сжатия KV у MLA-моделей (0 — обычное GQA/MHA внимание)."""
        return int(self.kv.get(f"{self.arch}.attention.kv_lora_rank", 0) or 0)

    @property
    def rope_dim(self) -> int:
        """Размерность RoPE-части MLA-ключа (k_pe)."""
        return int(self.kv.get(f"{self.arch}.rope.dimension_count", 0) or 0)

    @property
    def is_mla(self) -> bool:
        """Сжатый MLA-KV: одна строка (latent + rope) вместо голов K/V."""
        return self.kv_lora_rank > 0 and self.rope_dim > 0

    @property
    def n_layer_nextn(self) -> int:
        """Число MTP-слоёв в хвосте (0 — нет встроенной MTP-головы)."""
        return int(self.kv.get(f"{self.arch}.nextn_predict_layers", 0) or 0)

    def tensor_sizes(self) -> list[tuple[str, int]]:
        """(имя, байты) для каждого тензора.

        Размер — через разность offsets соседних тензоров, поэтому включает
        выравнивающие промежутки. Это честно для VRAM: llama.cpp тоже
        кладёт тензоры с выравниванием.
        """
        out = []
        n = len(self.tensors)
        for i, (name, _dtype, off, _nelem) in enumerate(self.tensors):
            if i + 1 < n:
                size = self.tensors[i + 1][2] - off
            else:
                size = self.size_bytes - self.data_start - off
            out.append((name, max(0, size)))
        return out

    def summary(self) -> str:
        bits = [f"arch={self.arch or '?'}", f"layers={self.n_layer}",
                f"embd={self.n_embd}", f"head={self.n_head}/{self.n_head_kv}",
                f"hd={self.head_dim}", f"ctx_trained={self.n_ctx_trained}"]
        if self.is_moe:
            bits.append(f"experts={self.n_expert_used}/{self.n_expert}")
        if self.n_params:
            bits.append(f"params={self.n_params / 1e9:.2f}B")
        bits.append(f"file={self.size_bytes / 2**30:.2f}GiB")
        return "  ".join(bits)


class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def read(self, n: int) -> bytes:
        data = self.fh.read(n)
        if len(data) != n:
            raise GGUFError(f"обрыв файла: ждали {n} байт, получили {len(data)}")
        return data

    def uint(self, fmt: str, size: int) -> int:
        return struct.unpack(fmt, self.read(size))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.uint("<Q", 8)
        if n > (1 << 20):
            raise GGUFError(f"нереальная длина строки: {n}")
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, vtype: int, depth: int = 0):
        if vtype in _SCALARS:
            fmt, size = _SCALARS[vtype]
            return self.uint(fmt, size)
        if vtype == STRING:
            return self.string()
        if vtype == ARRAY:
            if depth > 4:
                raise GGUFError("слишком глубокая вложенность массива")
            etype = self.uint("<I", 4)
            n = self.uint("<Q", 8)
            if n > MAX_ARRAY_ELEMS:
                raise GGUFError(f"подозрительно длинный массив: {n}")
            if n > MAX_ARRAY_STORED:
                # длинные массивы (словарь, merges) нам не нужны.
                # Скаляры фиксированной длины пропускаем seek'ом вместо
                # тысяч мелких read() — меньше syscall на больших словарях.
                if etype in _FIXED_SKIP:
                    self.fh.seek(n * _FIXED_SKIP[etype], 1)
                    return f"<array of {n} elems, elided>"
                for _ in range(n):
                    self.value(etype, depth + 1)
                return f"<array of {n} elems, elided>"
            return [self.value(etype, depth + 1) for _ in range(n)]
        raise GGUFError(f"неизвестный тип GGUF: {vtype}")


def read_meta(path: str | Path) -> ModelMeta:
    """Читает метаданные GGUF. Бросает GGUFError, если это не GGUF."""
    p = Path(path).expanduser()
    st = p.stat()
    meta = ModelMeta(path=p, size_bytes=st.st_size)
    with p.open("rb") as fh:
        r = _Reader(fh)
        if r.read(4) != b"GGUF":
            raise GGUFError("нет магии GGUF — это не GGUF-файл")
        version = r.uint("<I", 4)
        if version not in (1, 2, 3):
            raise GGUFError(f"версия GGUF v{version} не поддерживается")
        n_tensors = r.uint("<Q", 8)
        n_kv = r.uint("<Q", 8)
        if n_kv > 1 << 20:
            raise GGUFError(f"подозрительное число KV: {n_kv}")
        for _ in range(n_kv):
            key = r.string()
            vtype = r.uint("<I", 4)
            meta.kv[key] = r.value(vtype)
        meta.arch = str(meta.kv.get("general.architecture", ""))
        # таблица тензоров идёт сразу за KV. Битый хвост не должен ронять
        # метаданные: не разобралось — останется пустой список.
        try:
            _read_tensor_infos(fh, r, meta, n_tensors)
        except (GGUFError, struct.error, OSError, ValueError):
            meta.tensors = []
            meta.data_start = 0
    return meta


def _read_tensor_infos(fh, r: _Reader, meta: ModelMeta, n_tensors: int) -> None:
    """Читает (имя, тип, offset, nelem) всех тензоров. Веса не трогает."""
    if n_tensors > 1 << 20:
        raise GGUFError(f"подозрительное число тензоров: {n_tensors}")
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.u32()
        if n_dims > 8:
            raise GGUFError(f"подозрительная размерность тензора {name!r}: {n_dims}")
        nelem = 1
        for _ in range(n_dims):
            nelem *= r.u64()
        dtype = r.u32()
        offset = r.u64()
        infos.append((name, dtype, offset, nelem))
    align = meta.kv.get("general.alignment", 32) or 32
    try:
        align = int(align)
    except (TypeError, ValueError):
        align = 32
    if align < 1:
        align = 32
    pos = fh.tell()
    meta.tensors = infos
    meta.data_start = pos + (-pos % align)


def probe(path: str | Path) -> ModelMeta:
    """read_meta, но с человекочитаемой ошибкой вместо исключения."""
    try:
        return read_meta(path)
    except (GGUFError, OSError, struct.error) as exc:
        raise GGUFError(f"{path}: {exc}") from exc


def is_gguf(path: str | Path) -> bool:
    try:
        with Path(path).open("rb") as fh:
            return fh.read(4) == b"GGUF"
    except OSError:
        return False


def find_mmproj_near(model: str | Path, max_scanned: int = 400) -> str | None:
    """Ищет mmproj-проектор рядом с моделью (в том числе в snapshots/ HF-кэша)."""
    m = Path(model).expanduser()
    cands = []
    for d in (m.parent, m.parent.parent, m.parent.parent.parent):
        if not d.is_dir():
            continue
        try:
            entries = sorted(d.glob("**/mmproj*.gguf"))
        except OSError:
            continue
        cands.extend(entries[:max_scanned])
        if cands:
            break
    if not cands:
        return None
    # ближайший по общему префиксу имени модели выигрывает
    stem = m.stem.split("-G")[0].lower()
    best, best_score = None, -1
    for c in cands:
        cstem = c.stem.lower()
        score = sum(1 for tok in stem.split("-") if tok and tok in cstem)
        if score > best_score:
            best, best_score = c, score
    return str(best)
