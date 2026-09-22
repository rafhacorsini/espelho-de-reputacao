"""Armazenamento do conjunto ouro: os rótulos que você atribui à mão.

Formato: um arquivo JSONL, uma linha por review rotulada (a última linha de
cada review_id vence, então recarregar um valor existente é seguro).
"""

import json
from pathlib import Path

TARGET_LABELED = 150


def load_labels(path: Path) -> dict:
    """Lê o arquivo de rótulos e devolve {review_id: registro}, a última entrada vence."""
    if not path.exists():
        return {}
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            record = json.loads(line)
            labels[record["review_id"]] = record
    return labels


def save_label(path: Path, review_id: str, mentions: list[dict], discarded: bool = False) -> None:
    """Acrescenta um rótulo ao arquivo. Não reescreve o arquivo inteiro."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"review_id": review_id, "mentions": mentions, "discarded": discarded}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def progress(labels: dict, target: int = TARGET_LABELED) -> dict:
    """Quantas reviews foram rotuladas (sem contar descarte) e quantas faltam."""
    kept = [r for r in labels.values() if not r.get("discarded")]
    return {
        "labeled": len(kept),
        "discarded": len(labels) - len(kept),
        "target": target,
        "remaining": max(0, target - len(kept)),
        "done": len(kept) >= target,
    }
