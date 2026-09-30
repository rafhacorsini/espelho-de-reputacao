import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("import_real", Path("scripts/import_real_reviews.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_parse_valid_and_invalid_lines():
    text = "\n".join([
        "# comentário",
        "== Restaurante A ==",
        "2 | delivery | Chegou frio e atrasado.",
        "5 | salão | Lasanha perfeita!",
        "4 | ? | Gostei bastante.",
        "7 | salao | nota impossível",
        "sem separador",
    ])
    rows, problems = module.parse(text)
    assert [r["channel"] for r in rows] == ["delivery", "salao", None]
    assert rows[0]["restaurant"] == "A"
    assert len(problems) == 2
