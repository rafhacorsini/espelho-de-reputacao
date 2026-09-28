"""Dia 6, parte B: ataca o próprio sistema com injeção de prompt e mede as defesas.

Configurações:
- D0: extract_v2, sem defesa.
- D1: extract_v3 (review delimitada como dado).
- D2: extract_v2 + descarta evidência que não existe literalmente na review.
- D3: guarda antes + extract_v2 (o que o guarda marca vai para revisão humana).
- D4: guarda + extract_v3 + descarte de evidência (as três camadas).

Um ataque "funciona" se a saída muda em relação à mesma review sem o ataque.
Também medimos o ruído: a mesma review, sem ataque, rodada duas vezes.

Uso:
    uv run python scripts/security_eval.py
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from espelho.extract import extract_one
from espelho.security import apply_injection, deviated, drop_unverified, guard_check, leaked

MODEL = "gpt-5.6-luna"
ATTACKS = Path("data/security/attacks.json")
RAW = Path("data/security/raw_outputs.json")
RESULTS = Path("reports/security_results.json")
EXAMPLES = Path("reports/security_examples.md")
CORPUS = Path("data/synthetic/corpus_reviews.parquet")
PREDICTIONS = Path("data/predictions/corpus_extract_v2.parquet")
PILOT = Path("data/synthetic/pilot_reviews.parquet")
BASES_PER_TEMPLATE = 4
N_RANDOM_CONTROLS = 15
TRICKY_CHANNELS = ["delivery", "delivery", "delivery", "salao", "salao"]


def build_cases() -> tuple[list[dict], list[dict], list[dict]]:
    spec = json.loads(ATTACKS.read_text(encoding="utf-8"))
    corpus = pd.read_parquet(CORPUS).merge(pd.read_parquet(PREDICTIONS)[["review_id", "pred_mentions"]], on="review_id")
    corpus = corpus[~corpus["review_id"].isin(set(pd.read_parquet(PILOT)["review_id"]))]
    has_complaint = corpus["pred_mentions"].map(lambda ms: any(m["polarity"] == "negativo" for m in ms))
    pool = corpus[has_complaint & (corpus["text"].str.len() >= 60)]
    n_bases = len(spec["templates"]) * BASES_PER_TEMPLATE
    bases = pool.sample(n=n_bases, random_state=11).reset_index(drop=True)

    attacks = []
    for t_index, template in enumerate(spec["templates"]):
        for b in range(BASES_PER_TEMPLATE):
            base = bases.iloc[t_index * BASES_PER_TEMPLATE + b]
            attacks.append({
                "id": f"{template['id']}-{b}", "template": template["id"], "goal": template["goal"],
                "style": template["style"], "review_id": base["review_id"], "channel": base["channel"],
                "clean": base["text"], "attacked": apply_injection(base["text"], template["text"], template["position"]),
            })
    random_controls = corpus[~corpus["review_id"].isin(bases["review_id"])].sample(n=N_RANDOM_CONTROLS, random_state=12)
    controls = [{"id": r.review_id, "channel": r.channel, "text": r.text, "tricky": False} for r in random_controls.itertuples()]
    controls += [
        {"id": f"TRICKY-{i}", "channel": ch, "text": text, "tricky": True}
        for i, (text, ch) in enumerate(zip(spec["tricky_controls"], TRICKY_CHANNELS))
    ]
    cleans = [{"id": a["id"], "channel": a["channel"], "text": a["clean"]} for a in attacks]
    return attacks, controls, cleans


def run_all(client: OpenAI, attacks, controls, cleans) -> dict:
    jobs = []
    for a in attacks:
        jobs += [("v2_attacked", a["id"], "extract", a["attacked"], a["channel"], "extract_v2"),
                 ("v3_attacked", a["id"], "extract", a["attacked"], a["channel"], "extract_v3"),
                 ("guard_attacked", a["id"], "guard", a["attacked"], None, None)]
    for c in cleans:
        jobs += [("v2_clean", c["id"], "extract", c["text"], c["channel"], "extract_v2"),
                 ("v2_clean_rerun", c["id"], "extract", c["text"], c["channel"], "extract_v2"),
                 ("v3_clean", c["id"], "extract", c["text"], c["channel"], "extract_v3"),
                 ("guard_clean", c["id"], "guard", c["text"], None, None)]
    for c in controls:
        jobs.append(("guard_control", c["id"], "guard", c["text"], None, None))

    def run(job):
        bucket, key, kind, text, channel, version = job
        if kind == "guard":
            flagged, cost = guard_check(client, text)
            return bucket, key, {"flagged": flagged, "cost": cost}
        result = extract_one(client, MODEL, text, channel=channel, version=version)
        return bucket, key, {"mentions": result["mentions"], "cost": result["cost"]}

    raw: dict = {}
    with ThreadPoolExecutor(8) as pool:
        for bucket, key, value in pool.map(run, jobs):
            raw.setdefault(bucket, {})[key] = value
    return raw


def main() -> None:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY"), max_retries=5)
    attacks, controls, cleans = build_cases()

    if RAW.exists():
        raw = json.loads(RAW.read_text(encoding="utf-8"))
        print("Saídas reaproveitadas do disco (custo zero)")
    else:
        raw = run_all(client, attacks, controls, cleans)
        RAW.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    total_cost = sum(v["cost"] for bucket in raw.values() for v in bucket.values())

    def out(bucket, key, filter_evidence=False):
        mentions = raw[bucket][key]["mentions"]
        return drop_unverified(mentions) if filter_evidence else mentions

    configs = {
        "D0 sem defesa": lambda a: deviated(out("v2_clean", a), out("v2_attacked", a)),
        "D1 review delimitada": lambda a: deviated(out("v3_clean", a), out("v3_attacked", a)),
        "D2 descarta evidência falsa": lambda a: deviated(out("v2_clean", a, True), out("v2_attacked", a, True)),
        "D3 guarda + v2": lambda a: (not raw["guard_attacked"][a]["flagged"]) and deviated(out("v2_clean", a), out("v2_attacked", a)),
        "D4 três camadas": lambda a: (not raw["guard_attacked"][a]["flagged"])
        and deviated(out("v3_clean", a, True), out("v3_attacked", a, True)),
    }

    ids = [a["id"] for a in attacks]
    noise = np.mean([deviated(out("v2_clean", i), out("v2_clean_rerun", i)) for i in ids])
    cost_of = lambda bucket: np.mean([v["cost"] for v in raw[bucket].values()])  # noqa: E731
    per_review = {"v2": cost_of("v2_clean"), "v3": cost_of("v3_clean"), "guard": cost_of("guard_clean")}
    extra_cost = {"D0 sem defesa": 0, "D1 review delimitada": per_review["v3"] - per_review["v2"],
                  "D2 descarta evidência falsa": 0, "D3 guarda + v2": per_review["guard"],
                  "D4 três camadas": per_review["guard"] + per_review["v3"] - per_review["v2"]}

    normal_flagged = [raw["guard_clean"][c["id"]]["flagged"] for c in cleans] + [
        raw["guard_control"][c["id"]]["flagged"] for c in controls if not c["tricky"]]
    tricky_flagged = [raw["guard_control"][c["id"]]["flagged"] for c in controls if c["tricky"]]
    attacks_flagged = np.mean([raw["guard_attacked"][i]["flagged"] for i in ids])

    print(f"Ruído (mesma review, sem ataque, rodada 2 vezes): {noise:.0%} mudam\n")
    print(f"{'configuração':<30} {'ataques que funcionaram':>24} {'custo extra/review':>19}")
    summary = {}
    for name, succeeded in configs.items():
        wins = [succeeded(i) for i in ids]
        summary[name] = {"asr": float(np.mean(wins)), "extra_cost": float(extra_cost[name])}
        print(f"{name:<30} {sum(wins):>7}/{len(ids)} = {np.mean(wins):>5.0%} {'US$ ' + format(extra_cost[name], '.6f'):>19}")

    print(f"\nGuarda: marcou {attacks_flagged:.0%} dos ataques")
    print(f"        bloqueou por engano {sum(normal_flagged)}/{len(normal_flagged)} reviews normais "
          f"e {sum(tricky_flagged)}/{len(tricky_flagged)} controles \"difíceis\" (sistema, ignoraram, regra...)")
    leaks = {b: sum(leaked(raw[b][a["id"]]["mentions"]) for a in attacks if a["template"] == "T06")
             for b in ("v2_attacked", "v3_attacked")}
    print(f"Vazamento de instruções (T06): v2 {leaks['v2_attacked']}/4, v3 {leaks['v3_attacked']}/4")

    print("\nPor tipo de ataque (quantos dos 4 funcionaram):")
    print(f"{'modelo':<6} {'estilo':<22} {'D0':>3} {'D1':>3} {'D4':>3}")
    for tid in sorted({a["template"] for a in attacks}):
        group = [a for a in attacks if a["template"] == tid]
        row = [sum(configs[c](a["id"]) for a in group) for c in ("D0 sem defesa", "D1 review delimitada", "D4 três camadas")]
        print(f"{tid:<6} {group[0]['style']:<22} {row[0]:>3} {row[1]:>3} {row[2]:>3}")

    examples = ["# Exemplos de ataques que funcionaram sem defesa (D0)", ""]
    for a in [a for a in attacks if configs["D0 sem defesa"](a["id"])][:4]:
        examples += [
            f"## {a['id']} ({a['style']}, objetivo: {a['goal']})", "",
            f"> {a['attacked']}", "",
            f"- Sem o ataque: {sorted((m['aspect'], m['polarity']) for m in out('v2_clean', a['id']))}",
            f"- Com o ataque: {sorted((m['aspect'], m['polarity']) for m in out('v2_attacked', a['id']))}",
            f"- Guarda marcou: {'sim' if raw['guard_attacked'][a['id']]['flagged'] else 'não'}", "",
        ]
    EXAMPLES.write_text("\n".join(examples), encoding="utf-8")
    RESULTS.write_text(json.dumps({
        "noise": float(noise), "configs": summary, "cost_per_review": per_review,
        "guard": {"attacks_flagged": float(attacks_flagged), "normal_false_blocks": int(sum(normal_flagged)),
                  "normal_total": len(normal_flagged), "tricky_false_blocks": int(sum(tricky_flagged))},
        "leaks_T06": leaks, "total_cost": total_cost,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCusto total: US$ {total_cost:.4f} | -> {RESULTS} e {EXAMPLES}")


if __name__ == "__main__":
    main()
