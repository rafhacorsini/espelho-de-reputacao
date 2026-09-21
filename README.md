# Espelho de Reputação

Case study de Engenharia de IA: transforma reviews de restaurantes em dores e forças
mensuráveis, detecta problemas novos e estima quanto cada dor custa em estrelas.

Status: em construção (semana 1). O plano e o diário de estudo ficam em um documento à parte.

## Como rodar

```bash
uv sync
copy .env.example .env   # depois cole sua chave em .env
uv run python scripts/smoke_test.py
uv run pytest
```

Nunca versione o arquivo `.env`.
