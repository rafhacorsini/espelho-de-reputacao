"""Definições compartilhadas: os 8 aspectos, as polaridades e os canais.

Estes aspectos são a "lista fixa" do sistema. No dia 4 descobrimos categorias
direto dos dados e comparamos com esta lista.
"""

ASPECTS = {
    "atendimento": "Educação, atenção e agilidade da equipe no salão (garçom, caixa, gerente).",
    "tempo_espera_salao": "Fila para sentar ou demora da cozinha e do prato até a mesa.",
    "prazo_entrega": "Tempo total até o pedido chegar por delivery.",
    "condicao_entrega": "Estado do pedido entregue: embalagem, temperatura, item errado ou faltando.",
    "qualidade_comida": "Sabor, ponto, frescor e porção da comida.",
    "preco_valor": "Preço e se vale o que custa.",
    "ambiente_limpeza": "Limpeza, barulho, conforto e clima do lugar.",
    "resposta_canal": "Rapidez e qualidade das respostas do restaurante por WhatsApp, telefone ou app.",
}

POLARITIES = ("positivo", "negativo")

CHANNELS = ("salao", "delivery")

# Quais aspectos podem aparecer em cada canal.
ASPECTS_BY_CHANNEL = {
    "salao": [
        "atendimento",
        "tempo_espera_salao",
        "qualidade_comida",
        "preco_valor",
        "ambiente_limpeza",
        "resposta_canal",
    ],
    "delivery": [
        "prazo_entrega",
        "condicao_entrega",
        "qualidade_comida",
        "preco_valor",
        "resposta_canal",
    ],
}
