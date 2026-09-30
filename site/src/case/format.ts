// Números no formato brasileiro (vírgula decimal).
export const num = (value: number, digits = 2) =>
  value.toLocaleString("pt-BR", { minimumFractionDigits: digits, maximumFractionDigits: digits })

export const pct = (value: number, digits = 0) =>
  (value * 100).toLocaleString("pt-BR", { minimumFractionDigits: digits, maximumFractionDigits: digits }) + "%"

export const usd = (value: number, digits = 2) => "US$ " + num(value, digits)

export const ASPECT_NAMES: Record<string, string> = {
  atendimento: "Atendimento",
  tempo_espera_salao: "Espera no salão",
  prazo_entrega: "Prazo de entrega",
  condicao_entrega: "Condição da entrega",
  qualidade_comida: "Comida",
  preco_valor: "Preço",
  ambiente_limpeza: "Ambiente",
  resposta_canal: "Resposta no WhatsApp/app",
}

export const categoryLabel = (category: string) => {
  const [aspect, polarity] = category.split("|")
  return `${ASPECT_NAMES[aspect] ?? aspect} · ${polarity === "positivo" ? "elogio" : "reclamação"}`
}

export const REPO_URL = "https://github.com/rafhacorsini/espelho-de-reputacao"
