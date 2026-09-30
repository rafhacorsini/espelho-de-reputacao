import { ArrowDown, BellRing, Braces, Github, Scale, ShieldCheck, Sparkles, Target } from "lucide-react"

import { DotPattern } from "@/components/dot-pattern"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import data from "@/data/case.json"
import { num, REPO_URL, usd } from "./format"

const official = data.evaluation_test[0].f1
const baseline = data.evaluation_test[data.evaluation_test.length - 1].f1
const totalCost = data.costs.reduce((sum, c) => sum + c.usd, 0)

export function Hero() {
  return (
    <section id="topo" className="relative overflow-hidden pb-12 pt-16 sm:pt-24">
      <div className="absolute inset-0">
        <DotPattern className="opacity-100" size="md" fadeStyle="ellipse" />
      </div>
      <div className="container relative mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-4xl text-center">
          <Badge variant="outline" className="mb-8 px-4 py-2">
            Case de Engenharia de IA · 7 dias · {usd(totalCost)} de API
          </Badge>
          <h1 className="mb-6 text-4xl font-bold tracking-tight sm:text-6xl">
            Avisar o restaurante
            <span className="bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent"> semanas antes </span>
            da nota cair
          </h1>
          <p className="mx-auto mb-10 max-w-2xl text-lg text-muted-foreground sm:text-xl">
            Um LLM lê cada review e extrai o assunto, o sentimento e o trecho que prova. Estatística transforma isso em
            alarme e em estrelas perdidas. Cada número abaixo foi medido contra um gabarito feito à mão.
          </p>
          <div className="flex flex-col gap-3 sm:flex-row sm:justify-center">
            <Button size="lg" asChild>
              <a href="#resultados">
                Ver os resultados
                <ArrowDown className="ml-2 h-4 w-4" />
              </a>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <a href={REPO_URL} target="_blank" rel="noreferrer">
                <Github className="mr-2 h-4 w-4" />
                Ver o código
              </a>
            </Button>
          </div>
        </div>
      </div>
    </section>
  )
}

export function Stats() {
  const stats = [
    { icon: Target, value: num(official[0]), label: "F1 no test cego", note: `contra ${num(baseline[0])} de um baseline de palavras-chave` },
    { icon: BellRing, value: "2–3 sem.", label: "de antecedência", note: "sobre a nota média, nos problemas de uma loja" },
    { icon: Scale, value: `κ ${num(data.kappa)}`, label: "concordância humana", note: "o anotador consigo mesmo, rotulando às cegas" },
    { icon: Sparkles, value: usd(totalCost), label: "de API no total", note: `para ${data.reviews.toLocaleString("pt-BR")} reviews, avaliação e ataques` },
  ]
  return (
    <section id="resultados" className="scroll-mt-20 py-8">
      <div className="container mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          {stats.map((s) => (
            <Card key={s.label} className="bg-background/60 py-0 backdrop-blur-sm">
              <CardContent className="p-5">
                <s.icon className="mb-3 h-5 w-5 text-primary" />
                <p className="text-2xl font-bold sm:text-3xl">{s.value}</p>
                <p className="font-medium">{s.label}</p>
                <p className="mt-1 text-sm text-muted-foreground">{s.note}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}

export function HowItWorks() {
  const steps = [
    { icon: Braces, title: "Extração com prova", text: "O LLM devolve aspecto, elogio ou reclamação e o trecho literal. O código confere se o trecho existe no texto: 132 de 132 bateram." },
    { icon: Sparkles, title: "Temas descobertos", text: "Embeddings e agrupamento acham assuntos que ninguém listou, como um prato novo fazendo sucesso." },
    { icon: Scale, title: "Impacto em estrelas", text: "Uma regressão estima quanto cada reclamação custa na nota, com intervalo de confiança." },
    { icon: BellRing, title: "Alarme por aspecto", text: "Um CUSUM vigia cada tipo de reclamação por loja e toca quando ela cresce de verdade." },
    { icon: ShieldCheck, title: "Defesas", text: "Review é texto de terceiros dentro do prompt: um guarda barra injeção antes da extração." },
  ]
  return (
    <section className="py-10">
      <div className="container mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {steps.map((s, i) => (
            <div key={s.title} className="rounded-xl border p-5">
              <p className="mb-3 flex items-center gap-2 text-sm text-muted-foreground">
                <s.icon className="h-4 w-4 text-primary" /> {i + 1}
              </p>
              <p className="mb-1 font-semibold">{s.title}</p>
              <p className="text-sm text-muted-foreground">{s.text}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
