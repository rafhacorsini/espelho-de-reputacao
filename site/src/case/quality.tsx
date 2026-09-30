import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import data from "@/data/case.json"
import { num, pct, usd } from "./format"
import { Section } from "./layout"

export function EvaluationSection() {
  return (
    <Section
      id="avaliacao"
      eyebrow="Avaliação"
      title="Números protegidos contra o próprio autor"
      lead={
        <>
          150 reviews rotuladas à mão, divididas em dev (para ajustar o prompt) e test (aberto uma única vez). O gabarito do
          test foi commitado no git antes de o test ser aberto, e rotulado duas vezes às cegas (κ {num(data.kappa)}).
        </>
      }
    >
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Sistema, no test cego ({data.test_reviews} reviews)</TableHead>
            <TableHead className="text-right">F1</TableHead>
            <TableHead className="text-right">Intervalo de 95%</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.evaluation_test.map((e) => (
            <TableRow key={e.system}>
              <TableCell>{e.system}</TableCell>
              <TableCell className="text-right font-semibold tabular-nums">{num(e.f1[0])}</TableCell>
              <TableCell className="text-right tabular-nums text-muted-foreground">
                {num(e.f1[1])} a {num(e.f1[2])}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <p className="mt-4 text-sm text-muted-foreground">
        O v2 (oficial) não superou o v1 no test: a diferença cabe no intervalo. Ele continua oficial porque foi escolhido antes
        de abrir o test; trocar agora seria escolher olhando a prova.
      </p>
    </Section>
  )
}

export function CostSection() {
  return (
    <Section
      id="custo"
      eyebrow="Custo"
      title="Um modelo pequeno resolve o fácil, o LLM resolve o difícil"
      lead="Um modelo pequeno (regressão logística sobre embeddings) aprendeu com as respostas do LLM. Na cascata, ele responde quando está seguro e manda o resto ao LLM."
    >
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Sistema</TableHead>
            <TableHead className="text-right">F1 no test</TableHead>
            <TableHead className="text-right">Custo por 1.000 reviews</TableHead>
            <TableHead className="text-right">Vai ao LLM</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.cascade.map((c) => (
            <TableRow key={c.system}>
              <TableCell>{c.system}</TableCell>
              <TableCell className="text-right tabular-nums">{num(c.f1_test)}</TableCell>
              <TableCell className="text-right tabular-nums">{usd(c.cost_per_1000, 3)}</TableCell>
              <TableCell className="text-right tabular-nums">{pct(c.share_llm)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <p className="mt-4 text-sm text-muted-foreground">
        A cascata é 74% mais barata por 3 pontos de F1. Nesta escala a economia em dólares é irrelevante; ela importa com
        milhões de reviews ou quando a latência pesa.
      </p>
    </Section>
  )
}

const ATTACKS = 40

export function SecuritySection() {
  const sec = data.security
  const configs = Object.entries(sec.configs)
  return (
    <Section
      id="seguranca"
      eyebrow="Segurança"
      title="Atacamos o próprio sistema com injeção de prompt"
      lead={`Review é texto de terceiros que entra no prompt. Foram ${ATTACKS} ataques escritos à mão (10 estilos) e 60 reviews normais de controle. Ruído de base: ${pct(sec.noise, 1)} das reviews mudam de saída rodando duas vezes, sem ataque.`}
    >
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Configuração</TableHead>
            <TableHead className="text-right">Ataques que mudaram a saída</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {configs.map(([name, v]) => (
            <TableRow key={name}>
              <TableCell>{name}</TableCell>
              <TableCell className="text-right tabular-nums">
                {Math.round(v.asr * ATTACKS)} de {ATTACKS}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <p className="mt-4 text-sm text-muted-foreground">
        Só 1 ataque funcionou de verdade sem defesa (uma ordem escondida no meio da review virou uma reclamação em elogio); as
        2 mudanças com a review delimitada não seguiam o objetivo do ataque, são o ruído de base. O guarda marcou{" "}
        {pct(sec.guard.attacks_flagged)} dos ataques e bloqueou por engano {sec.guard.normal_false_blocks} de{" "}
        {sec.guard.normal_total} reviews normais e {sec.guard.tricky_false_blocks} de 5 controles difíceis (“ignoraram meu
        pedido”, “regra da casa”). Leitura honesta: ataques ingênuos quase não funcionam em modelos de 2026, e ataques
        adaptativos não foram testados. A defesa principal é a contenção: o extrator não tem ferramentas nem ações.
      </p>
    </Section>
  )
}

const MISTAKES = [
  {
    title: "Um F1 de 1,00 que não valia nada",
    text: "Depois de corrigir o gabarito olhando os erros do modelo e escrever regras no mesmo dev, o v2 fez 1,00 no dev. No test, não superou o v1. Era viés de adjudicação e ajuste ao dev, e isso só aparece porque o test ficou fechado.",
  },
  {
    title: "Alarme v1: 38 alarmes falsos em 112 séries",
    text: "A aproximação normal não serve para contagens raras: uma única menção parecia um terremoto. E o alarme parava de vigiar depois do primeiro toque. A v2 usa a razão de verossimilhança binomial, reinicia depois de cada alarme e foi validada num conjunto que nunca viu.",
  },
  {
    title: "Embeddings agrupam por assunto, não por sentimento",
    text: "“Preço caro” e “preço justo” caíram no mesmo grupo, e a demora na entrega e a espera no salão também: o texto é igual, quem separa é o canal. Por isso o agrupamento acha o assunto e o extrator dá a polaridade.",
  },
  {
    title: "A análise de erros estava contada errado",
    text: "O relato inicial dizia 21 menções inventadas; recontando a lista, eram 19 (10 erros do gabarito, 6 ambiguidades de regra, 3 erros do modelo). A correção está no histórico do repositório.",
  },
  {
    title: "Os dados são sintéticos",
    text: "Texto gerado por IA é mais limpo que o real, e LLMs acertam ironia sintética muito mais do que ironia humana. Os números aqui são um teto otimista; a validação com reviews reais, copiadas à mão e rotuladas antes do modelo, está em andamento.",
  },
]

export function MistakesSection() {
  return (
    <Section id="erros" eyebrow="O que deu errado" title="E o que cada erro ensinou">
      <Accordion type="single" collapsible className="w-full">
        {MISTAKES.map((m) => (
          <AccordionItem key={m.title} value={m.title}>
            <AccordionTrigger className="cursor-pointer text-left">{m.title}</AccordionTrigger>
            <AccordionContent className="text-muted-foreground">{m.text}</AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </Section>
  )
}
