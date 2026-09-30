import { useState } from "react"

import { Card, CardContent } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import data from "@/data/case.json"
import { ImpactChart, MarkLegend, TimelineChart } from "./charts"
import { categoryLabel, num, pct } from "./format"
import { Section } from "./layout"

const ORDER = ["E1", "E2", "E5", "E4", "E3"]

function delay(from: number, to: number | null) {
  return to === null ? "não avisou" : `+${to - from} dias`
}

export function AlarmSection() {
  const [selected, setSelected] = useState("E1")
  const event = data.events.find((e) => e.id === selected)!
  const marks = { start: event.start, aspectAlarm: event.aspect_alarm, starAlarm: event.star_alarm }
  const val = data.alarm_validation

  return (
    <Section
      id="alarme"
      eyebrow="Alarme antecipado"
      title="O alarme por aspecto avisa antes da nota média"
      lead={
        <>
          Cinco problemas foram plantados nos dados, com dia de início conhecido. O alarme vigia cada tipo de reclamação
          por loja; o jeito comum vigia só a nota média, que oscila tanto que uma queda real demora a se destacar.
        </>
      }
    >
      <Tabs value={selected} onValueChange={setSelected} className="mb-6">
        <TabsList className="flex h-auto flex-wrap">
          {ORDER.map((id) => {
            const e = data.events.find((x) => x.id === id)!
            return (
              <TabsTrigger key={id} value={id} className="cursor-pointer">
                {e.description}
              </TabsTrigger>
            )
          })}
        </TabsList>
      </Tabs>

      <div className="mb-6 grid gap-4 sm:grid-cols-3">
        <Card className="py-0"><CardContent className="p-4"><p className="text-sm text-muted-foreground">Onde e quando começou</p><p className="text-xl font-semibold">{event.store}, dia {event.start}</p></CardContent></Card>
        <Card className="py-0"><CardContent className="p-4"><p className="text-sm text-muted-foreground">Alarme por aspecto</p><p className="text-xl font-semibold">{delay(event.start, event.aspect_alarm)}</p></CardContent></Card>
        <Card className="py-0"><CardContent className="p-4"><p className="text-sm text-muted-foreground">Alarme pela nota média</p><p className="text-xl font-semibold">{delay(event.start, event.star_alarm)}</p></CardContent></Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <p className="mb-1 font-medium">Reviews com “{categoryLabel(`${event.aspect}|${event.polarity}`)}”</p>
          <p className="mb-2 text-sm text-muted-foreground">% das reviews, média de 7 dias</p>
          <TimelineChart data={event.series} metric="share" marks={marks} />
        </div>
        <div>
          <p className="mb-1 font-medium">Nota média</p>
          <p className="mb-2 text-sm text-muted-foreground">estrelas, média de 7 dias</p>
          <TimelineChart data={event.series} metric="stars" marks={marks} />
        </div>
      </div>
      <div className="mt-3">
        <MarkLegend />
      </div>

      <p className="mt-6 text-sm text-muted-foreground">
        {selected === "E3"
          ? "O prato novo (um evento positivo) escapou do alarme e da nota média, mas foi achado pelos embeddings (veja Temas). "
          : ""}
        A primeira versão do alarme deu 38 alarmes falsos em 112 séries. A versão atual foi validada num conjunto novo, que
        ela nunca viu: achou {Object.keys(val.detections).length} de 5 eventos, com {num(val.false_per_1000_series_days, 1)}{" "}
        alarmes falsos por 1.000 séries-dia (cerca de 1 por loja por mês).
      </p>
    </Section>
  )
}

export function ThemesSection() {
  const d = data.dishes
  const clusters = [...data.taxonomy].sort((a, b) => b.size - a.size)
  return (
    <Section
      id="temas"
      eyebrow="Temas descobertos"
      title="Os embeddings acharam sozinhos um evento que ninguém listou"
      lead={
        <>
          Os {d.snippets_total.toLocaleString("pt-BR")} trechos viraram pontos num mapa de significado e um algoritmo de
          agrupamento (HDBSCAN) achou os bairros, sem saber quantos existiam.
        </>
      }
    >
      <div className="mb-8 grid gap-4 md:grid-cols-2">
        <Card className="py-0">
          <CardContent className="p-6">
            <p className="mb-2 text-sm text-muted-foreground">Grupo “Risoto de camarão”</p>
            <p className="mb-2 text-3xl font-bold">{d.risotto_snippets} trechos</p>
            <p className="text-muted-foreground">
              {pct(d.risotto_share_store2)} da Loja 2 e {pct(d.risotto_share_after_day25)} a partir do dia 25: exatamente o
              prato novo plantado. A lista fixa de aspectos diria só “comida, elogio”.
            </p>
          </CardContent>
        </Card>
        <Card className="py-0">
          <CardContent className="p-6">
            <p className="mb-2 text-sm text-muted-foreground">Descobrir não é detectar</p>
            <p className="mb-2 text-3xl font-bold">{d.burger_complaints} reclamações</p>
            <p className="text-muted-foreground">
              do hambúrguer se perdiam ao detectar pelo centro do grupo, puxado pelos {d.burger_praise} elogios. Detectar
              pelo nome do prato pegou todas: a ferramenta mais simples ganhou.
            </p>
          </CardContent>
        </Card>
      </div>
      <p className="mb-2 text-sm text-muted-foreground">Os 11 temas, com nomes sugeridos pelo LLM e revisados à mão:</p>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Tema</TableHead>
            <TableHead className="text-right">Trechos</TableHead>
            <TableHead className="hidden sm:table-cell">Polaridade</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {clusters.map((c) => (
            <TableRow key={c.cluster}>
              <TableCell>{c.name}</TableCell>
              <TableCell className="text-right tabular-nums">{c.size.toLocaleString("pt-BR")}</TableCell>
              <TableCell className="hidden text-muted-foreground sm:table-cell">{c.polarity}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Section>
  )
}

export function ImpactSection() {
  return (
    <Section
      id="impacto"
      eyebrow="Impacto em estrelas"
      title="Quanto cada reclamação custa na nota"
      lead={
        <>
          Regressão linear da nota nas menções extraídas. A barra é o que o sistema mediu, a linha fina é o intervalo de 95%
          e o traço é o efeito que plantamos. A nota inteira de 1 a 5 “achata” um pouco os efeitos. É associação, não causa.
        </>
      }
    >
      <div className="mb-3 flex gap-5 text-sm text-muted-foreground">
        <span className="flex items-center gap-2"><span className="inline-block h-3 w-3 rounded-sm" style={{ background: "var(--viz-series)" }} />elogio</span>
        <span className="flex items-center gap-2"><span className="inline-block h-3 w-3 rounded-sm" style={{ background: "var(--viz-neg)" }} />reclamação</span>
        <span className="flex items-center gap-2"><span className="inline-block h-3 w-0.5" style={{ background: "var(--viz-ink)" }} />efeito plantado</span>
      </div>
      <ImpactChart rows={data.impact} />
    </Section>
  )
}
