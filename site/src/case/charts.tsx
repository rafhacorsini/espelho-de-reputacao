import {
  Bar,
  CartesianGrid,
  Cell,
  ComposedChart,
  ErrorBar,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import { useEffect, useState } from "react"

import { categoryLabel, num, pct } from "./format"

function useIsNarrow(maxWidth = 640) {
  const query = `(max-width: ${maxWidth}px)`
  const [narrow, setNarrow] = useState(() => window.matchMedia(query).matches)
  useEffect(() => {
    const media = window.matchMedia(query)
    const update = () => setNarrow(media.matches)
    media.addEventListener("change", update)
    return () => media.removeEventListener("change", update)
  }, [query])
  return narrow
}

// Cores e tinta vêm das variáveis --viz-* (index.css), que trocam sozinhas no modo escuro.
const AXIS = { stroke: "var(--viz-axis)", tick: { fill: "var(--viz-muted)", fontSize: 12 } }
const TOOLTIP = {
  contentStyle: {
    background: "var(--popover)",
    color: "var(--popover-foreground)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    fontSize: 13,
  },
  labelStyle: { color: "var(--viz-ink-2)" },
}

type Point = { day: number; share: number; stars: number }

type Marks = { start: number; aspectAlarm: number | null; starAlarm: number | null }

// As linhas não levam texto dentro do gráfico: com eventos a poucos dias um do outro, os rótulos
// se atropelavam. Quem nomeia cada linha é a legenda (MarkLegend), logo abaixo.
function markLines({ start, aspectAlarm, starAlarm }: Marks) {
  return [
    <ReferenceLine key="start" x={start} stroke="var(--viz-ink-2)" strokeDasharray="4 4" />,
    aspectAlarm !== null && <ReferenceLine key="aspect" x={aspectAlarm} stroke="var(--viz-ink)" strokeWidth={1.5} />,
    starAlarm !== null && <ReferenceLine key="stars" x={starAlarm} stroke="var(--viz-muted)" strokeWidth={1.5} />,
  ]
}

export function MarkLegend() {
  const items = [
    { label: "início do problema", style: { borderTop: "1.5px dashed var(--viz-ink-2)" } },
    { label: "alarme por aspecto", style: { borderTop: "1.5px solid var(--viz-ink)" } },
    { label: "alarme pela nota média", style: { borderTop: "1.5px solid var(--viz-muted)" } },
  ]
  return (
    <div className="flex flex-wrap gap-x-5 gap-y-1 text-sm text-muted-foreground">
      {items.map((i) => (
        <span key={i.label} className="flex items-center gap-2">
          <span className="inline-block w-5" style={i.style} />
          {i.label}
        </span>
      ))}
    </div>
  )
}

// Marcas redondas no eixo: 0%, 5%, 10%... em vez de dividir a altura em partes iguais.
function shareTicks(data: Point[]) {
  const top = Math.ceil(Math.max(...data.map((d) => d.share)) / 0.05) * 0.05
  return Array.from({ length: Math.round(top / 0.05) + 1 }, (_, i) => i * 0.05)
}

export function TimelineChart({ data, metric, marks }: { data: Point[]; metric: "share" | "stars"; marks: Marks }) {
  const isShare = metric === "share"
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 18, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="var(--viz-grid)" vertical={false} />
        <XAxis dataKey="day" type="number" domain={[0, data.length - 1]} tickCount={7} {...AXIS} />
        <YAxis
          width={48}
          domain={isShare ? [0, "dataMax"] : ["auto", "auto"]}
          ticks={isShare ? shareTicks(data) : undefined}
          tickFormatter={(v: number) => (isShare ? pct(v) : num(v, 1))}
          {...AXIS}
        />
        <Tooltip
          {...TOOLTIP}
          labelFormatter={(d) => `dia ${d}`}
          formatter={(v) => [isShare ? pct(Number(v), 1) : num(Number(v), 2), isShare ? "das reviews" : "nota média"]}
        />
        {markLines(marks)}
        <Line dataKey={metric} stroke="var(--viz-series)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
      </LineChart>
    </ResponsiveContainer>
  )
}

type ImpactRow = { category: string; plantado: number; pipeline: number; pipeline_low: number; pipeline_high: number }

export function ImpactChart({ rows }: { rows: ImpactRow[] }) {
  const narrow = useIsNarrow()
  const data = [...rows]
    .sort((a, b) => a.pipeline - b.pipeline)
    .map((r) => ({
      label: categoryLabel(r.category),
      value: r.pipeline,
      planted: r.plantado,
      // ErrorBar recebe as distâncias até as pontas do intervalo, não as pontas.
      error: [r.pipeline - r.pipeline_low, r.pipeline_high - r.pipeline],
      low: r.pipeline_low,
      high: r.pipeline_high,
    }))

  return (
    <ResponsiveContainer width="100%" height={data.length * (narrow ? 38 : 30) + 40}>
      <ComposedChart layout="vertical" data={data} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
        <CartesianGrid stroke="var(--viz-grid)" horizontal={false} />
        <XAxis type="number" domain={[-1.75, 0.75]} ticks={[-1.5, -1, -0.5, 0, 0.5]}
          tickFormatter={(v: number) => (v > 0 ? "+" : "") + num(v, 1)} {...AXIS} />
        <YAxis type="category" dataKey="label" width={narrow ? 118 : 230} interval={0}
          {...AXIS} tick={{ ...AXIS.tick, fontSize: narrow ? 11 : 12 }} />
        <ReferenceLine x={0} stroke="var(--viz-axis)" />
        <Tooltip
          {...TOOLTIP}
          cursor={{ fill: "var(--muted)", opacity: 0.4 }}
          formatter={(_v, _n, item) => {
            const p = item.payload as (typeof data)[number]
            return [`${num(p.value)} (95%: ${num(p.low)} a ${num(p.high)}) · plantado ${num(p.planted)}`, "estrelas"]
          }}
        />
        <Bar dataKey="value" barSize={12} radius={2} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.label} fill={d.value < 0 ? "var(--viz-neg)" : "var(--viz-series)"} />
          ))}
          <ErrorBar dataKey="error" direction="x" width={6} stroke="var(--viz-ink-2)" strokeWidth={1.5} />
        </Bar>
        <Scatter
          dataKey="planted"
          isAnimationActive={false}
          shape={(props: { cx?: number; cy?: number }) => (
            <line x1={props.cx} x2={props.cx} y1={(props.cy ?? 0) - 8} y2={(props.cy ?? 0) + 8} stroke="var(--viz-ink)" strokeWidth={2} />
          )}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
