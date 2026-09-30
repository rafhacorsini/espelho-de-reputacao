import type { ReactNode } from "react"
import { Github } from "lucide-react"

import { ModeToggle } from "@/components/mode-toggle"
import { Button } from "@/components/ui/button"
import { REPO_URL } from "./format"

const LINKS = [
  { href: "#alarme", label: "Alarme" },
  { href: "#temas", label: "Temas" },
  { href: "#impacto", label: "Impacto" },
  { href: "#avaliacao", label: "Avaliação" },
  { href: "#custo", label: "Custo" },
  { href: "#seguranca", label: "Segurança" },
  { href: "#erros", label: "O que deu errado" },
]

export function Navbar() {
  return (
    <header className="sticky top-0 z-50 border-b bg-background/80 backdrop-blur">
      <div className="container mx-auto flex h-14 items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <a href="#topo" className="font-semibold tracking-tight">
          Espelho de Reputação
        </a>
        <nav className="hidden items-center gap-5 text-sm text-muted-foreground lg:flex">
          {LINKS.map((l) => (
            <a key={l.href} href={l.href} className="transition-colors hover:text-foreground">
              {l.label}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" asChild>
            <a href={REPO_URL} target="_blank" rel="noreferrer">
              <Github className="mr-2 h-4 w-4" />
              Código
            </a>
          </Button>
          <ModeToggle />
        </div>
      </div>
    </header>
  )
}

export function Section({ id, eyebrow, title, lead, children }: {
  id: string
  eyebrow: string
  title: string
  lead?: ReactNode
  children: ReactNode
}) {
  return (
    <section id={id} className="scroll-mt-20 py-14 sm:py-20">
      <div className="container mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
        <p className="mb-2 text-sm font-medium uppercase tracking-wider text-primary/70">{eyebrow}</p>
        <h2 className="mb-4 text-3xl font-bold tracking-tight sm:text-4xl">{title}</h2>
        {lead && <div className="mb-8 max-w-3xl text-lg text-muted-foreground">{lead}</div>}
        {children}
      </div>
    </section>
  )
}

export function Footer() {
  return (
    <footer className="border-t py-10 text-sm text-muted-foreground">
      <div className="container mx-auto flex max-w-5xl flex-col gap-2 px-4 sm:px-6 lg:px-8">
        <p>
          Case de portfólio de Engenharia de IA. Dados sintéticos com verdade plantada; validação com reviews reais em
          andamento. Código, gabarito e resultados no{" "}
          <a className="underline underline-offset-4 hover:text-foreground" href={REPO_URL} target="_blank" rel="noreferrer">
            GitHub
          </a>
          .
        </p>
        <p>
          Interface baseada no{" "}
          <a className="underline underline-offset-4 hover:text-foreground" href="https://github.com/shadcnstore/shadcn-dashboard-landing-template" target="_blank" rel="noreferrer">
            shadcn-dashboard-landing-template
          </a>{" "}
          (licença MIT, © ShadcnStore). Esta página não usa cookies nem rastreamento.
        </p>
      </div>
    </footer>
  )
}
