import { HowItWorks, Hero, Stats } from "./intro"
import { Footer, Navbar } from "./layout"
import { CostSection, EvaluationSection, MistakesSection, SecuritySection } from "./quality"
import { AlarmSection, ImpactSection, ThemesSection } from "./results"

export function CasePage() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main>
        <Hero />
        <Stats />
        <HowItWorks />
        <AlarmSection />
        <ThemesSection />
        <ImpactSection />
        <EvaluationSection />
        <CostSection />
        <SecuritySection />
        <MistakesSection />
      </main>
      <Footer />
    </div>
  )
}
