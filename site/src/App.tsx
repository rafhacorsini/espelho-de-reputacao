import { ThemeProvider } from "@/components/theme-provider"
import { CasePage } from "@/case/case-page"

function App() {
  return (
    <div className="font-sans antialiased" style={{ fontFamily: "var(--font-inter)" }}>
      <ThemeProvider defaultTheme="system" storageKey="espelho-theme">
        <CasePage />
      </ThemeProvider>
    </div>
  )
}

export default App
