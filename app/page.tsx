import { SmoothScroll } from "@/components/enhinged/smooth-scroll"
import { CustomCursor } from "@/components/enhinged/custom-cursor"
import { Navbar } from "@/components/enhinged/navbar"
import { Hero } from "@/components/enhinged/hero"
import { TechMarquee } from "@/components/enhinged/tech-marquee"
import { Chat } from "@/components/enhinged/chat"
import { SectionBlend } from "@/components/enhinged/section-blend"
import { Architecture } from "@/components/enhinged/architecture"
import { ModelOverview } from "@/components/enhinged/model-overview"
import { Metrics } from "@/components/enhinged/metrics"
import { TrainingGraphs } from "@/components/enhinged/training-graphs"
import { AttentionHeatmap } from "@/components/enhinged/attention-heatmap"
import { TechnicalDetails } from "@/components/enhinged/technical-details"
import { About } from "@/components/enhinged/about"
import { Footer } from "@/components/enhinged/footer"

export default function Home() {
  return (
    <SmoothScroll>
      <CustomCursor />
      <Navbar />
      
      <main>
        <Hero />
        <TechMarquee />
        <Chat />
        <SectionBlend />
        <Architecture />
        <SectionBlend />
        <ModelOverview />
        <SectionBlend />
        <Metrics />
        <SectionBlend />
        <TrainingGraphs />
        <SectionBlend />
        <AttentionHeatmap />
        <SectionBlend />
        <TechnicalDetails />
        <SectionBlend />
        <About />
      </main>

      <Footer />
    </SmoothScroll>
  )
}