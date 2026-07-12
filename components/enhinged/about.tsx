"use client"

import { useRef } from "react"
import { motion, useScroll, useTransform, useSpring } from "framer-motion"

const statements = [
  "Building products that think.",
  "Custom architectures for nuanced language.",
  "Hinglish + English, both done right.",
  "Fine-tuned, not from scratch.",
  "RLHF for genuine quality improvement.",
  "KV-cache. Faster every step.",
  "Premium interfaces for real models.",
]

export function About() {
  const containerRef = useRef<HTMLElement>(null)
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"],
  })

  const x = useTransform(scrollYProgress, [0, 1], ["0%", "-100%"])
  const smoothX = useSpring(x, { stiffness: 100, damping: 30 })

  return (
    <section id="about" ref={containerRef} className="relative py-32 overflow-hidden bg-background">
      {/* Section Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8 }}
        className="px-8 md:px-12 mb-0 py-20 max-w-7xl mx-auto"
      >
        <p className="font-mono text-[10px] tracking-[0.3em] text-muted-foreground uppercase mb-4">08 — About V2</p>
        <h2 className="font-sans text-3xl md:text-5xl font-light italic">What&apos;s New in V2</h2>
        <div className="mt-8 grid md:grid-cols-2 gap-6 max-w-3xl">
          {[
            { title: "Fixed data pipeline", desc: "Boundary-aligned sampling eliminates the cross-pair gradient leakage that caused disjointed V1 responses." },
            { title: "Bilingual fluency", desc: "English conversational data added alongside Hinglish, with controllable per-batch language mixing." },
            { title: "RLHF quality pass", desc: "PPO fine-tuning with AI-judged preference data improves conversational naturalness and relevance." },
            { title: "Faster generation", desc: "KV-caching reduces generation cost from quadratic to linear, delivering a 2x+ speedup on CPU hosting." },
          ].map((item) => (
            <div key={item.title} className="border border-border/50 rounded-xl p-5">
              <h3 className="font-medium mb-2">{item.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Horizontal Scroll Container */}
      <div className="relative flex items-center overflow-hidden py-0 gap-0 h-24">
        <motion.div style={{ x: smoothX }} className="flex gap-16 md:gap-24 px-8 md:px-12 whitespace-nowrap">
          {statements.map((statement, index) => (
            <motion.p
              key={index}
              className="text-4xl md:text-6xl lg:text-7xl font-sans font-light tracking-tight text-foreground/90"
              style={{
                WebkitTextStroke: index % 2 === 0 ? "none" : "1px var(--border)",
                color: index % 2 === 0 ? "inherit" : "transparent",
              }}
            >
              {statement}
            </motion.p>
          ))}
        </motion.div>
      </div>

      {/* Decorative Line */}
      <motion.div
        initial={{ scaleX: 0 }}
        whileInView={{ scaleX: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 1.5, ease: [0.25, 0.46, 0.45, 0.94] }}
        className="mt-24 max-w-7xl mx-auto h-px bg-gradient-to-r from-transparent via-border to-transparent origin-left"
      />
    </section>
  )
}
