"use client"

import { useRef } from "react"
import { motion, useScroll, useTransform, useSpring } from "framer-motion"

const technologies = [
  "TRANSFORMER",
  "HINGLISH",
  "GPT-2 BPE",
  "30.04M PARAMS",
  "ATTENTION",
  "EMBEDDINGS",
  "AUTOREGRESSIVE",
  "NANO-GPT",
  "FASTAPI",
  "NEXT.JS",
]

export function TechMarquee() {
  const containerRef = useRef<HTMLElement>(null)
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"],
  })

  // We want it to move opposite to scroll direction
  const x1 = useTransform(scrollYProgress, [0, 1], ["0%", "-50%"])
  const x2 = useTransform(scrollYProgress, [0, 1], ["-50%", "0%"])
  
  const smoothX1 = useSpring(x1, { stiffness: 100, damping: 30 })
  const smoothX2 = useSpring(x2, { stiffness: 100, damping: 30 })

  // Duplicate items to ensure seamless infinite effect
  const repeatedTech = [...technologies, ...technologies, ...technologies]

  return (
    <section ref={containerRef} className="relative overflow-hidden border-y border-border bg-background py-8 md:py-12">
      {/* Top row - scrolls left */}
      <div className="relative flex items-center overflow-hidden h-16 md:h-24">
        <motion.div style={{ x: smoothX1 }} className="flex gap-8 md:gap-16 px-4 whitespace-nowrap">
          {repeatedTech.map((tech, index) => (
            <div key={`row1-${index}`} className="flex items-center gap-8 md:gap-16">
              <span 
                className="text-4xl md:text-6xl font-sans font-bold tracking-tighter"
                style={{
                  WebkitTextStroke: "1px var(--border)",
                  color: "transparent",
                }}
              >
                {tech}
              </span>
              <span className="h-2 w-2 rounded-full bg-primary" />
            </div>
          ))}
        </motion.div>
      </div>

      {/* Bottom row - scrolls right */}
      <div className="relative flex items-center overflow-hidden h-16 md:h-24 mt-4">
        <motion.div style={{ x: smoothX2 }} className="flex gap-8 md:gap-16 px-4 whitespace-nowrap">
          {repeatedTech.map((tech, index) => (
            <div key={`row2-${index}`} className="flex items-center gap-8 md:gap-16">
              <span 
                className="text-4xl md:text-6xl font-sans font-bold tracking-tighter text-muted-foreground transition-colors hover:text-foreground cursor-default"
              >
                {tech}
              </span>
              <span className="h-2 w-2 rounded-full bg-border" />
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}
