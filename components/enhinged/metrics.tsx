"use client"

import { useRef, useEffect, useState } from "react"
import { motion, useInView } from "framer-motion"

function Counter({ from, to, duration = 2, decimals = 0, suffix = "" }: { from: number, to: number, duration?: number, decimals?: number, suffix?: string }) {
  const nodeRef = useRef<HTMLSpanElement>(null)
  const inView = useInView(nodeRef, { once: true, margin: "-100px" })
  const [value, setValue] = useState(from)

  useEffect(() => {
    if (!inView) return

    let startTime: number
    let animationFrame: number

    const update = (timestamp: number) => {
      if (!startTime) startTime = timestamp
      const progress = Math.min((timestamp - startTime) / (duration * 1000), 1)
      
      // easeOutQuart
      const easeProgress = 1 - Math.pow(1 - progress, 4)
      
      setValue(from + (to - from) * easeProgress)

      if (progress < 1) {
        animationFrame = requestAnimationFrame(update)
      }
    }

    animationFrame = requestAnimationFrame(update)
    return () => cancelAnimationFrame(animationFrame)
  }, [inView, from, to, duration])

  return (
    <span ref={nodeRef}>
      {value.toFixed(decimals)}{suffix}
    </span>
  )
}

export function Metrics() {
  return (
    <section id="metrics" className="relative py-24 md:py-32 bg-background border-t border-border overflow-hidden">
      {/* Background decoration */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-primary/5 rounded-full blur-[100px] pointer-events-none" />

      <div className="max-w-7xl mx-auto px-6 md:px-12 relative z-10">
        <div className="text-center mb-16 md:mb-24">
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="font-mono text-[10px] tracking-[0.3em] text-muted-foreground uppercase mb-4"
          >
            04 — Metrics
          </motion.p>
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-4xl md:text-6xl font-light tracking-tight"
          >
            Performance <span className="italic">Snapshot</span>
          </motion.h2>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-8">
          {[
            { label: "Total Parameters", value: 124, decimals: 0, suffix: "M", delay: 0.1 },
            { label: "Base Val Loss", value: 1.4363, decimals: 4, delay: 0.2 },
            { label: "Base Perplexity", value: 4.2, decimals: 1, delay: 0.3 },
            { label: "Context Window", value: 1024, decimals: 0, delay: 0.4 },
            { label: "Vocab Size", value: 50257, decimals: 0, delay: 0.5 },
            { label: "Batch Size", value: 32, decimals: 0, delay: 0.6 },
            { label: "RLHF PPO Iters", value: 500, decimals: 0, suffix: "+", delay: 0.7 },
            { label: "Gen Speedup", value: 2, decimals: 0, suffix: "x+", delay: 0.8 },
          ].map((metric, i) => (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: metric.delay, ease: [0.25, 0.46, 0.45, 0.94] }}
              className="group glass-shell rounded-2xl p-6 md:p-8 hover:bg-card/40 transition-colors"
            >
              <p className="font-mono text-[10px] md:text-xs tracking-widest text-muted-foreground uppercase mb-4">
                {metric.label}
              </p>
              <p className="text-3xl md:text-4xl lg:text-5xl font-light tabular-nums tracking-tighter text-foreground group-hover:text-primary transition-colors">
                <Counter from={0} to={metric.value} decimals={metric.decimals} suffix={metric.suffix} />
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}
