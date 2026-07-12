"use client"

import { useRef } from "react"
import { motion, useScroll, useTransform } from "framer-motion"
import { ArrowRight } from "lucide-react"

export function Hero() {
  const containerRef = useRef<HTMLElement>(null)
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end start"],
  })

  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0])
  const scale = useTransform(scrollYProgress, [0, 0.5], [1, 0.8])
  const y = useTransform(scrollYProgress, [0, 0.5], [0, 100])

  return (
    <section id="home" ref={containerRef} className="relative h-screen w-full overflow-hidden bg-background">
      {/* Background Orbs & Grid */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="grid-line absolute inset-0 opacity-20" aria-hidden="true" />
        <motion.div
          aria-hidden="true"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 2, ease: "easeOut" }}
          className="animate-orb absolute left-[15%] top-[20%] h-[32rem] w-[32rem] rounded-full bg-[radial-gradient(circle,rgba(37,99,235,0.25)_0%,rgba(37,99,235,0.05)_40%,transparent_70%)] blur-3xl"
        />
        <motion.div
          aria-hidden="true"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 2, delay: 0.5, ease: "easeOut" }}
          className="animate-orb absolute right-[10%] bottom-[10%] h-[24rem] w-[24rem] rounded-full bg-[radial-gradient(circle,rgba(139,92,246,0.2)_0%,rgba(139,92,246,0.05)_40%,transparent_70%)] blur-3xl"
          style={{ animationDelay: "-4.5s" }}
        />
      </div>

      <motion.div style={{ opacity, scale, y }} className="relative z-10 h-full flex flex-col justify-center items-center text-center p-6 md:p-12 pt-20">
        


        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="max-w-5xl text-balance text-6xl font-light tracking-tighter sm:text-7xl md:text-8xl lg:text-9xl"
        >
          ENHINGED V2
          <br />
          <span className="italic text-muted-foreground">Hinglish AI</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mt-8 max-w-2xl text-balance text-lg text-muted-foreground md:text-xl font-light"
        >
          A GPT-2-small (124M) model fine-tuned bilingually on Hinglish and English conversations, then improved with RLHF. Fast, compact, and genuinely bilingual.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.7 }}
          className="mt-12 flex flex-wrap items-center justify-center gap-4"
        >
          <button
            onClick={() => document.querySelector("#chat")?.scrollIntoView({ behavior: "smooth" })}
            className="group relative inline-flex items-center gap-2 overflow-hidden rounded-full bg-primary px-8 py-4 font-mono text-sm uppercase tracking-widest text-primary-foreground transition-all hover:scale-105 active:scale-95"
          >
            <span className="relative z-10 flex items-center gap-2">
              Start Chatting <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
            </span>
            <div className="absolute inset-0 z-0 bg-gradient-to-r from-transparent via-white/20 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700 ease-in-out" />
          </button>
          
          <button
            onClick={() => document.querySelector("#architecture")?.scrollIntoView({ behavior: "smooth" })}
            className="inline-flex items-center rounded-full border border-border bg-transparent px-8 py-4 font-mono text-sm uppercase tracking-widest text-foreground transition-colors hover:bg-muted"
          >
            Explore Architecture
          </button>
        </motion.div>
        
        {/* Key Stats Row */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.9 }}
          className="mt-16 flex flex-wrap items-center justify-center gap-8 md:gap-16 border-t border-border pt-8 w-full max-w-4xl"
        >
          {[
            { label: "Parameters", value: "124M" },
            { label: "Layers", value: "12" },
            { label: "Context", value: "1024" },
            { label: "RLHF", value: "✓" }
          ].map((stat) => (
            <div key={stat.label} className="text-center">
              <p className="text-3xl md:text-4xl font-light tabular-nums">{stat.value}</p>
              <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">{stat.label}</p>
            </div>
          ))}
        </motion.div>
      </motion.div>

      {/* Scroll Indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20"
      >
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 1.5, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" }}
          className="flex flex-col items-center gap-2"
        >
          <span className="font-mono text-[10px] tracking-widest text-muted-foreground uppercase">Scroll</span>
          <div className="w-px h-8 bg-gradient-to-b from-border to-transparent" />
        </motion.div>
      </motion.div>
    </section>
  )
}
