"use client"

import { useRef, useState } from "react"
import { motion, useScroll, useTransform, AnimatePresence } from "framer-motion"

const blocks = [
  { id: "input", name: "Input Embedding", desc: "Token indices mapped to 768-dimensional dense vectors. (GPT-2-small: n_embd=768)" },
  { id: "pos", name: "Position Embedding", desc: "Absolute positional encodings added for sequence context. Block size: 1024 tokens." },
  { id: "b1", name: "Transformer Block 1", desc: "Multi-Head Attention (12 heads) + Feed Forward." },
  { id: "b2", name: "Transformer Block 2", desc: "Feature extraction & context building." },
  { id: "b3", name: "Transformer Block 3", desc: "Mid-level semantic processing." },
  { id: "b4", name: "Transformer Block 4", desc: "High-level syntactic processing." },
  { id: "b5", name: "Transformer Block 5", desc: "Contextual refinement." },
  { id: "b6", name: "Transformer Block 6", desc: "Cross-lingual representation learning." },
  { id: "b7", name: "Transformer Block 7", desc: "Bilingual semantic integration." },
  { id: "b8", name: "Transformer Block 8", desc: "Advanced contextual reasoning." },
  { id: "b9", name: "Transformer Block 9", desc: "Response coherence modeling." },
  { id: "b10", name: "Transformer Block 10", desc: "High-level conversational abstraction." },
  { id: "b11", name: "Transformer Block 11", desc: "Pre-output refinement." },
  { id: "b12", name: "Transformer Block 12", desc: "Final contextual representations before projection." },
  { id: "norm", name: "Layer Norm", desc: "Final normalization before projection." },
  { id: "head", name: "Language Modeling Head", desc: "Projects 768D vectors back to 50,257 vocab logits. (Tied weights)" },
]

export function Architecture() {
  const containerRef = useRef<HTMLElement>(null)
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"],
  })

  const [activeBlock, setActiveBlock] = useState<string | null>(null)

  const y = useTransform(scrollYProgress, [0, 1], [100, -100])
  const opacity = useTransform(scrollYProgress, [0, 0.2, 0.8, 1], [0, 1, 1, 0])

  return (
    <section id="architecture" ref={containerRef} className="relative min-h-screen py-32 overflow-hidden bg-background">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,var(--border)_1px,transparent_1px),linear-gradient(to_bottom,var(--border)_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)] opacity-20" />

      <div className="relative z-10 max-w-7xl mx-auto px-6 md:px-12">
        <div className="text-center mb-24">
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="font-mono text-[10px] tracking-[0.3em] text-muted-foreground uppercase mb-4"
          >
            02 — Architecture
          </motion.p>
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-4xl md:text-6xl font-light tracking-tight"
          >
            A Lean <span className="italic">Transformer</span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="mt-6 text-muted-foreground max-w-2xl mx-auto"
          >
            Hover over the pipeline to inspect the flow of data through the 124M parameter GPT-2-small causal language model, fine-tuned bilingually on Hinglish and English.
          </motion.p>
        </div>

        <div className="grid lg:grid-cols-[1fr_300px] gap-12 lg:gap-24 items-start">
          {/* Visual Pipeline */}
          <motion.div style={{ y, opacity }} className="relative flex flex-col items-center">
            
            {/* Animated token flow line */}
            <div className="absolute top-0 bottom-0 w-px bg-border left-1/2 -translate-x-1/2 -z-10" />
            <motion.div 
              className="absolute top-0 w-[3px] h-32 bg-primary blur-[2px] left-1/2 -translate-x-1/2 -z-10"
              animate={{ top: ["0%", "100%"], opacity: [0, 1, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
            />

            <div className="flex flex-col gap-6 w-full max-w-md">
              {blocks.map((block, i) => (
                <motion.div
                  key={block.id}
                  initial={{ opacity: 0, x: i % 2 === 0 ? -20 : 20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1, duration: 0.5 }}
                  className={`
                    relative p-4 rounded-xl border transition-all duration-300 cursor-default
                    ${activeBlock === block.id 
                      ? "bg-primary/10 border-primary shadow-[0_0_30px_-5px_rgba(var(--primary),0.3)] scale-105 z-20" 
                      : "bg-card/50 border-border hover:border-muted-foreground/50 z-10 backdrop-blur-sm"}
                  `}
                  onMouseEnter={() => setActiveBlock(block.id)}
                  onMouseLeave={() => setActiveBlock(null)}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[10px] text-muted-foreground">{`0${i + 1}`}</span>
                    <span className="text-sm font-medium">{block.name}</span>
                    <span className="w-4" /> {/* Spacer */}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Details Panel */}
          <div className="sticky top-1/2 -translate-y-1/2 hidden lg:block">
            <AnimatePresence mode="wait">
              {activeBlock ? (
                <motion.div
                  key={activeBlock}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="p-6 rounded-2xl border border-primary/20 bg-primary/5 backdrop-blur-md"
                >
                  <h3 className="text-xl font-medium mb-2">{blocks.find(b => b.id === activeBlock)?.name}</h3>
                  <p className="text-muted-foreground text-sm leading-relaxed">
                    {blocks.find(b => b.id === activeBlock)?.desc}
                  </p>
                </motion.div>
              ) : (
                <motion.div
                  key="default"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="p-6 rounded-2xl border border-border/50 bg-card/20 backdrop-blur-sm"
                >
                  <p className="text-muted-foreground text-sm text-center">
                    Hover over a component in the pipeline to see its details.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </section>
  )
}
