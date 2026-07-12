"use client"

import { useRef } from "react"
import { motion, useInView } from "framer-motion"

export function TrainingGraphs() {
  const containerRef = useRef<HTMLDivElement>(null)
  const inView = useInView(containerRef, { once: true, margin: "-100px" })

  return (
    <section id="training" className="relative py-24 md:py-32 bg-background border-t border-border">
      <div className="max-w-7xl mx-auto px-6 md:px-12">
        <div className="grid lg:grid-cols-[1fr_2fr] gap-12 lg:gap-24 items-center">
          
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="space-y-6"
          >
            <p className="font-mono text-[10px] tracking-[0.3em] text-muted-foreground uppercase mb-4">05 — Training</p>
            <h2 className="text-4xl md:text-5xl font-light tracking-tight text-balance">
              Convergence <span className="italic">Dynamics</span>
            </h2>
            <p className="text-muted-foreground leading-relaxed">
              Trained for 5000 iterations using AdamW optimizer with a cosine learning rate decay schedule. The loss converges smoothly to 1.4363, achieving a perplexity of 4.2.
            </p>
            
            <div className="p-6 rounded-2xl bg-card border border-border space-y-4">
              <div className="flex justify-between items-center border-b border-border pb-2">
                <span className="font-mono text-sm text-muted-foreground">Optimizer</span>
                <span className="font-medium text-foreground">AdamW</span>
              </div>
              <div className="flex justify-between items-center border-b border-border pb-2">
                <span className="font-mono text-sm text-muted-foreground">Max LR</span>
                <span className="font-medium text-foreground">6e-4</span>
              </div>
              <div className="flex justify-between items-center border-b border-border pb-2">
                <span className="font-mono text-sm text-muted-foreground">Min LR</span>
                <span className="font-medium text-foreground">6e-5</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="font-mono text-sm text-muted-foreground">Warmup</span>
                <span className="font-medium text-foreground">200 iters</span>
              </div>
            </div>
          </motion.div>

          {/* Graph Visualization */}
          <div ref={containerRef} className="glass-shell rounded-3xl p-6 md:p-10 border border-border h-[400px] flex flex-col relative overflow-hidden">
            <div className="flex justify-between items-center mb-6">
              <h3 className="font-mono text-sm uppercase tracking-widest text-muted-foreground">Training Loss</h3>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-primary/40 border border-primary" />
                  <span className="text-xs text-muted-foreground">Train</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-green-500/40 border border-green-500" />
                  <span className="text-xs text-muted-foreground">Val</span>
                </div>
              </div>
            </div>

            <div className="flex-1 relative border-l border-b border-border/50">
              {/* Y Axis Labels */}
              <div className="absolute -left-8 top-0 text-[10px] text-muted-foreground font-mono">10.0</div>
              <div className="absolute -left-8 top-1/4 text-[10px] text-muted-foreground font-mono">7.5</div>
              <div className="absolute -left-8 top-1/2 text-[10px] text-muted-foreground font-mono">5.0</div>
              <div className="absolute -left-8 top-[75%] text-[10px] text-muted-foreground font-mono">2.5</div>
              <div className="absolute -left-8 bottom-0 text-[10px] text-muted-foreground font-mono transform translate-y-1/2">0.0</div>
              
              {/* Grid lines */}
              <div className="absolute inset-0 flex flex-col justify-between">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="w-full h-px bg-border/30" />
                ))}
              </div>

              {/* Simulated Loss Curve SVG */}
              <svg className="absolute inset-0 w-full h-full overflow-visible" preserveAspectRatio="none" viewBox="0 0 100 100">
                {/* Train Loss Line */}
                <motion.path
                  d="M0,10 Q10,70 30,80 T60,83 T100,85"
                  fill="none"
                  stroke="var(--primary)"
                  strokeWidth="0.8"
                  strokeOpacity="0.8"
                  initial={{ pathLength: 0 }}
                  animate={inView ? { pathLength: 1 } : { pathLength: 0 }}
                  transition={{ duration: 2, ease: "easeInOut" }}
                />
                
                {/* Val Loss Line */}
                <motion.path
                  d="M0,12 Q12,68 32,78 T62,82 T100,85.6"
                  fill="none"
                  stroke="#22c55e"
                  strokeWidth="0.8"
                  strokeOpacity="0.8"
                  initial={{ pathLength: 0 }}
                  animate={inView ? { pathLength: 1 } : { pathLength: 0 }}
                  transition={{ duration: 2, delay: 0.2, ease: "easeInOut" }}
                />
              </svg>
            </div>
            
            {/* X Axis Labels */}
            <div className="flex justify-between mt-2 pt-2 text-[10px] text-muted-foreground font-mono border-t border-transparent">
              <span>0</span>
              <span>1250</span>
              <span>2500</span>
              <span>3750</span>
              <span>5000</span>
            </div>
          </div>

        </div>
      </div>
    </section>
  )
}
