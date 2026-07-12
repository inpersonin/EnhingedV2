"use client"

import { useState } from "react"
import { motion } from "framer-motion"

// Sample sentence tokens
const tokens = ["Haan", "bhai", "sab", "theek", "hai", ",", "tu", "bata", "?"]

// Generate a fake attention matrix for demonstration
const generateAttentionMatrix = (size: number, headSeed: number) => {
  const matrix = []
  for (let i = 0; i < size; i++) {
    const row = []
    for (let j = 0; j < size; j++) {
      // Causal masking (can only attend to past tokens)
      if (j > i) {
        row.push(0)
      } else {
        // Pseudo-random attention weights biased towards recent tokens and specific heads
        let weight = Math.max(0, 1 - (i - j) * 0.2) * Math.random()
        // Special case: Head 0 pays attention to punctuation, Head 1 to subjects, etc.
        if (headSeed === 0 && (tokens[j] === "?" || tokens[j] === ",")) weight += 0.5
        if (headSeed === 1 && (tokens[j] === "bhai" || tokens[j] === "tu")) weight += 0.5
        
        row.push(weight)
      }
    }
    // Normalize row
    const sum = row.reduce((a, b) => a + b, 0)
    matrix.push(row.map(v => sum > 0 ? v / sum : 0))
  }
  return matrix
}

export function AttentionHeatmap() {
  const [activeHead, setActiveHead] = useState(0)
  const [hoveredToken, setHoveredToken] = useState<number | null>(null)
  
  const matrix = generateAttentionMatrix(tokens.length, activeHead)

  return (
    <section id="attention" className="relative py-24 md:py-32 bg-background border-t border-border">
      <div className="max-w-7xl mx-auto px-6 md:px-12">
        <div className="text-center mb-16">
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="font-mono text-[10px] tracking-[0.3em] text-muted-foreground uppercase mb-4"
          >
            06 — Interpretability
          </motion.p>
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-4xl md:text-6xl font-light tracking-tight"
          >
            Attention <span className="italic">Mechanisms</span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="mt-6 text-muted-foreground max-w-2xl mx-auto"
          >
            Interactive visualization of self-attention weights. See how the model dynamically weighs context when processing Hinglish.
          </motion.p>
        </div>

        <div className="grid lg:grid-cols-[250px_1fr] gap-12 items-start max-w-5xl mx-auto">
          
          {/* Head Selector */}
          <div className="flex flex-col gap-2">
            <h3 className="font-mono text-sm tracking-widest text-muted-foreground uppercase mb-4">Select Head</h3>
            {[0, 1, 2, 3, 4, 5].map((head) => (
              <button
                key={head}
                onClick={() => setActiveHead(head)}
                className={`
                  text-left px-4 py-3 rounded-xl border font-mono text-sm transition-all
                  ${activeHead === head 
                    ? "bg-primary text-primary-foreground border-primary" 
                    : "bg-card text-foreground border-border hover:bg-muted"}
                `}
              >
                Attention Head {head + 1}
              </button>
            ))}
          </div>

          {/* Heatmap */}
          <div className="glass-shell rounded-3xl p-6 md:p-10 border border-border overflow-x-auto">
            <div className="min-w-[500px]">
              
              {/* Columns (Target tokens) */}
              <div className="flex mb-2">
                <div className="w-24 shrink-0" /> {/* Spacer */}
                <div className="flex-1 grid" style={{ gridTemplateColumns: `repeat(${tokens.length}, minmax(0, 1fr))` }}>
                  {tokens.map((token, i) => (
                    <div key={`col-${i}`} className="text-center font-mono text-xs text-muted-foreground truncate px-1">
                      {token}
                    </div>
                  ))}
                </div>
              </div>

              {/* Rows */}
              <div className="flex flex-col gap-1">
                {tokens.map((token, i) => (
                  <div 
                    key={`row-${i}`} 
                    className="flex items-center gap-2 group"
                    onMouseEnter={() => setHoveredToken(i)}
                    onMouseLeave={() => setHoveredToken(null)}
                  >
                    <div className="w-24 shrink-0 text-right font-mono text-xs text-foreground pr-2 truncate">
                      {token}
                    </div>
                    
                    <div className="flex-1 grid gap-1" style={{ gridTemplateColumns: `repeat(${tokens.length}, minmax(0, 1fr))` }}>
                      {matrix[i].map((weight, j) => {
                        const isMasked = j > i
                        const isHighlighted = hoveredToken === i || hoveredToken === j
                        return (
                          <div 
                            key={`cell-${i}-${j}`} 
                            className={`
                              aspect-square rounded-md transition-all duration-300
                              ${isMasked ? 'bg-transparent border border-border/20' : ''}
                              ${!isMasked && isHighlighted ? 'ring-1 ring-primary/50' : ''}
                            `}
                            style={{
                              backgroundColor: !isMasked ? `rgba(37, 99, 235, ${weight * 1.5})` : undefined
                            }}
                            title={`Weight: ${weight.toFixed(3)}`}
                          />
                        )
                      })}
                    </div>
                  </div>
                ))}
              </div>

            </div>
          </div>

        </div>
      </div>
    </section>
  )
}
