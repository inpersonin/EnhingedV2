"use client"

import { motion } from "framer-motion"
import { Database, Zap, HardDrive, Cpu } from "lucide-react"

export function ModelOverview() {
  return (
    <section id="model" className="relative py-24 md:py-32 bg-background border-t border-border">
      <div className="max-w-7xl mx-auto px-6 md:px-12">
        <div className="grid md:grid-cols-2 gap-12 lg:gap-24 items-center">
          
          {/* Text Content */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="space-y-8"
          >
            <div>
              <p className="font-mono text-[10px] tracking-[0.3em] text-muted-foreground uppercase mb-4">03 — Details</p>
              <h2 className="text-4xl md:text-5xl font-light tracking-tight text-balance">
                Fine-Tuned, <span className="italic text-primary">Bilingual</span>
              </h2>
            </div>
            
            <p className="text-muted-foreground text-lg leading-relaxed">
              Enhinged V2 is GPT-2-small (124M parameters) fine-tuned on a curated bilingual dataset of everyday Hinglish and English conversations — then improved further with a reinforcement learning from AI feedback (RLHF) pass.
            </p>

            <div className="grid grid-cols-2 gap-6 pt-4">
              <div className="space-y-2">
                <Database className="w-5 h-5 text-primary/70" />
                <h4 className="font-medium">Dataset</h4>
                <p className="text-sm text-muted-foreground">Bilingual Hinglish + English corpus (~120M+ tokens), memory-mapped for streaming.</p>
              </div>
              <div className="space-y-2">
                <Zap className="w-5 h-5 text-primary/70" />
                <h4 className="font-medium">Tokenizer</h4>
                <p className="text-sm text-muted-foreground">GPT-2 Byte-Pair Encoding (BPE) via tiktoken. 50,257 vocab.</p>
              </div>
              <div className="space-y-2">
                <HardDrive className="w-5 h-5 text-primary/70" />
                <h4 className="font-medium">RLHF</h4>
                <p className="text-sm text-muted-foreground">PPO fine-tuning with AI-judged preference data improves response naturalness and relevance.</p>
              </div>
              <div className="space-y-2">
                <Cpu className="w-5 h-5 text-primary/70" />
                <h4 className="font-medium">Inference</h4>
                <p className="text-sm text-muted-foreground">KV-cache for faster generation, top-p/top-k sampling, repetition penalties.</p>
              </div>
            </div>
          </motion.div>

          {/* Visual Card */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="relative"
          >
            <div className="absolute -inset-1 bg-gradient-to-r from-primary to-purple-600 rounded-3xl blur opacity-20 animate-pulse" />
            <div className="relative glass-shell rounded-3xl p-8 border border-white/10 bg-black/40 backdrop-blur-xl">
              
              <div className="flex justify-between items-center mb-8 border-b border-white/10 pb-4">
                <span className="font-mono text-sm tracking-widest text-white/50">MODEL_CONFIG</span>
                <span className="flex h-2 w-2 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
              </div>

              <div className="space-y-4 font-mono text-sm">
                <div className="flex justify-between">
                  <span className="text-white/50">block_size</span>
                  <span className="text-primary-foreground">1024</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">vocab_size</span>
                  <span className="text-primary-foreground">50257</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">n_layer</span>
                  <span className="text-primary-foreground">12</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">n_head</span>
                  <span className="text-primary-foreground">12</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">n_embd</span>
                  <span className="text-primary-foreground">768</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">dropout</span>
                  <span className="text-primary-foreground">0.1</span>
                </div>
                <div className="flex justify-between pt-4 border-t border-white/10">
                  <span className="text-white/50">bias</span>
                  <span className="text-orange-400">true</span>
                </div>
              </div>

              <div className="mt-8 pt-4 border-t border-white/10">
                <p className="font-mono text-xs text-white/40">
                  // Total Parameters: ~124M<br/>
                  // Base: GPT-2-small + bilingual fine-tune + RLHF<br/>
                  // Best Checkpoint: checkpoints/rlhf_best.pt
                </p>
              </div>
            </div>
          </motion.div>
          
        </div>
      </div>
    </section>
  )
}
