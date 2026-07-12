"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"

const tabs = [
  {
    id: "training",
    label: "Training Pipeline",
    code: `python prepare_data.py \\
  --input_file data/enhinged_corpus.txt \\
  --output_dir data/ \\
  --val_ratio 0.1

python train.py \\
  --mode train \\
  --data_dir data/ \\
  --out_dir checkpoints/ \\
  --block_size 256 \\
  --n_layer 6 \\
  --n_head 6 \\
  --n_embd 384 \\
  --max_iters 5000 \\
  --batch_size 32`,
    description: "Data is tokenized using GPT-2 BPE into memory-mapped numpy arrays (train.bin, val.bin). Training uses standard cross-entropy loss, AdamW optimizer, and cosine learning rate decay."
  },
  {
    id: "inference",
    label: "Inference API",
    code: `@app.post("/generate")
async def generate(req: GenerateRequest):
    # Context assembly
    context = ""
    for msg in req.conversation_history[-8:]:
        role = "User" if msg.role == "user" else "Assistant"
        context += f"{role}: {msg.content}\\n"
    context += f"User: {req.prompt}\\nAssistant:"

    # Autoregressive generation
    reply = generate_text(
        model, 
        encoding, 
        context, 
        device,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty
    )
    return {"response": reply}`,
    description: "The backend is exposed via a FastAPI application running on Hugging Face Spaces. It handles context assembly, tokenization, generation loop, and decoding."
  }
]

export function TechnicalDetails() {
  const [activeTab, setActiveTab] = useState(tabs[0].id)

  return (
    <section id="technical" className="relative py-24 md:py-32 bg-card border-t border-border">
      <div className="max-w-7xl mx-auto px-6 md:px-12">
        <div className="grid lg:grid-cols-[1fr_1.5fr] gap-12 lg:gap-24 items-center">
          
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="space-y-6"
          >
            <p className="font-mono text-[10px] tracking-[0.3em] text-muted-foreground uppercase mb-4">07 — Technical</p>
            <h2 className="text-4xl md:text-5xl font-light tracking-tight">
              Under the <span className="italic">Hood</span>
            </h2>
            <p className="text-muted-foreground leading-relaxed">
              Enhinged features a complete pipeline from raw text processing to a production-ready FastAPI backend.
            </p>
            
            <div className="flex flex-col gap-2 mt-8">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    text-left px-6 py-4 rounded-xl font-mono text-sm transition-all
                    ${activeTab === tab.id 
                      ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20 scale-105 origin-left" 
                      : "bg-background text-foreground border border-border hover:bg-muted"}
                  `}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="glass-shell rounded-3xl overflow-hidden border border-border bg-black/60 shadow-2xl"
          >
            <div className="flex items-center gap-2 px-4 py-3 border-b border-white/10 bg-black/40">
              <span className="w-3 h-3 rounded-full bg-red-500/80" />
              <span className="w-3 h-3 rounded-full bg-yellow-500/80" />
              <span className="w-3 h-3 rounded-full bg-green-500/80" />
            </div>
            
            <div className="p-6 relative">
              <AnimatePresence mode="wait">
                {tabs.map((tab) => (
                  activeTab === tab.id && (
                    <motion.div
                      key={tab.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.3 }}
                      className="space-y-6"
                    >
                      <pre className="text-sm font-mono text-green-400 overflow-x-auto p-4 rounded-xl bg-black/40 border border-white/5 scrollbar-hide">
                        <code>{tab.code}</code>
                      </pre>
                      <p className="text-sm text-white/60 leading-relaxed font-mono">
                        // {tab.description}
                      </p>
                    </motion.div>
                  )
                ))}
              </AnimatePresence>
            </div>
          </motion.div>

        </div>
      </div>
    </section>
  )
}
