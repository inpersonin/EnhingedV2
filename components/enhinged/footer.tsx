"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { ArrowUpRight } from "lucide-react"

export function Footer() {
  const [time, setTime] = useState("")

  useEffect(() => {
    const updateTime = () => {
      const now = new Date()
      const hours = now.getHours().toString().padStart(2, "0")
      const minutes = now.getMinutes().toString().padStart(2, "0")
      const seconds = now.getSeconds().toString().padStart(2, "0")
      const milliseconds = now.getMilliseconds().toString().padStart(3, "0")
      setTime(`${hours}:${minutes}:${seconds}.${milliseconds}`)
    }

    updateTime()
    const interval = setInterval(updateTime, 10)
    return () => clearInterval(interval)
  }, [])

  return (
    <footer className="relative bg-background border-t border-border">
      {/* Main CTA */}
      <motion.a
        href="https://github.com/inpersonin"
        target="_blank"
        rel="noreferrer"
        className="group relative block overflow-hidden border-b border-border"
      >
        <div className="absolute inset-0 bg-primary translate-y-full group-hover:translate-y-0 transition-transform duration-500 ease-[0.25,0.46,0.45,0.94]" />

        <div className="relative py-16 md:py-24 px-8 md:px-12 max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-8">
          <h2 className="font-sans text-4xl md:text-6xl lg:text-8xl font-light tracking-tight text-center md:text-left text-foreground group-hover:text-primary-foreground transition-colors duration-300">
            Explore the <span className="italic">Source</span>
          </h2>

          <div className="text-foreground group-hover:text-primary-foreground group-hover:rotate-45 transition-all duration-300">
            <ArrowUpRight className="w-12 h-12 md:w-16 md:h-16" />
          </div>
        </div>
      </motion.a>

      {/* Footer Info */}
      <div className="px-8 md:px-12 py-8 max-w-7xl mx-auto">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6 text-center md:text-left">
          
          <div className="font-mono text-[10px] tracking-widest text-muted-foreground uppercase">
            <span className="mr-2">Local Time</span>
            <span className="text-foreground tabular-nums">{time}</span>
          </div>

          <div className="flex gap-8">
            <a
              href="https://github.com/inpersonin/EnhingedV2"
              target="_blank"
              rel="noreferrer"
              className="font-mono text-[10px] tracking-widest text-muted-foreground hover:text-foreground transition-colors uppercase"
            >
              GitHub
            </a>
            <a
              href={process.env.NEXT_PUBLIC_API_URL || "https://inpersonin-enhingedv2.hf.space"}
              target="_blank"
              rel="noreferrer"
              className="font-mono text-[10px] tracking-widest text-muted-foreground hover:text-foreground transition-colors uppercase"
            >
              Hugging Face
            </a>
          </div>

          <p className="font-mono text-[10px] tracking-widest text-muted-foreground uppercase">
            © {new Date().getFullYear()} Enhinged V2
          </p>
        </div>
      </div>
    </footer>
  )
}
