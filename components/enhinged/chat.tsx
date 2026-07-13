"use client"

import { AnimatePresence, motion } from "framer-motion"
import { ArrowUpRight, SendHorizontal, Sparkles, RefreshCcw, Settings, X, Check } from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"

type Message = {
  id: number
  role: "user" | "assistant"
  content: string
  streaming?: boolean
}

const suggestions = [
  "Explain attention in Hinglish",
  "Mujhe ek short reply likh do",
  "How was this model trained?",
  "Ek funny line bolo",
]

const DEFAULT_API = process.env.NEXT_PUBLIC_API_URL || ""
const LS_KEY = "enhinged_api_url"

function getApiBase(): string {
  if (typeof window !== "undefined") {
    const stored = localStorage.getItem(LS_KEY)
    if (stored && stored.trim()) return stored.trim().replace(/\/$/, "")
  }
  return DEFAULT_API.replace(/\/$/, "")
}

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      role: "assistant",
      content: "Main Enhinged hoon. Hindi, English, ya Hinglish me poochho — main backend se reply launga.",
    },
  ])
  const [input, setInput] = useState("")
  const [pending, setPending] = useState(false)
  const [status, setStatus] = useState("ready")
  const nextId = useRef(2)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Settings panel state
  const [showSettings, setShowSettings] = useState(false)
  const [apiInput, setApiInput] = useState("")
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    // Pre-fill settings input with current stored URL
    if (typeof window !== "undefined") {
      setApiInput(localStorage.getItem(LS_KEY) || "")
    }
  }, [])

  useEffect(() => {
    // Scroll to bottom on new messages
    const scrollContainer = scrollRef.current?.querySelector('[data-radix-scroll-area-viewport]')
    if (scrollContainer) {
      scrollContainer.scrollTo({ top: scrollContainer.scrollHeight, behavior: "smooth" })
    }
  }, [messages, pending])

  const conversationHistory = useMemo(
    () =>
      messages.map((message) => ({
        role: message.role,
        content: message.content,
      })),
    [messages],
  )

  const handleSaveUrl = () => {
    if (typeof window !== "undefined") {
      const trimmed = apiInput.trim().replace(/\/$/, "")
      if (trimmed) {
        localStorage.setItem(LS_KEY, trimmed)
      } else {
        localStorage.removeItem(LS_KEY)
      }
    }
    setSaved(true)
    setTimeout(() => {
      setSaved(false)
      setShowSettings(false)
    }, 1200)
  }

  const handleReset = () => {
    setMessages([{
      id: nextId.current++,
      role: "assistant",
      content: "Conversation reset. Kya poochhna chahte ho?",
    }])
    setStatus("ready")
  }

  const sendMessage = async (rawText: string) => {
    const trimmed = rawText.trim()
    if (!trimmed || pending) return

    const currentApiBase = getApiBase()
    if (!currentApiBase) {
      setMessages((current) => [
        ...current,
        {
          id: nextId.current++,
          role: "assistant",
          content: "⚙️ Backend URL not set. Click the settings icon in the chat header and paste your Railway URL.",
          streaming: false,
        },
      ])
      setShowSettings(true)
      return
    }

    const userMessage: Message = { id: nextId.current++, role: "user", content: trimmed }
    const assistantId = nextId.current++
    setInput("")
    setPending(true)
    setStatus("thinking")
    setMessages((current) => [
      ...current,
      userMessage,
      { id: assistantId, role: "assistant", content: "", streaming: true },
    ])

    try {
      const response = await fetch(`${currentApiBase}/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: trimmed,
          max_new_tokens: 110,
          temperature: 0.82,
          top_k: 40,
          top_p: 0.95,
          repetition_penalty: 1.08,
          do_sample: true,
          conversation_history: [...conversationHistory, { role: "user", content: trimmed }].slice(-8),
        }),
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const data: { response?: string } = await response.json()
      const text = (data.response || "No response returned.").trim() || "No response returned."

      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId ? { ...message, content: text, streaming: false } : message,
        ),
      )
      setStatus("connected")
    } catch {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: "Backend unavailable. Make sure your Railway service is running and the URL in ⚙️ settings is correct.",
                streaming: false,
              }
            : message,
        ),
      )
      setStatus("error")
    } finally {
      setPending(false)
    }
  }

  const currentApiBase = getApiBase()

  return (
    <section id="chat" className="relative min-h-screen py-24 px-4 md:px-8 lg:px-12 bg-background flex flex-col justify-center">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-primary/10 via-background to-background pointer-events-none" />
      
      <div className="relative z-10 w-full max-w-5xl mx-auto grid gap-12 lg:grid-cols-[1fr_1.5fr] items-center">
        {/* Left Side: Context */}
        <motion.div 
          initial={{ opacity: 0, x: -40 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="space-y-6"
        >
          <p className="font-mono text-[10px] tracking-[0.3em] text-muted-foreground uppercase">01 — Interface</p>
          <h2 className="text-4xl md:text-5xl font-light tracking-tight">
            Seamless <span className="italic text-muted-foreground">Interaction</span>
          </h2>
          <p className="text-muted-foreground text-sm leading-relaxed">
            The frontend is a static export deployed on GitHub Pages. The backend runs on Railway with a custom FastAPI wrapper around the PyTorch model.
          </p>
          <p className="text-muted-foreground text-sm leading-relaxed">
            The chat context window holds 256 tokens, meaning it can remember the last few turns of conversation before rolling over. Try asking it something in Hinglish.
          </p>
          
          {currentApiBase ? (
            <a
              href={currentApiBase}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-4 py-2 font-mono text-[10px] tracking-[0.3em] uppercase transition-colors hover:bg-muted mt-4"
            >
              View Backend <ArrowUpRight className="size-3.5" />
            </a>
          ) : (
            <button
              onClick={() => setShowSettings(true)}
              className="inline-flex items-center gap-2 rounded-full border border-primary/50 bg-primary/10 px-4 py-2 font-mono text-[10px] tracking-[0.3em] uppercase transition-colors hover:bg-primary/20 mt-4 text-primary"
            >
              <Settings className="size-3.5" /> Set Backend URL
            </button>
          )}
        </motion.div>

        {/* Right Side: Chat UI */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="glass-shell relative overflow-hidden rounded-[2rem] border border-border bg-card/40 shadow-2xl"
        >
          {/* Settings Panel Overlay */}
          <AnimatePresence>
            {showSettings && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-4 bg-card/95 backdrop-blur-sm p-6"
              >
                <p className="font-mono text-[10px] tracking-[0.3em] text-muted-foreground uppercase">Backend URL</p>
                <p className="text-xs text-muted-foreground text-center max-w-xs">
                  Paste your Railway service URL (e.g. <code className="text-primary">https://enhingedv2-production.up.railway.app</code>)
                </p>
                <input
                  autoFocus
                  value={apiInput}
                  onChange={(e) => setApiInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") handleSaveUrl() }}
                  placeholder="https://your-service.up.railway.app"
                  className="w-full rounded-xl border border-border bg-background/80 px-4 py-3 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                />
                <div className="flex gap-3 w-full">
                  <button
                    onClick={() => setShowSettings(false)}
                    className="flex-1 flex items-center justify-center gap-2 rounded-xl border border-border py-2.5 text-xs text-muted-foreground hover:bg-muted transition-colors"
                  >
                    <X className="size-3.5" /> Cancel
                  </button>
                  <button
                    onClick={handleSaveUrl}
                    className="flex-1 flex items-center justify-center gap-2 rounded-xl bg-primary py-2.5 text-xs text-primary-foreground hover:bg-primary/90 transition-colors"
                  >
                    {saved ? <Check className="size-3.5" /> : <Check className="size-3.5" />}
                    {saved ? "Saved!" : "Save & Connect"}
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Header */}
          <div className="border-b border-border/50 px-5 py-4 backdrop-blur-md">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <span className="flex size-10 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                  <Sparkles className="size-5" />
                </span>
                <div>
                  <p className="font-mono text-[10px] tracking-[0.3em] text-muted-foreground uppercase">Enhinged V2 Chat</p>
                  <p className="text-xs text-foreground/80 font-medium">Ready for inference</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className={`rounded-full border px-3 py-1 font-mono text-[10px] tracking-[0.25em] uppercase transition-colors
                  ${status === 'thinking' ? 'border-primary/50 text-primary animate-pulse' : 
                    status === 'error' ? 'border-destructive/50 text-destructive' : 
                    'border-border text-muted-foreground'}
                `}>
                  {status}
                </span>
                <button
                  onClick={() => setShowSettings(true)}
                  className="p-2 hover:bg-muted rounded-full transition-colors text-muted-foreground"
                  title="Configure Backend URL"
                >
                  <Settings className="size-4" />
                </button>
                <button 
                  onClick={handleReset}
                  className="p-2 hover:bg-muted rounded-full transition-colors text-muted-foreground"
                  title="Reset Conversation"
                >
                  <RefreshCcw className="size-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Messages Area */}
          <ScrollArea ref={scrollRef} className="h-[28rem] md:h-[32rem] px-5 py-5">
            <div className="flex flex-col gap-4">
              <AnimatePresence initial={false}>
                {messages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 10, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    transition={{ duration: 0.3 }}
                    className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[85%] rounded-[1.2rem] px-5 py-3.5 text-sm leading-relaxed shadow-sm ${
                        message.role === "user"
                          ? "rounded-br-sm bg-primary text-primary-foreground"
                          : "rounded-bl-sm border border-border/50 bg-card/80 text-card-foreground backdrop-blur-sm"
                      }`}
                    >
                      {message.content || (message.streaming ? <span className="animate-caret">|</span> : null)}
                      {message.streaming ? <span className="ml-1 inline-block h-4 w-[2px] bg-foreground/70 align-middle" /> : null}
                    </div>
                  </motion.div>
                ))}

                {pending ? (
                  <motion.div
                    key="thinking"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex justify-start"
                  >
                    <div className="flex items-center gap-1.5 rounded-[1.2rem] rounded-bl-sm border border-border/50 bg-card/80 px-5 py-4 backdrop-blur-sm shadow-sm">
                      {[0, 1, 2].map((dot) => (
                        <motion.span
                          key={dot}
                          className="size-1.5 rounded-full bg-primary/70"
                          animate={{ y: [0, -4, 0], opacity: [0.4, 1, 0.4] }}
                          transition={{ duration: 0.8, repeat: Infinity, delay: dot * 0.15 }}
                        />
                      ))}
                    </div>
                  </motion.div>
                ) : null}
              </AnimatePresence>
            </div>
          </ScrollArea>

          {/* Suggestions */}
          <div className="flex gap-2 overflow-x-auto border-t border-border/50 px-5 py-3 bg-card/30 backdrop-blur-sm scrollbar-hide">
            {suggestions.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => sendMessage(suggestion)}
                disabled={pending}
                className="shrink-0 rounded-full border border-border/50 bg-background/50 px-3.5 py-1.5 font-mono text-[10px] tracking-[0.15em] text-muted-foreground uppercase transition-all hover:border-border hover:bg-muted hover:text-foreground disabled:opacity-45"
              >
                {suggestion}
              </button>
            ))}
          </div>

          {/* Input Area */}
          <form
            className="flex items-center gap-3 border-t border-border/50 bg-card/50 p-4 backdrop-blur-md"
            onSubmit={(event) => {
              event.preventDefault()
              void sendMessage(input)
            }}
          >
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Type a prompt..."
              aria-label="Message Enhinged"
              className="flex-1 rounded-full border border-border bg-background/80 px-5 py-3.5 text-sm outline-none transition-all placeholder:text-muted-foreground focus:border-primary focus:ring-1 focus:ring-primary shadow-inner"
            />
            <button
              type="submit"
              disabled={!input.trim() || pending}
              className="flex size-12 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground transition-all hover:scale-105 active:scale-95 disabled:scale-100 disabled:opacity-50 shadow-md hover:shadow-primary/20"
              aria-label="Send message"
            >
              <SendHorizontal className="size-4 -ml-0.5" />
            </button>
          </form>
        </motion.div>
      </div>
    </section>
  )
}
