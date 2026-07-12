"use client"

export function SectionBlend() {
  return (
    <div className="relative h-24 md:h-32 bg-background w-full overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-transparent to-background/50 pointer-events-none" />
      <div className="absolute top-1/2 left-0 right-0 h-px bg-gradient-to-r from-transparent via-border to-transparent opacity-50" />
    </div>
  )
}
