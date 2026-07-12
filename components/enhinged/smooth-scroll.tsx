"use client"

import type { ReactNode } from "react"
import { useEffect } from "react"

export function SmoothScroll({ children }: { children: ReactNode }) {
  useEffect(() => {
    // Add smooth scrolling behavior to html element
    document.documentElement.style.scrollBehavior = "smooth"
    
    return () => {
      document.documentElement.style.scrollBehavior = "auto"
    }
  }, [])

  return <>{children}</>
}
