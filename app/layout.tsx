import type { Metadata, Viewport } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import { ThemeProvider } from "@/components/theme-provider"
import "../styles/animations.css"
import "./globals.css"

const geistSans = Geist({
  subsets: ["latin"],
  variable: "--font-geist",
})

const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
})

export const metadata: Metadata = {
  title: "Enhinged V2 | Bilingual Hinglish GPT Chat Interface",
  description: "Enhinged V2 — A GPT-2-small (124M) model fine-tuned bilingually on Hinglish and English, with RLHF quality tuning and faster KV-cached generation.",
  openGraph: {
    title: "Enhinged V2 | Bilingual Hinglish GPT",
    description: "Experience a bilingual Hinglish+English conversational AI with RLHF quality tuning in a premium, glassmorphic interface.",
    type: "website",
  },
}

export const viewport: Viewport = {
  themeColor: "#050505",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`} suppressHydrationWarning>
      <body className="overflow-x-hidden antialiased bg-background text-foreground">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
          disableTransitionOnChange
        >
          <div className="noise-overlay" />
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}