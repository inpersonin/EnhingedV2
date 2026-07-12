const rawBasePath = process.env.NEXT_PUBLIC_BASE_PATH || ""
const basePath = rawBasePath === "/" ? "" : rawBasePath.replace(/\/$/, "")

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  basePath: basePath || undefined,
  assetPrefix: basePath ? `${basePath}/` : undefined,
}

export default nextConfig