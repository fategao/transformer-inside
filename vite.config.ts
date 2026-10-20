import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'
import { readFileSync } from 'node:fs'

const siteConfig = JSON.parse(readFileSync(new URL('./site.config.json', import.meta.url), 'utf8')) as {
  basePath: string
}
const rawBase = process.env.BASE_PATH ?? (process.env.NODE_ENV === 'production' ? siteConfig.basePath : '/')
const base = rawBase.endsWith('/') ? rawBase : `${rawBase}/`

export default defineConfig({
  base,
  plugins: [react(), tailwindcss()],
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 1000,
  },
})
