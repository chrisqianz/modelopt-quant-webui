import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// dev：/api 代理到本机 serve.py（7862）；部署后同端口直接出 /api/*
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:7862',
    },
  },
  build: {
    outDir: 'dist',
  },
})
