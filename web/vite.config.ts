import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// dev：/api 代理到本机单入口 app.py（7861）；部署后同端口直接出 /api/*
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:7861',
    },
  },
  build: {
    outDir: 'dist',
  },
})
