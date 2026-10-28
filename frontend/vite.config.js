import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// During `npm run dev`, requests to /api/* are proxied to the FastAPI
// backend on port 8000. That way the React code just fetches
// /api/chat, /api/session/... and it Just Works without CORS gymnastics
// or hard-coded absolute URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
