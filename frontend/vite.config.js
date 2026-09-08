import { defineConfig } from 'vite'

// Dev proxy: forward API requests to local backend to avoid CORS during dev.
// Adjust target port if your backend runs on a different port.
export default defineConfig({
  server: {
    proxy: {
      '/process-url': { target: 'http://localhost:10000', changeOrigin: true },
      '/status': { target: 'http://localhost:10000', changeOrigin: true },
      '/recognize': { target: 'http://localhost:10000', changeOrigin: true },
      '/jobs': { target: 'http://localhost:10000', changeOrigin: true },
    },
  },
})
