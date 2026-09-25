import { defineConfig } from "vite";

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/v1": { target: "http://127.0.0.1:8000", ws: true, changeOrigin: true },
      "/healthz": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
