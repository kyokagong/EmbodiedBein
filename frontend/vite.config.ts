import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { fileURLToPath, URL } from "node:url";

const apiPort = process.env.EB_PORT_API || "8000";
const vitePort = Number(process.env.EB_PORT_VITE || 5173);
const apiOrigin = `http://127.0.0.1:${apiPort}`;

export default defineConfig({
  plugins: [vue()],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  server: {
    port: vitePort,
    strictPort: true,
    proxy: {
      "/api": apiOrigin,
      "/ws": { target: `ws://127.0.0.1:${apiPort}`, ws: true },
      "/healthz": apiOrigin,
    },
  },
  build: {
    outDir: process.env.EB_WEB_OUT || "../backend/web/dist",
    emptyOutDir: true,
  },
});
