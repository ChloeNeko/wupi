import { defineConfig } from "vite";

export default defineConfig({
  root: "src",
  publicDir: "../public",
  // Relative base so assets resolve correctly under Tauri's custom protocol
  // (tauri://localhost / https://tauri.localhost). An absolute "/base" 404s.
  base: "./",
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
});
