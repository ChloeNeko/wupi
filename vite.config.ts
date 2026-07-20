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
    // The entry HTML is `wupi.html` (renamed from the Vite-default
    // index.html per AGENTS.md §8C). Vite picks up the entry via
    // rollupOptions.input; without this it would look for index.html in
    // the `src` root and emit nothing. The Tauri window's `url: "wupi.html"`
    // (tauri.conf.json) loads this emitted file at runtime.
    rollupOptions: {
      input: "src/wupi.html",
    },
  },
});
