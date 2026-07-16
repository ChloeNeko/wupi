import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  root: "src",
  publicDir: "../public",
  // Relative base so multi-page assets resolve correctly under Tauri's custom
  // protocol (tauri://localhost / https://tauri.localhost). An absolute "/base"
  // breaks the secondary terminal.html window — its <script src="/assets/...">
  // resolves to the protocol root and 404s, leaving the window unstyled/white.
  base: "./",
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    // Multi-page: the main OS shell + the terminal window. Each HTML entry
    // becomes its own bundle; Tauri loads them by URL (index.html / terminal.html).
    rollupOptions: {
      input: {
        main: resolve(__dirname, "src/index.html"),
        terminal: resolve(__dirname, "src/terminal.html"),
      },
    },
  },
});
