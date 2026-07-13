import { defineConfig } from "vite";

export default defineConfig({
  root: "src",
  publicDir: "../public",
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
