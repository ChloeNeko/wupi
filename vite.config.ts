import { defineConfig } from "vite";

export default defineConfig({
  root: "src",
  publicDir: "../public",
  clearScreen: false,
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
});
