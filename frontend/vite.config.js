import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // During development the API runs on its own port; /api/predict -> :8000/predict
      "/api": {
        target: "http://localhost:8000",
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  // Same output folder as Create React App, so existing deploy settings keep working
  build: { outDir: "build" },
});
