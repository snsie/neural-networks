import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    rolldownOptions: {
      input: {
        explorer: fileURLToPath(new URL('./index.html', import.meta.url)),
        exercise: fileURLToPath(new URL('./exercise.html', import.meta.url)),
      },
    },
  },
});
