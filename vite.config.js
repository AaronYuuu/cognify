import { defineConfig } from 'vite'
import { resolve } from 'path'
import viteCompression from 'vite-plugin-compression'

export default defineConfig({
  root: 'atrophy',
  base: '/',
  publicDir: '../public',
  plugins: [
    // Generate gzip compressed files
    viteCompression({
      algorithm: 'gzip',
      ext: '.gz',
      threshold: 1024,
      deleteOriginFile: false
    }),
    // Generate brotli compressed files (better compression than gzip)
    viteCompression({
      algorithm: 'brotliCompress',
      ext: '.br',
      threshold: 1024,
      deleteOriginFile: false
    })
  ],
  server: {
    open: true
  },
  preview: {
    port: 8088
  },
  build: {
    target: 'es2015',
    minify: 'esbuild',
    outDir: '../dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 2000,
    rollupOptions: {
      output: {
        manualChunks(id) {
          // Split TensorFlow.js into its own chunk (largest dependency)
          if (id.includes('@tensorflow/tfjs')) {
            return 'vendor-tf';
          }
          // NiiVue visualization library
          if (id.includes('@niivue/niivue')) {
            return 'vendor-niivue';
          }
          // 3D math utilities
          if (id.includes('gl-matrix')) {
            return 'vendor-math';
          }
          // Compression libraries
          if (id.includes('blosc') || id.includes('lz4') || id.includes('zstd')) {
            return 'vendor-compression';
          }
          // Other vendor libraries
          if (id.includes('node_modules')) {
            return 'vendor';
          }
        }
      }
    }
  },
  esbuild: {
    treeShaking: true
  }
})
