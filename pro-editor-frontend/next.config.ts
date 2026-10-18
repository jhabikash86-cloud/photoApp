import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // the app launcher uses a separate production build, so it never clashes with `npm run dev`
  distDir: process.env.NEP_PROD ? ".next-prod" : ".next",
  /* config options here */
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
