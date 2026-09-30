import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Self-contained server bundle so the Docker image stays small.
  output: "standalone",
};

export default nextConfig;
