/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Inferência em CPU pode passar de 30s (default do proxy) numa página cheia de contas.
  experimental: { proxyTimeout: 180_000 },
  // Proxy /api/* → FastAPI em dev, evitando CORS e hardcode de URL no cliente.
  async rewrites() {
    const apiBase = process.env.API_BASE_URL ?? "http://localhost:8000";
    return [{ source: "/api/:path*", destination: `${apiBase}/:path*` }];
  },
};

export default nextConfig;
