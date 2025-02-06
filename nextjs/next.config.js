/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    // Add this to allow connections from any host
    async rewrites() {
      return [
        {
          source: '/api/:path*',
          destination: 'http://backend:8005/api/:path*',
        },
      ];
    },
  };
  
  module.exports = nextConfig;

  