import type { MetadataRoute } from 'next';

// lets Chrome / Safari install the editor as its own app (own window and Dock icon)
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: 'Neural Edit Pro',
    short_name: 'Neural Edit',
    description: 'Local, private photo editor with scene-aware looks, masks and sky tools',
    start_url: '/',
    display: 'standalone',
    background_color: '#111113',
    theme_color: '#111113',
    icons: [
      { src: '/icon-192.png', sizes: '192x192', type: 'image/png' },
      { src: '/icon-512.png', sizes: '512x512', type: 'image/png' },
    ],
  };
}
