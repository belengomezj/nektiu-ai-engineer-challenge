import type { Metadata } from 'next';

import './globals.css';

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL ??
      'https://nektibot-rag.srainvisible.chatgpt.site',
  ),
  title: 'NektiBot · Asistente documental',
  description:
    'Consulta el manual de NektiBot con respuestas fundamentadas y fuentes visibles.',
  openGraph: {
    title: 'NektiBot · Asistente documental',
    description: 'Respuestas con fundamento y fuentes visibles.',
    images: ['/og.png'],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'NektiBot · Asistente documental',
    description: 'Respuestas con fundamento y fuentes visibles.',
    images: ['/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
