import type { Metadata } from "next";
import "./globals.css";

// No next/font/google: it downloads font files from Google at dev/build time. The
// font stacks are system fonts, set in globals.css.
export const metadata: Metadata = {
  title: "AI Research Copilot",
  description:
    "Ask questions across AI papers and follow the citations to the source.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
