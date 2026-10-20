"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/", label: "Digest" },
  { href: "/chat", label: "Chat" },
  { href: "/search", label: "Search" },
  { href: "/compare", label: "Compare" },
  { href: "/timeline", label: "Timeline" },
  { href: "/library", label: "Library" },
  { href: "/settings", label: "Settings" },
];

export function SiteNav() {
  const pathname = usePathname();
  return (
    <header className="border-b bg-card">
      <nav
        aria-label="Main"
        className="mx-auto flex h-14 max-w-6xl items-center gap-6 px-4"
      >
        <Link href="/" className="font-semibold tracking-tight">
          AI Research Copilot
        </Link>
        <ul className="flex items-center gap-1 text-sm">
          {LINKS.map(({ href, label }) => {
            const active = pathname === href || pathname.startsWith(`${href}/`);
            return (
              <li key={href}>
                <Link
                  href={href}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "rounded-md px-3 py-1.5 transition-colors hover:bg-accent hover:text-accent-foreground",
                    active && "bg-accent font-medium text-accent-foreground",
                  )}
                >
                  {label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </header>
  );
}
