import Link from "next/link";

import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const PAGES = [
  {
    href: "/chat",
    title: "Chat",
    text: "Ask a question across the indexed papers. Every claim cites a source.",
  },
  {
    href: "/search",
    title: "Search",
    text: "Find passages and papers by topic, category or date.",
  },
];

export default function Home() {
  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-16">
      <h1 className="text-3xl font-semibold tracking-tight">
        AI Research Copilot
      </h1>
      <p className="mt-3 text-muted-foreground">
        A local research assistant for AI papers: ask, check the cited passage,
        follow the link to arXiv.
      </p>
      <div className="mt-10 grid gap-4 sm:grid-cols-2">
        {PAGES.map(({ href, title, text }) => (
          <Link key={href} href={href} className="group">
            <Card className="h-full transition-colors group-hover:border-primary">
              <CardHeader>
                <CardTitle>{title}</CardTitle>
                <CardDescription>{text}</CardDescription>
              </CardHeader>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
