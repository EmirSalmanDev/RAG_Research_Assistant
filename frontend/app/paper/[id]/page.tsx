import type { Metadata } from "next";

import { PaperView } from "@/components/paper/paper-view";
import { decodeRouteParam } from "@/lib/links";

export const metadata: Metadata = { title: "Paper | AI Research Copilot" };

export default async function PaperPage(props: PageProps<"/paper/[id]">) {
  const { id } = await props.params;
  return <PaperView id={decodeRouteParam(id)} />;
}
