"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  // One client per browser tab. useState keeps it across re-renders (not module scope,
  // so it is never shared between server renders).
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // /search embeds the question (a paid OpenAI request), so a query is never
            // retried or refetched behind the user's back, and a result stays fresh for
            // five minutes: going back to a page does not ask again.
            retry: false,
            refetchOnWindowFocus: false,
            staleTime: 5 * 60 * 1000,
          },
        },
      }),
  );
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
