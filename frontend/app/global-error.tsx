"use client";

// Replaces the root layout when the layout itself fails, so it brings its own <html> and
// <body> and cannot use the app's components or stylesheet.
export default function GlobalError({
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  return (
    <html lang="en">
      <body style={{ fontFamily: "system-ui, sans-serif", padding: "2rem" }}>
        <h1>AI Research Copilot could not be shown</h1>
        <p>Something failed while building the page.</p>
        <button type="button" onClick={() => retry()}>
          Try again
        </button>
      </body>
    </html>
  );
}
