import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto w-full max-w-3xl space-y-3 px-4 py-8">
      <h1 className="text-2xl font-semibold tracking-tight">Page not found</h1>
      <p className="text-muted-foreground">
        There is no page at this address.{" "}
        <Link href="/" className="underline underline-offset-4">
          Back to the digest
        </Link>
      </p>
    </div>
  );
}
