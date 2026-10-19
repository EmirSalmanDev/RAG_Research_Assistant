import type { Metadata } from "next";

import { DigestSettings } from "@/components/settings/digest-settings";
import { InterestsSection } from "@/components/settings/interests-section";

export const metadata: Metadata = { title: "Settings | AI Research Copilot" };

export default function SettingsPage() {
  return (
    <div className="mx-auto w-full max-w-3xl space-y-10 px-4 py-8">
      <h1 className="text-3xl font-semibold tracking-tight">Settings</h1>
      <InterestsSection />
      <section aria-labelledby="digest-settings-heading" className="space-y-4">
        <h2 id="digest-settings-heading" className="text-xl font-semibold">
          Digest and fetching
        </h2>
        <DigestSettings />
      </section>
    </div>
  );
}
