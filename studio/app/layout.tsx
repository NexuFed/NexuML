import type { Metadata } from "next";
import "@fontsource-variable/geist";
import "@fontsource-variable/montserrat";
import "@xyflow/react/dist/style.css";
import "./globals.css";

export const metadata: Metadata = { title: "NexuML Studio", description: "Your installed NexuML, visually." };

export default function Layout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
