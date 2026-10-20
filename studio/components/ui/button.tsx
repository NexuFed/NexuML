"use client";

import { Button as Primitive } from "@base-ui/react/button";

// shadcn-style local ownership: only the required Base UI primitive, corporate tokens.
export function Button({ className = "", ...props }: React.ComponentProps<typeof Primitive>) {
  return <Primitive className={`button ${className}`} {...props} />;
}
