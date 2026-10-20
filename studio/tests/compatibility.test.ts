import { expect, test } from "vitest";
import { backendRequirement, compatibilityError } from "../bin/compatibility.mjs";

test("Studio supports independent backend patch releases, not unknown interfaces/releases",()=>{
  for(const version of ["0.2.1","0.2.2","0.2.99"])expect(compatibilityError({interface_version:2,nexuml_version:version})).toBeNull();
  for(const version of ["0.2.0","0.1.99","0.3.0","1.0.0","0.2.1rc1","0.2.1.dev1","unknown",""]){
    const error=compatibilityError({interface_version:2,nexuml_version:version});expect(error).toContain(backendRequirement);expect(error).toContain(version);
  }
  expect(compatibilityError({interface_version:3,nexuml_version:"0.2.1"})).toContain("interface 3");
  expect(compatibilityError({})).toContain("unknown");
});
