export const backendRequirement = "NexuML >=0.2.1,<0.3.0 with API interface 2";

/** @param {{interface_version?:number, nexuml_version?:string}} identity */
export function compatibilityError(identity) {
  const version = /^0\.2\.(0|[1-9]\d*)$/.exec(identity.nexuml_version ?? "");
  return identity.interface_version === 2 && version && Number(version[1]) >= 1 ? null :
    `Studio requires ${backendRequirement}; selected NexuML ${identity.nexuml_version ?? "unknown"}, interface ${identity.interface_version ?? "unknown"}. Select a compatible installation with nexuml-studio --python /path/to/project/.venv/bin/python . or explicitly reinstall the tool: uv tool install --reinstall "nexuml[api,library]>=0.2.1,<0.3.0". Nothing is updated automatically.`;
}
