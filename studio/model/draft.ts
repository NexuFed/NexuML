import { createStore } from "zustand/vanilla";
import { stringify } from "yaml";
import { newSnapshot, signature } from "./graph";
import type { Document, Snapshot } from "./types";

export function yamlText(snapshot: Snapshot): string {
  const stages = new Map(snapshot.order.map(stage => [stage, snapshot.config.pipeline.stages[stage]]));
  return stringify({...snapshot.config, pipeline: {...snapshot.config.pipeline, stages}});
}

export type DraftState = {
  draft: Snapshot | null; saved: string; savedLayout: string; path: string; baseRevision: string | null;
  semanticRevision: string; past: Snapshot[]; future: Snapshot[]; blocked: boolean;
  load: (document: Document, layout?: Pick<Snapshot,"ids"|"positions">) => void;
  change: (snapshot: Snapshot) => void; undo: () => void; redo: () => void;
  markSaved: (document: Document, source?:Snapshot) => void;
  markLayoutSaved: (source:Snapshot) => void; block: (value: boolean) => void;
};

export const layoutSignature=(snapshot:Snapshot)=>JSON.stringify([snapshot.ids,snapshot.positions]);

export function createDraftStore() {
  return createStore<DraftState>((set, get) => ({
    draft: null, saved: "", savedLayout:"", path: "scenario.yaml", baseRevision: null, semanticRevision: "",
    past: [], future: [], blocked: false,
    load: (document, layout) => {
      const draft = newSnapshot(document.data, document.stage_order);
      if (layout?.positions && layout?.ids &&
          document.stage_order.every(stage => Array.isArray(layout.ids[stage]) &&
            layout.ids[stage].length === draft.ids[stage].length && layout.ids[stage].every(id=>typeof id==="string")) &&
          new Set(Object.values(layout.ids).flat()).size === Object.values(layout.ids).flat().length &&
          Object.values(layout.positions).every(position=>position && Number.isFinite(position.x) && Number.isFinite(position.y))) {
        draft.ids = layout.ids; draft.positions = layout.positions;
      }
      set({draft, saved: document.path ? signature(draft) : "",savedLayout:layoutSignature(draft), path: document.path ?? "scenario.yaml",
        baseRevision: document.base_revision ?? null, semanticRevision: document.semantic_revision,
        past: [], future: [], blocked: false});
    },
    change: draft => {
      const state = get();
      if (state.blocked || !state.draft) return;
      if (signature(draft) === signature(state.draft)) { set({draft}); return; }
      set({draft, past: [...state.past, state.draft].slice(-100), future: []});
    },
    undo: () => {
      const state = get();
      if (!state.blocked && state.past.length && state.draft) set({draft: state.past.at(-1)!,
        past: state.past.slice(0,-1), future: [state.draft, ...state.future]});
    },
    redo: () => {
      const state = get();
      if (!state.blocked && state.future.length && state.draft) set({draft: state.future[0],
        future: state.future.slice(1), past: [...state.past,state.draft]});
    },
    markSaved: (document,source) => {
      const draft = source ?? get().draft!;
      set({saved: signature(draft), path: document.path!, baseRevision: document.base_revision!,
        semanticRevision: document.semantic_revision});
    },
    markLayoutSaved: source => set({savedLayout:layoutSignature(source)}),
    block: blocked => set({blocked}),
  }));
}
