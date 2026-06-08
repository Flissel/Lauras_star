# Laura ↔ VibeMind-OS Integration — Complete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Laura (editorial editor) into VibeMind-OS as a submodule of the video-production space: backend as a managed service, `video.*` editorial events via the existing VideoAgent, and Laura's UI in the Tauri launcher.

**Architecture:** Four layers built in order — (1) submodule + backend service, (2) Tauri bridge shim + Laura webview window, (3) `laura_tools` + VideoAgent + registry events, (4) media flow. Laura's renderer stays unchanged; only its `window.laura` bridge gets a Tauri implementation.

**Tech Stack:** Python 3.11 (httpx, pytest) for tools/agent; YAML registry; Rust/Tauri 2 (launcher window + shell plugin); PowerShell (start script); TypeScript (Laura bridge shim, vitest).

**Two repos:**
- **Laura** = `C:\Users\User\Desktop\Laura` (remote `github.com/Vibemind-LAB/Lauras_star`).
- **VibeMind-OS** = `C:\Users\User\Desktop\Vibemind_V1\vibemind-os` (the integration host).

**Spec:** [`docs/superpowers/specs/2026-06-03-vibemind-laura-integration-design.md`](../specs/2026-06-03-vibemind-laura-integration-design.md)

**Commands:** Python tools tested from `vibemind-os/voice/python` (its venv). Laura TS from `apps/desktop` with `npm --prefix apps/desktop`. Each task says which repo + cwd. Commit in the repo the files live in. If `No space left on device`, STOP and report BLOCKED.

---

## File Structure

**VibeMind-OS:**
- `.gitmodules` + `spaces/video/laura/` — new submodule (Layer 1).
- `scripts/vibemind-start.ps1` — start Laura backend + media_server (Layer 1).
- `voice/python/spaces/video/tools/laura_tools.py` — httpx tools → Laura API (Layer 3).
- `voice/python/spaces/video/agents/video_agent.py` — extend EVENT_TO_TOOL/PARAM_MAPPING/_load_tools (Layer 3).
- `config/space_agent_registry.yml` — new `video.*` editorial events (Layer 3).
- `launcher-app/src-tauri/src/lib.rs` — `open_laura` WebviewWindow command (Layer 2).
- `launcher-app/src-tauri/capabilities/` — dialog/fs capabilities for the Laura window (Layer 2).
- `launcher-app/src/index.html` — "Open Laura" button (Layer 2).

**Laura:**
- `apps/desktop/src/bridge/lauraBridge.ts` — shared `LauraBridge` interface (extracted) (Layer 2).
- `apps/desktop/src/bridge/tauriBridge.ts` — Tauri implementation of the bridge (Layer 2).
- `apps/desktop/src/bridge/tauriBridge.test.ts` — unit test of the path/url helpers (Layer 2).
- `apps/desktop/vite.tauri.config.ts` + build wiring — static renderer build for Tauri (Layer 2).
- `services/local-api/` — a documented headless start entrypoint (Layer 1; may already exist as `laura-api`).

---

# PHASE 1 — Submodule + Backend Service (VibeMind-OS)

## Task 1.1: Confirm submodule location + add Laura as a submodule

**Repo:** VibeMind-OS. **Files:** `.gitmodules`, `spaces/video/laura/`.

- [ ] **Step 1: Discover the canonical video-space submodule layout**
Run (from `C:\Users\User\Desktop\Vibemind_V1\vibemind-os`):
```
git config -f .gitmodules --get-regexp path | grep -i video
git ls-files --stage | grep -i "spaces/video"
```
Confirm: vibevideo is registered as `spaces/video/vibevideo` at the OS root. Note whether `voice/python/spaces/video` is a copy/sync of `spaces/video` (the earlier finding: identical `media_server.py`). Record the canonical path for new video submodules. Expected: new submodules register at OS-root `spaces/video/<name>`.

- [ ] **Step 2: Add Laura as a submodule**
```
git submodule add https://github.com/Vibemind-LAB/Lauras_star.git spaces/video/laura
git submodule update --init spaces/video/laura
```
Expected: `.gitmodules` gains a `[submodule "spaces/video/laura"]` entry; `spaces/video/laura/` populated.

- [ ] **Step 3: If `voice/python/spaces/video` is a separate working copy, mirror the submodule there too**
If Step 1 showed the agent consumes `voice/python/spaces/video/`, ensure `voice/python/spaces/video/laura` resolves (submodule add there as well, or a sync). If they are the same tree via a link, skip. Report which case applied.

- [ ] **Step 4: Commit**
```
git add .gitmodules spaces/video/laura
git status   # only .gitmodules + the gitlink
git commit -m "$(printf 'feat(spaces): add Laura editorial editor as a video-space submodule\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

## Task 1.2: Headless backend start entrypoint (Laura)

**Repo:** Laura. **Files:** `services/local-api` (verify/створ entrypoint).

- [ ] **Step 1: Verify Laura's backend can start headless on a fixed host/port**
From `C:\Users\User\Desktop\Laura\services\local-api`:
```
uv run --no-sync python -c "from laura.main import create_app; from laura.config import Settings; print('ok', create_app(Settings()).title if hasattr(create_app(Settings()),'title') else 'app')"
```
Confirm the app builds without GPU/models (Laura invariant). Identify the console-script/uvicorn entry (e.g. `laura-api` in `pyproject.toml [project.scripts]`, or `uvicorn laura.main:app`). Record the exact start command + default host/port (127.0.0.1:8765) + how the token is set (env `LAURA_TOKEN` or none).

- [ ] **Step 2: Document the start command**
Append to Laura `docs/06-storage.md` (or a new `docs/embedding.md`) a short "Headless start (for embedding)" note with the exact command and env. Commit (Laura repo):
```
git add docs/embedding.md
git commit -m "$(printf 'docs: headless backend start command for embedding\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

## Task 1.3: Start Laura backend from the VibeMind start script

**Repo:** VibeMind-OS. **Files:** `scripts/vibemind-start.ps1`.

- [ ] **Step 1: Read the current start script**
Read `scripts/vibemind-start.ps1` to learn how existing services are launched (background jobs, logging, health waits), and whether `media_server.py` is already started there.

- [ ] **Step 2: Add a Laura-backend start block (match the script's existing style)**
Add a block that starts Laura's backend headless on 127.0.0.1:8765 in the background, mirroring how the script starts other Python services (use the exact command from Task 1.2, the script's job/logging idiom, and a `MEDIA_SERVER`/media start if not already present). Example shape — adapt to the script's actual conventions:
```powershell
# --- Laura editorial backend (video space editor) ---
$lauraDir = Join-Path $RepoRoot "spaces\video\laura\services\local-api"
Start-LauraBackend -Dir $lauraDir -BindHost "127.0.0.1" -Port 8765   # use the script's existing helper/idiom
```
If the script has no helper, replicate the existing inline pattern other services use (e.g. `Start-Process`/`Start-Job` + log redirect).

- [ ] **Step 3: Manual verify (MANUAL — document, don't fully automate)**
Run the start script (or just the Laura block) and confirm `curl http://127.0.0.1:8765/healthz` returns OK. Record result.

- [ ] **Step 4: Commit**
```
git add scripts/vibemind-start.ps1
git commit -m "$(printf 'feat(start): launch Laura editorial backend with the stack\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

# PHASE 2 — Tauri Bridge Shim + Laura Webview Window

## Task 2.1: Extract the `LauraBridge` interface (Laura)

**Repo:** Laura. **Files:** `apps/desktop/src/bridge/lauraBridge.ts`, `apps/desktop/src/preload.ts`, `apps/desktop/src/global.d.ts`.

- [ ] **Step 1: Read `preload.ts` + `global.d.ts`** to capture the exact `LauraBridge` shape (method names + signatures): `getServiceInfo`, `pickMediaFile`, `pickMediaFiles`, `pickFolder`, `listMediaInFolder`, `pathForFile`, `saveTextFile`.

- [ ] **Step 2: Create the shared interface** `apps/desktop/src/bridge/lauraBridge.ts`:
```ts
import type { ServiceInfo } from "../shared/ipc";

export interface LauraBridge {
  getServiceInfo(): Promise<ServiceInfo | null>;
  pickMediaFile(): Promise<string | null>;
  pickMediaFiles(): Promise<string[]>;
  pickFolder(): Promise<string | null>;
  listMediaInFolder(folder: string): Promise<string[]>;
  pathForFile(file: File): string;
  saveTextFile(defaultName: string, content: string): Promise<string | null>;
}
```
(Match the ACTUAL signatures from Step 1; adjust if any differ.)

- [ ] **Step 3: Make `global.d.ts` use the shared interface** — change `window.laura` typing to `LauraBridge` imported from `./bridge/lauraBridge`. Keep the Electron `preload.ts` `bridge` object as-is but assert it satisfies `LauraBridge` (`const bridge: LauraBridge = {…}` or `satisfies LauraBridge`).

- [ ] **Step 4: Typecheck + test + commit**
```
npm --prefix apps/desktop run typecheck
npm --prefix apps/desktop test
git add apps/desktop/src/bridge/lauraBridge.ts apps/desktop/src/global.d.ts apps/desktop/src/preload.ts
git commit -m "$(printf 'refactor(desktop): extract shared LauraBridge interface\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

## Task 2.2: Tauri bridge implementation (Laura)

**Repo:** Laura. **Files:** `apps/desktop/src/bridge/tauriBridge.ts`, `tauriBridge.test.ts`. **Deps:** add `@tauri-apps/api`, `@tauri-apps/plugin-dialog`, `@tauri-apps/plugin-fs` to `apps/desktop` devdeps (offline-safe: only if install possible; else mark BLOCKED on deps).

- [ ] **Step 1: Write the failing unit test** for the pure helpers (the parts not needing Tauri runtime) — `apps/desktop/src/bridge/tauriBridge.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { mediaExtFilter, isMediaPath } from "./tauriBridge";

describe("tauri bridge helpers", () => {
  it("filters media file paths by extension", () => {
    expect(["/a/x.mp4", "/a/y.txt", "/a/z.wav"].filter(isMediaPath)).toEqual(["/a/x.mp4", "/a/z.wav"]);
  });
  it("exposes a dialog ext filter list", () => {
    expect(mediaExtFilter()).toContain("mp4");
  });
});
```

- [ ] **Step 2: Run → fail.** `npm --prefix apps/desktop test -- tauriBridge`.

- [ ] **Step 3: Implement** `apps/desktop/src/bridge/tauriBridge.ts`:
```ts
import { open as openDialog, save as saveDialog } from "@tauri-apps/plugin-dialog";
import { readDir, writeTextFile } from "@tauri-apps/plugin-fs";

import type { ServiceInfo } from "../shared/ipc";
import type { LauraBridge } from "./lauraBridge";

const MEDIA_EXTS = ["mp4","mov","mkv","m4v","avi","webm","mxf","mpg","mpeg","wav","aif","aiff","flac","mp3","m4a","aac"];

export function mediaExtFilter(): string[] {
  return MEDIA_EXTS;
}
export function isMediaPath(p: string): boolean {
  const dot = p.lastIndexOf(".");
  return dot >= 0 && MEDIA_EXTS.includes(p.slice(dot + 1).toLowerCase());
}

// The launcher injects the backend coordinates here before loading the window.
declare global {
  interface Window { __LAURA_SERVICE__?: ServiceInfo }
}

export function createTauriBridge(): LauraBridge {
  return {
    async getServiceInfo() {
      return window.__LAURA_SERVICE__ ?? null;
    },
    async pickMediaFile() {
      const r = await openDialog({ multiple: false, filters: [{ name: "Media", extensions: mediaExtFilter() }] });
      return typeof r === "string" ? r : null;
    },
    async pickMediaFiles() {
      const r = await openDialog({ multiple: true, filters: [{ name: "Media", extensions: mediaExtFilter() }] });
      return Array.isArray(r) ? r : r ? [r] : [];
    },
    async pickFolder() {
      const r = await openDialog({ directory: true, multiple: false });
      return typeof r === "string" ? r : null;
    },
    async listMediaInFolder(folder: string) {
      const entries = await readDir(folder);
      return entries
        .filter((e) => e.isFile && isMediaPath(e.name))
        .map((e) => `${folder}/${e.name}`);
    },
    pathForFile(_file: File): string {
      // In Tauri the file-drop event delivers OS paths directly; DataTransfer.files
      // has no path. The renderer's drop handler uses the Tauri drop event instead;
      // this method is unused under Tauri and returns "" as a guarded fallback.
      return "";
    },
    async saveTextFile(defaultName: string, content: string) {
      const path = await saveDialog({ defaultPath: defaultName });
      if (!path) return null;
      await writeTextFile(path, content);
      return path;
    },
  };
}
```

- [ ] **Step 4: Run → pass.** `npm --prefix apps/desktop test -- tauriBridge` → 2 passed.

- [ ] **Step 5: Bootstrap selection** — in the renderer entry (`apps/desktop/src/main.tsx` or wherever `window.laura` is first used), if `window.laura` is undefined but `window.__TAURI__` exists, set `window.laura = createTauriBridge()`. (Electron keeps its preload `window.laura`; Tauri gets the shim.) Add this guard in a small `apps/desktop/src/bridge/install.ts` imported first.

- [ ] **Step 6: Typecheck + commit**
```
npm --prefix apps/desktop run typecheck
git add apps/desktop/src/bridge/tauriBridge.ts apps/desktop/src/bridge/tauriBridge.test.ts apps/desktop/src/bridge/install.ts apps/desktop/package.json
git commit -m "$(printf 'feat(desktop): Tauri implementation of the Laura bridge\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

## Task 2.3: Tauri file-drop in the DropZone (Laura)

**Repo:** Laura. **Files:** `apps/desktop/src/components/DropZone.tsx`.

- [ ] **Step 1: Add a Tauri drop path** — when `window.__TAURI__` exists, subscribe to Tauri's `tauri://file-drop` event (via `@tauri-apps/api/event` `listen`) to receive OS paths, and route them through the existing `classifyDrop`/`onImport` logic; keep the Electron `pathForFile` path for Electron. Guard by runtime so both work. (Implement minimally; verified by typecheck + manual smoke since drag-drop is not unit-testable headless.)

- [ ] **Step 2: Typecheck + commit**
```
npm --prefix apps/desktop run typecheck && npm --prefix apps/desktop test
git add apps/desktop/src/components/DropZone.tsx
git commit -m "$(printf 'feat(desktop): Tauri file-drop support in DropZone\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

## Task 2.4: Static renderer build for Tauri (Laura)

**Repo:** Laura. **Files:** `apps/desktop/vite.tauri.config.ts`, `apps/desktop/package.json` (script).

- [ ] **Step 1: Add a renderer-only static build** that outputs `apps/desktop/dist-tauri/` (HTML/JS/CSS, no Electron main/preload). Reuse the existing Vite renderer config; set `base: "./"` so assets load from a `tauri://` or file origin. Add script `"build:tauri": "vite build -c vite.tauri.config.ts"`.

- [ ] **Step 2: Build + verify output**
```
npm --prefix apps/desktop run build:tauri
test -f apps/desktop/dist-tauri/index.html && echo OK
```

- [ ] **Step 3: Commit**
```
git add apps/desktop/vite.tauri.config.ts apps/desktop/package.json
git commit -m "$(printf 'build(desktop): static renderer build for Tauri embedding\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

## Task 2.5: `open_laura` Tauri command + window (VibeMind-OS)

**Repo:** VibeMind-OS. **Files:** `launcher-app/src-tauri/src/lib.rs`, `launcher-app/src-tauri/capabilities/`, `launcher-app/src-tauri/Cargo.toml`, `launcher-app/src/index.html`.

- [ ] **Step 1: Read `lib.rs`** (the `start_stack` command + `invoke_handler` + `Builder`) to match the exact Tauri 2 idiom used.

- [ ] **Step 2: Add an `open_laura` command** that creates a `WebviewWindow` loading Laura's built renderer, and injects the backend coordinates. Use Tauri 2 `WebviewWindowBuilder`. The renderer is served either from Laura's static build (a `tauri://laura` asset dir) or by pointing at the running Vite/static server; for v1 point at a local static file URL of `spaces/video/laura/apps/desktop/dist-tauri/index.html` (or a small static server on a fixed port). Inject `window.__LAURA_SERVICE__` via an `initialization_script`:
```rust
#[tauri::command]
fn open_laura(app: tauri::AppHandle) -> Result<(), String> {
    let url = /* file/asset URL to dist-tauri/index.html */;
    let init = r#"window.__LAURA_SERVICE__ = { baseUrl: "http://127.0.0.1:8765", token: "" };"#;
    tauri::WebviewWindowBuilder::new(&app, "laura", tauri::WebviewUrl::External(url.parse().unwrap()))
        .title("Laura — Editorial")
        .initialization_script(init)
        .inner_size(1280.0, 820.0)
        .build()
        .map_err(|e| e.to_string())?;
    Ok(())
}
```
Add `open_laura` to `invoke_handler![…]`. Adapt URL resolution + token source to the repo's conventions; if the renderer must be bundled as a Tauri asset, configure that in `tauri.conf.json` instead.

- [ ] **Step 3: Capabilities** — add a `capabilities/laura.json` granting the Laura window `dialog:allow-open`, `dialog:allow-save`, `fs:allow-read-dir`, `fs:allow-write-text-file` (the permissions the bridge needs). Add the matching plugins to `Cargo.toml` (`tauri-plugin-dialog`, `tauri-plugin-fs`) and `.plugin(...)` in the `Builder`.

- [ ] **Step 4: UI button** — in `launcher-app/src/index.html`, add an "Open Laura" button that calls `invoke('open_laura')` (the file already uses `@tauri-apps/api/core invoke` per the lib.rs comment).

- [ ] **Step 5: Manual verify (MANUAL)** — `tauri dev` in `launcher-app`, click "Open Laura": the window opens, loads the renderer, and Laura talks to `:8765`. Record result + any capability errors.

- [ ] **Step 6: Commit**
```
git add launcher-app/src-tauri/src/lib.rs launcher-app/src-tauri/capabilities launcher-app/src-tauri/Cargo.toml launcher-app/src/index.html
git commit -m "$(printf 'feat(launcher): open_laura webview window + capabilities\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

# PHASE 3 — laura_tools + VideoAgent + Registry (VibeMind-OS)

## Task 3.1: `laura_tools.py`

**Repo:** VibeMind-OS. **Files:** `voice/python/spaces/video/tools/laura_tools.py`, `voice/python/spaces/video/tools/test_laura_tools.py`. **cwd:** `voice/python` (its venv/pytest).

- [ ] **Step 1: Write the failing test** (mock httpx) — `test_laura_tools.py`:
```python
from unittest.mock import patch, MagicMock
from spaces.video.tools.laura_tools import laura_import, laura_status

def _resp(json_obj, ok=True):
    m = MagicMock(); m.json.return_value = json_obj; m.raise_for_status.return_value = None
    m.status_code = 200 if ok else 500
    return m

def test_laura_import_posts_source_url():
    with patch("spaces.video.tools.laura_tools._client") as c:
        c.post.return_value = _resp({"asset_id": "a1", "job_id": "j1"})
        out = laura_import(project_id="p1", source="http://x/y.mp4")
    assert out["success"] is True
    assert out["data"]["asset_id"] == "a1"
    c.post.assert_called_once()

def test_laura_status_reads_import_status():
    with patch("spaces.video.tools.laura_tools._client") as c:
        c.get.return_value = _resp({"phase": "downloading", "downloaded_bytes": 5, "total_bytes": 10})
        out = laura_status(asset_id="a1")
    assert out["data"]["phase"] == "downloading"
```

- [ ] **Step 2: Run → fail.** From `voice/python`: `python -m pytest spaces/video/tools/test_laura_tools.py -v` (use the repo's test runner). Expected: import error (module missing).

- [ ] **Step 3: Implement** `laura_tools.py`:
```python
"""Laura editorial tools — call Laura's FastAPI (the video-space editor)."""
from __future__ import annotations
import os
from typing import Any, Dict
import httpx

LAURA_BASE = os.environ.get("LAURA_BASE_URL", "http://127.0.0.1:8765")
LAURA_TOKEN = os.environ.get("LAURA_TOKEN", "")
_client = httpx.Client(base_url=LAURA_BASE, timeout=30.0,
                       headers={"X-Laura-Token": LAURA_TOKEN} if LAURA_TOKEN else {})

def _ok(data: Any, msg: str = "ok") -> Dict[str, Any]:
    return {"success": True, "message": msg, "data": data}
def _err(msg: str) -> Dict[str, Any]:
    return {"success": False, "message": msg, "data": None}

def laura_import(project_id: str, source: str) -> Dict[str, Any]:
    """Import a URL or local path into a Laura project."""
    body = {"source_url": source} if "://" in source else {"source_path": source}
    try:
        r = _client.post(f"/projects/{project_id}/assets/import", json=body)
        r.raise_for_status()
        return _ok(r.json(), "import queued")
    except httpx.HTTPError as e:
        return _err(f"laura import failed: {e}")

def laura_status(asset_id: str) -> Dict[str, Any]:
    try:
        r = _client.get(f"/assets/{asset_id}/import-status")
        r.raise_for_status()
        return _ok(r.json())
    except httpx.HTTPError as e:
        return _err(f"laura status failed: {e}")

def laura_export(timeline_id: str, fmt: str = "otio") -> Dict[str, Any]:
    try:
        r = _client.post(f"/timelines/{timeline_id}/exports", json={"format": fmt})
        r.raise_for_status()
        return _ok(r.json(), "export started")
    except httpx.HTTPError as e:
        return _err(f"laura export failed: {e}")
```
(Confirm the export endpoint path against Laura's `api/timelines.py`; the import + import-status paths are from Laura's implemented API.)

- [ ] **Step 4: Run → pass.** `python -m pytest spaces/video/tools/test_laura_tools.py -v` → 2 passed.

- [ ] **Step 5: Commit** (VibeMind-OS)
```
git add voice/python/spaces/video/tools/laura_tools.py voice/python/spaces/video/tools/test_laura_tools.py
git commit -m "$(printf 'feat(video): laura_tools calling Laura editorial API\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

## Task 3.2: Wire Laura tools into the VideoAgent

**Repo:** VibeMind-OS. **Files:** `voice/python/spaces/video/agents/video_agent.py`.

- [ ] **Step 1: Write the failing test** — `voice/python/spaces/video/agents/test_video_agent_laura.py`:
```python
from spaces.video.agents.video_agent import get_video_agent

def test_video_agent_maps_editorial_events():
    a = get_video_agent()
    assert a._get_tool_name("video.import") == "laura_import"
    assert a._get_tool_name("video.export") == "laura_export"
    tools = a._load_tools()
    assert "laura_import" in tools and "laura_export" in tools
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Extend `video_agent.py`** — add to `EVENT_TO_TOOL`:
```python
        "video.import":   "laura_import",
        "video.status_ed":"laura_status",
        "video.export":   "laura_export",
```
add to `PARAM_MAPPING`:
```python
        "video.import": {"link": "source", "url": "source", "projekt": "project_id", "project": "project_id"},
        "video.export": {"format": "fmt", "ziel": "fmt", "timeline": "timeline_id"},
```
and in `_load_tools()` add the laura import + dict entries:
```python
            from spaces.video.tools.laura_tools import laura_import, laura_status, laura_export
            tools.update({"laura_import": laura_import, "laura_status": laura_status, "laura_export": laura_export})
```
(Place inside the existing try/except; if Laura tools fail to import, the agent still loads the rest.)

- [ ] **Step 4: Run → pass.** Run the new test + the existing agent tests → green.

- [ ] **Step 5: Commit**
```
git add voice/python/spaces/video/agents/video_agent.py voice/python/spaces/video/agents/test_video_agent_laura.py
git commit -m "$(printf 'feat(video): route video.import/export to Laura tools\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

## Task 3.3: Registry events

**Repo:** VibeMind-OS. **Files:** `config/space_agent_registry.yml`.

- [ ] **Step 1: Read the `video:` space block** in `space_agent_registry.yml` (events schema: `event: { tool, required_params }`).

- [ ] **Step 2: Add the editorial events** under the `video:` space `events:`:
```yaml
      video.import:   { tool: laura_import, required_params: [project_id, source] }
      video.status_ed:{ tool: laura_status, required_params: [asset_id] }
      video.export:   { tool: laura_export, required_params: [timeline_id] }
```

- [ ] **Step 3: Validate the registry loads** — run the loader (`voice/python/swarm/routing/space_agent_registry.py`) or its test, confirming the new events parse and map to tools. Record the command + result.

- [ ] **Step 4: Commit**
```
git add config/space_agent_registry.yml
git commit -m "$(printf 'feat(registry): video.import/export editorial events -> Laura\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

# PHASE 4 — Media Flow + Polish

## Task 4.1: Route Laura exports into the Rowboat media root

**Repo:** Laura (config) + VibeMind-OS (tool). **Files:** Laura export output dir; `laura_tools.py` (a `laura_publish`).

- [ ] **Step 1: Decide the hand-off** — Laura exports land in its `workspace/.../exports`. Add a `laura_publish(timeline_id, fmt)` tool that exports then copies/moves the result into `~/.rowboat/Videos/` (same `MEDIA_ROOT` the Gallery + media_server serve). Implement in `laura_tools.py`:
```python
import shutil
from pathlib import Path
MEDIA_ROOT = Path.home() / ".rowboat" / "Videos"

def laura_publish(timeline_id: str, fmt: str = "mp4") -> Dict[str, Any]:
    res = laura_export(timeline_id, fmt)
    if not res["success"]:
        return res
    out_path = res["data"].get("path")  # confirm Laura's export response field
    if not out_path:
        return _err("no export path returned")
    MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    dest = MEDIA_ROOT / Path(out_path).name
    shutil.copy2(out_path, dest)
    return _ok({"published": str(dest)}, "published to media root")
```

- [ ] **Step 2: Test** (mock laura_export + filesystem in tmp) and add `video.publish_ed: "laura_publish"` to the agent + registry (mirroring Task 3.2/3.3). Commit.

## Task 4.2: End-to-end verification (MANUAL)

- [ ] Start the stack (`vibemind-start.ps1`) → Laura backend up on :8765.
- [ ] Launcher → "Open Laura" → window loads, import a URL, see progress, build a rough cut, export.
- [ ] Fire `video.import {project_id, source}` via the event path (voice or a test publisher to `events:tasks:video`) → asset appears in Laura.
- [ ] `video.publish_ed` → file shows up under `~/.rowboat/Videos` and streams via media_server :8977.
- [ ] Record results; note anything for follow-up specs.

---

## Notes / cross-cutting
- **Tests:** Python tools/agent/registry are unit-tested (mocked httpx); the Tauri/Rust window, start script, and drag-drop are **manual** (not headless-testable). The TS bridge helpers are unit-tested.
- **Electron stays intact:** Laura keeps its Electron path; the Tauri bridge is additive (runtime-selected).
- **Open items deferred to follow-up specs:** full `video.*` editorial event set (roughcut/cut/assemble as the pipeline stages land), token/secret hardening for the injected `__LAURA_SERVICE__`, bundling the renderer as a Tauri asset vs. external URL, and reconciling the `spaces/video` vs `voice/python/spaces/video` duplication.
