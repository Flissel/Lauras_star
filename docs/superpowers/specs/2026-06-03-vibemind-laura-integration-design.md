# Laura ↔ VibeMind-OS — Integrations-Design

- **Datum:** 2026-06-03
- **Status:** Entwurf (zur Review)
- **Betrifft:** `vibemind-os` (Submodul, VideoAgent, Tools, Registry, Tauri-Launcher) **und** `Laura` (Tauri-Bridge-Shim, Renderer-Build)
- **Repos:** Laura = `github.com/Vibemind-LAB/Lauras_star` · VibeMind-OS lokal unter `Vibemind_V1/vibemind-os`

## Ziel

Laura (frame-genauer Editorial-Editor: Electron-Renderer + FastAPI `:8765`) **voll** in
VibeMind-OS integrieren: als **Submodul-Space**, mit **Backend als verwaltetem Service**,
**`video.*`-Events** über den bestehenden VideoAgent, und **Lauras UI im Tauri-Launcher**
(eigenes Webview-Fenster). vibevideo = *generieren*, Laura = *editieren* — beide am selben
Video-Space.

## Entschiedene Weichen (Brainstorming)

1. **Video-Space erweitern** (kein eigener `editorial`-Space): Laura-Tools + `video.*`-Events
   in den bestehenden `VideoAgent`.
2. **Bau-Reihenfolge:** zuerst **Submodul + Backend-Service**, dann Tauri-UI/Bridge, dann
   Events/Tools, dann Medien-Politur.

## Architektur

```
VibeMind-Launcher (Tauri, statisches index.html + Tauri-Commands)
   │  startet/verwaltet
   ├─► Laura-Backend  (FastAPI :8765)        ── Space-Service (managed, headless-fähig)
   ├─► Laura-Webview-Fenster (Tauri)         ── lädt Lauras gebauten React-Renderer
   │       └─ window.laura  ⇒  Tauri-Shim (dialog/fs/path/file-drop + baseUrl:8765)
   └─► VideoAgent  (Redis stream events:tasks:video)
           └─ laura_tools.py  → Laura-FastAPI  (video.import / video.roughcut / video.export …)
   Medien: ~/.rowboat/Videos  ⇄  media_server :8977 (Range-Streaming)  /  Laura-Proxy-HTTP
```

Bestehendes Muster, das wir nutzen: `space_agent_registry.yml` (event→tool), `BaseBackendAgent`
(`stream`, `EVENT_TO_TOOL`, `PARAM_MAPPING`, `_load_tools`), `media_server.py`.

## Ebene 1 — Submodul + Backend-Service *(erstes Inkrement)*

- **Submodul:** `git submodule add https://github.com/Vibemind-LAB/Lauras_star.git spaces/video/laura`
  (im `voice`-Submodul-Pfad bzw. dem kanonischen `spaces/video/`, konsistent zu vibevideo).
- **Start-Hook:** ein Tool/Skript, das Lauras Backend startet
  (`uv run --project services/local-api laura-api` o. ä. auf `127.0.0.1:8765`). Nutzt Lauras
  Invariante „Backend startet ohne GPU/Modelle" → läuft headless im VibeMind-Kontext.
- **Health/Port:** Launcher/Agent kennt baseUrl+Token (Lauras `/healthz`, Token-Header).
- **Ergebnis:** Laura läuft als Service im VibeMind-Kontext (noch ohne UI-/Event-Verdrahtung).

## Ebene 2 — Tauri-Bridge-Shim + Laura-Webview-Fenster

Der Launcher öffnet ein **`WebviewWindow` „Laura"**, das Lauras gebauten Renderer lädt.
Lauras Renderer bleibt **unverändert** — nur die Preload-Bridge bekommt eine **Tauri-Variante**:

| Electron `window.laura` heute | Tauri-Ersatz |
|---|---|
| `getServiceInfo()` (baseUrl/Token) | Launcher injiziert baseUrl+Token (init-Script/Tauri-state) |
| `pickMediaFile()` / `pickMediaFiles()` / `pickFolder()` | `@tauri-apps/plugin-dialog` `open({multiple, directory})` |
| `listMediaInFolder(path)` | `@tauri-apps/plugin-fs` `readDir` + Endungs-Filter |
| `pathForFile(file)` (Drop) | Tauri **file-drop-Event** liefert OS-Pfade direkt |
| `saveTextFile(name, content)` | dialog `save` + fs `writeTextFile` |
| `laura-media://…` (Proxy-Streaming) | Laura-Backend-HTTP-Proxy (`/assets/{id}/files/proxy`) **oder** media_server :8977 |

- **Renderer-Build:** `vite build` der Laura-Renderer → statisches `dist`; das Tauri-Fenster
  lädt es (Tauri-`frontendDist`/asset-Protokoll oder lokaler Static-Server).
- **Shim-Form:** eine `laura-bridge`-Implementierung, die `window.laura` mit denselben
  Signaturen auf `window.__TAURI__`-APIs abbildet — so kompiliert Lauras Renderer ohne Änderung
  gegen dieselbe `LauraBridge`-Schnittstelle.
- **Risiko/Notiz:** Electron-`contextIsolation`/IPC entfällt; Tauri-Capabilities (dialog/fs)
  müssen in `tauri.conf`/capabilities freigeschaltet werden.

## Ebene 3 — laura_tools + VideoAgent-Erweiterung + Registry-Events

- **`spaces/video/tools/laura_tools.py`** (httpx → Laura-FastAPI):
  `laura_import(source)`, `laura_status(asset_id)`, `laura_roughcut(project_id)`,
  `laura_cut(scene_id, …)`, `laura_assemble(sequence_id, order)`, `laura_export(sequence_id, format)`.
- **`VideoBackendAgent`** erweitern: `_load_tools()` importiert die Laura-Tools zusätzlich;
  `EVENT_TO_TOOL` bekommt `video.import → laura_import` usw.; `PARAM_MAPPING` für deutsche
  Voice-Args (`„link"/„url" → source`, `„projekt" → project_id`).
- **`space_agent_registry.yml`** (Space `video`): neue `events:`
  `video.import`, `video.roughcut`, `video.cut`, `video.assemble`, `video.export`
  je mit `tool` + `required_params`.

## Ebene 4 — Medien-Fluss

- Laura-Exports/Proxies nach `~/.rowboat/Videos/` (Rowboat-Konvention) → vom vorhandenen
  **media_server :8977** mit Range gestreamt; alternativ direkt Laura-HTTP. Final in der
  Medien-Politur-Stufe festzurren.

## Was ist neu — pro Repo

| Repo | Neu |
|---|---|
| **vibemind-os** | Submodul `spaces/video/laura`; `laura_tools.py`; VideoAgent-EVENT_TO_TOOL/PARAM_MAPPING; Registry-Events; Launcher-Command + WebviewWindow + Start-Hook |
| **Laura** | `laura-bridge` Tauri-Shim (gleiche `LauraBridge`-Signaturen); Renderer-Static-Build-Konfig für Tauri; (Electron-Pfad bleibt parallel nutzbar) |

## Dekomposition / Bau-Reihenfolge (jede Ebene = eigener Spec+Plan)

1. **Submodul + Backend-Service** — Laura läuft im VibeMind-Kontext (Health prüfbar). ← Start.
2. **Tauri-Bridge-Shim + Laura-Webview-Fenster** — UI im Launcher (größter Brocken).
3. **laura_tools + Agent + Registry-Events** — voice/event-steuerbar.
4. **Medien-Fluss + Politur** — Rowboat/media_server, Aufräumen.

## Offene Punkte (in den Ebenen-Specs)

- Exakter Start-Mechanismus von Lauras Backend im VibeMind-Launcher (Sidecar vs. Skript vs.
  Python-Engine-Service) — Ebene-1-Spec.
- Tauri-Asset-Hosting des Renderers (eingebettet vs. lokaler Server) + Capabilities — Ebene-2-Spec.
- Voll-`video.*`-Event-Set + PARAM_MAPPING-Aliasse (deutsch) — Ebene-3-Spec.
- Token/Secret-Übergabe Launcher → Renderer/Tools — Ebene-1/2-Spec.
- vibevideo vs. Laura: gemeinsame vs. getrennte Tool-Namespaces im selben Agent — Ebene-3-Spec.
