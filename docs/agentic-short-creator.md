# Agentic Short-Creator (NL-Agent)

„Mach mir einen 60s-Short über X" → automatischer 9:16-Short aus einem analysierten Video, gebaut
von einem **Multi-Agent-System (AutoGen 0.4)**, das Lauras **bestehende** Fähigkeiten orchestriert
(CLIP-Suche, VLM, Shorts-Scoring, Transcript, Render). Design:
[`docs/superpowers/specs/2026-07-01-nl-agent-short-creator-design.md`](superpowers/specs/2026-07-01-nl-agent-short-creator-design.md).

## Installation (optionales Extra)

```bash
uv sync --extra autoshort
```

Das Backend startet **auch ohne** das Extra — der `auto-short`-Endpoint liefert dann `503`
(feature unavailable). Nichts von AutoGen wird beim Modul-Laden importiert.

## Voraussetzungen

- **Asset analysiert** (Shots + Transcript; für die visuellen Tools zusätzlich `shorts.embed_frames`).
- **Ollama lokal** für die Reasoning-Agenten und den VLM-Describe:
  - Agenten-Modell: z.B. `qwen2.5` (Default `LAURA_AGENT_MODEL`).
  - VLM (Frame-Beschreibung): `qwen3-vl:8b` via `LAURA_VLM_MODEL` (sonst degradiert der Describer
    graceful zu „keine Beschreibung").

## Nutzung

```
POST /assets/{asset_id}/auto-short
{ "topic": "…", "target_seconds": 60 }        →  { "job_id": "…" }
```

Fortschritt über `GET /jobs/{job_id}` pollen. Ergebnis ist eine bearbeitbare Timeline/Rough-Cut +
ein gerenderter 9:16-Short — alles über die bestehenden Tabellen/Endpoints, kein neuer Projektzustand.

## Orchestrierung — die Eskalations-Leiter

- **Stufe A (lokal, gratis, Provider `ollama`):** Magentic-One → bei hartem Fehlschlag automatisch
  GraphFlow (deterministischer Graph: `scout → {describer ∥ transcript} → director → editor → qa`,
  QA-„weak" schleift zurück zum Director).
- **Stufe B (Provider `9router`):** greift, wenn Stufe A **„zu schlecht"** ist — *hart* (beide
  Teams scheitern) → automatisch; *weich* (QA verwirft nach N Runden) → per Default **manuell**
  (du siehst den Stand + QA-Score und flippst selbst), oder automatisch mit
  `LAURA_AGENT_AUTO_ESCALATE=1`.

## Konfiguration (alles über Env, Zero-Config-Default lokal)

| Env | Default | Wirkung |
|---|---|---|
| `LAURA_AGENT_PROVIDER` | `ollama` | Stufe-A-Provider: `ollama` \| `9router` \| `openai-compat` |
| `LAURA_AGENT_MODEL` | `qwen2.5` | Modell der Alltags-Agenten |
| `LAURA_ORCHESTRATOR_MODEL` | = `LAURA_AGENT_MODEL` | Modell des Magentic-One-Orchestrators (darf stärker sein) |
| `LAURA_AGENT_ORCHESTRATION` | `magentic` | `magentic` (primär) \| `graph` (nur GraphFlow erzwingen) |
| `LAURA_AGENT_ESCALATE_PROVIDER` | `9router` | Stufe-B-Provider |
| `LAURA_AGENT_ESCALATE_MODEL` | `cc/claude-sonnet-4-5` | Stufe-B-Modell |
| `LAURA_AGENT_AUTO_ESCALATE` | `0` | `1` = auch weiche Qualität automatisch eskalieren |
| `LAURA_AGENT_QA_MAX_ROUNDS` | `2` | QA-Runden bevor „weich zu schlecht" gilt |
| `LAURA_9ROUTER_BASE_URL` | `http://localhost:20128/v1` | 9router-Endpoint (OpenAI-kompatibel) |
| `LAURA_9ROUTER_API_KEY` | — | 9router-Key (aus dem Dashboard) |
| `LAURA_AGENT_BASE_URL` / `LAURA_AGENT_API_KEY` | — | generischer `openai-compat`-Endpoint |
| `LAURA_VLM_MODEL` | — | VLM fürs Frame-Describe (Ollama, z.B. `qwen3-vl:8b`) |

## 9router (optional, für stärkere/günstigere Orchestrator-Modelle)

[9router](https://github.com/decolua/9router) ist ein selbst-gehostetes, OpenAI-kompatibles Gateway
(`localhost:20128/v1`) zu 40+ Providern. Setup:

```bash
npm install -g 9router      # läuft auf Port 20128
# Key im Dashboard (localhost:20128/dashboard) erzeugen, dann:
export LAURA_AGENT_ESCALATE_PROVIDER=9router
export LAURA_9ROUTER_API_KEY=<key>
export LAURA_AGENT_ESCALATE_MODEL=cc/claude-sonnet-4-5   # oder z.B. GLM (billig)
```

Damit läuft die Stufe-B-Eskalation über 9router; die Stufe-A-Agenten bleiben lokal/gratis auf Ollama.
**Hinweis:** Abo-Reuse (Claude Code / Copilot) via Proxy kann gegen die ToS des Anbieters verstoßen —
vor produktiver Nutzung prüfen.

## Manuell zu verifizieren (echter Lauf)

CI/Sandbox haben **kein** Ollama/AutoGen/Modell — daher ist der echte End-to-End-Lauf manuell:

1. `uv sync --extra autoshort` + Ollama mit `qwen2.5` und `qwen3-vl:8b` (`ollama pull …`).
2. Ein **echtes Sprechvideo** importieren + analysieren (Shots + Transcript + `shorts.embed_frames`).
3. `POST /assets/{id}/auto-short { "topic": "…" }` → `/jobs/{id}` pollen bis `succeeded`.
4. Prüfen: das Ergebnis-Result nennt `stage` (A/B), `team` (magentic/graph), `weak`, `escalated`;
   die erzeugte Timeline öffnen + den 9:16-Short ansehen.

Alle Nicht-Modell-Pfade (Provider-Factory, Tool-Bridge, Roster, Team-Zusammenbau, Eskalations-Logik,
Endpoint/Guard) sind mit gemockten Clients getestet (`tests/test_short_creator_*.py`).

## v2: Produktions-Sessions (Vibe-Editing)

Seit Slice 4 des v2-Umbaus (Spec `docs/superpowers/specs/2026-07-13-agent-vision-short-v2-design.md`)
läuft die Short-Produktion als **Session**: das 5er-Team (Vision-Reviewer, Story-Architekt,
Szenen-Autor, Coding-Agent, QA) arbeitet über das versionierte **Production Board** unter
`<workspace>/agent-runs/<session_id>/board/` — jedes Artefakt (Szenen-Reviews, Storyline, Skript,
Stimme, Cutlist, Render-Report, QA-Report) ist sofort persistiert, versioniert und wiederaufnehmbar.
Folge-Nachrichten bauen nur Downstream neu; „zurück zu Version X" ist ein echter Revert.

```bash
# Session starten (202 → {session_id, job_id}; läuft als Job production.run)
curl -X POST -H "X-Laura-Token: $TOKEN" -H "Content-Type: application/json" \
  -d '{"task": "Virales 60s-Insta-Short über das Overview-Video", "target_seconds": 60}' \
  http://127.0.0.1:8765/assets/<asset_id>/production

# Board-Stand ansehen (Artefakt-Versionen + resume_point)
curl -H "X-Laura-Token: $TOKEN" http://127.0.0.1:8765/production/<session_id>

# Nachjustieren oder zurückspulen (Freitext; das Team interpretiert)
curl -X POST -H "X-Laura-Token: $TOKEN" -H "Content-Type: application/json" \
  -d '{"text": "Kapitel 2 bitte mit einer anderen Szene, und der Hook kürzer"}' \
  http://127.0.0.1:8765/production/<session_id>/message
```

Hinweise: Sessions brauchen das `autoshort`-Extra (sonst 503); Jobs laufen mit `max_attempts=1`;
ein grobes Run-Log liegt unter `agent-runs/<session_id>/runs/`. Free-Tier-Resilienz:
`LAURA_AGENT_MODEL_POOL` / `LAURA_VLM_MODEL_POOL` rotieren bei leeren Antworten sticky auf das
nächste Modell. ChatPanel-Anbindung (Slice 5) folgt.

### Kontaktbogen-Checkpoint (Freigabe vor dem Render)

Zwischen `cutlist` und `render_report` sitzt der **Kontaktbogen** (`contact_sheet`): EIN
Grid-PNG mit dem Fenster-Mittel-Frame jedes Cutlist-Segments (aus dem Editorial-Proxy, als PNG —
nicht mjpeg), Kacheln in Segment-Reihenfolge und beschriftet mit `<order> S<szenennummer>`
(z. B. `2 S5` = drittes Segment, Szene 5; ohne brauchbaren Font degradiert der Bogen zu
unbeschrifteten Kacheln, `labeled: false`). Der Coding-Agent baut ihn **immer** direkt nach
`build_cutlist` und **vor** `render_production`; eine Cutlist-Änderung archiviert den aktuellen
Bogen automatisch (Board-Chain-Invalidierung) und erzwingt einen frischen.

```bash
# Board-Status: artifacts.contact_sheet trägt version, png_path, labeled + Kachel-Liste
curl -H "X-Laura-Token: $TOKEN" http://127.0.0.1:8765/production/<session_id>

# Das PNG selbst (404, solange noch keiner auf dem Board liegt)
curl -H "X-Laura-Token: $TOKEN" \
  http://127.0.0.1:8765/production/<session_id>/contact-sheet -o kontaktbogen.png
```

**Freigabe-Flow ohne neuen Session-Zustand** (bekanntes Muster im Task-Contract): Du steuerst
den Checkpoint rein über `/message` gegen den normalen Resume-Flow — es gibt keinen extra
Approval-Status. Beispiel:

```bash
# 1) Bis zum Kontaktbogen bauen, dann anhalten (der Run endet nach save_contact_sheet)
curl -X POST -H "X-Laura-Token: $TOKEN" -H "Content-Type: application/json" \
  -d '{"text": "bau bis zum Kontaktbogen, dann stopp"}' \
  http://127.0.0.1:8765/production/<session_id>/message

# 2) Bogen ansehen (siehe oben), ggf. nachjustieren ("Kachel 2 S5 raus, Hook kürzer" → neue
#    Cutlist → neuer Bogen) …

# 3) … und erst dann rendern lassen: der Folge-Run setzt am Resume-Punkt (render) auf
curl -X POST -H "X-Laura-Token: $TOKEN" -H "Content-Type: application/json" \
  -d '{"text": "render jetzt"}' \
  http://127.0.0.1:8765/production/<session_id>/message
```
