# Task 1 – Deterministische Materialsuche

## Ergebnis

Implementiert wurde die projektweite, deterministische Auto-Short-Materialsuche. Sie bevorzugt
den optionalen Semantic-Index, fällt bei fehlendem, leerem oder fehlerhaftem Index auf die
lexikalische Transkriptsuche zurück, summiert Treffer-Scores je Asset und ordnet jeden Treffer
einer Rough-Cut-Scene über deren Source-Frame-Range zu. Die Suche erzeugt dabei niemals eine
Timeline.

Commit: `0848866 feat(short-creator): topic discovery ranks material across the whole project`

## Geänderte Dateien

- `services/local-api/src/laura/short_creator/discovery.py`
  - Neue öffentliche Funktion `search_material` mit der vereinbarten Ergebnisform.
  - Read-only Scene-Mapping, Semantic-zu-Lexical-Fallback, Ranking, Snippet-Kappung und
    stabile Szenensortierung.
- `services/local-api/src/laura/db/repos.py`
  - Neuer read-only Helper `get_asset_rough_cut` mit dem bestehenden, neuesten Rough-Cut-SELECT.
  - `get_or_create_asset_rough_cut` verwendet den Helper weiter und behält sein bisheriges
    Schreibverhalten nur für den expliziten Create-Fall.
- `services/local-api/tests/test_discovery.py`
  - Lexical Ranking über mehrere Assets und Scenes, kein Rough-Cut-Write als Seiteneffekt,
    leerer Trefferfall sowie der optionale echte In-Memory-Semantic-Pfad.

## TDD-Nachweis

### RED

Ausgeführt in `services/local-api`:

```powershell
$env:UV_PROJECT_ENVIRONMENT='C:\Users\User\Desktop\Laura\services\local-api\.venv'; uv run --no-sync pytest tests/test_discovery.py -q; echo EXIT=$?
```

Ausgabe (relevanter Teil):

```text
ERROR tests/test_discovery.py
ImportError: cannot import name 'discovery' from 'laura.short_creator'
EXIT=False
```

Der Fehler war erwartet: Das neue Discovery-Modul existierte noch nicht.

### GREEN

Nach minimaler Implementierung und erneut nach dem Formatieren ausgeführt:

```powershell
$env:UV_PROJECT_ENVIRONMENT='C:\Users\User\Desktop\Laura\services\local-api\.venv'; uv run --no-sync pytest tests/test_discovery.py -q; echo EXIT=$?
```

Ausgabe:

```text
...s                                                                     [100%]
3 passed, 1 skipped
EXIT=True
```

Die eine übersprungene Prüfung ist der optionale echte Semantic-Pfad. `fastembed` und
`qdrant_client` sind lokal importierbar, aber der lokale FastEmbed-Cache enthält die benötigte
`model_optimized.onnx` nicht. Der Test verwendet dieselben `importorskip`-Gates wie die
bestehende Semantic-Suite und überspringt zusätzlich nur den nicht initialisierbaren optionalen
Modell-Setup, statt einen Download anzustoßen oder auf ihn zu warten. In einer Umgebung mit
verfügbarem Modell indexiert er zwei Segmente in einem echten In-Memory-`SemanticIndex` und
prüft `source == "semantic"` sowie ein nichtleeres Ranking.

## Weitere Verifikation

```text
uv run --no-sync ruff check src/laura/short_creator/discovery.py src/laura/db/repos.py tests/test_discovery.py
All checks passed!

uv run --no-sync ruff format --check src/laura/short_creator/discovery.py tests/test_discovery.py
2 files already formatted

uv run --no-sync pytest tests/test_short_creator_context.py tests/test_shorts_repos.py -q; echo EXIT=$?
34 passed
EXIT=True

uv run --no-sync pytest tests/test_production_liveness.py tests/test_short_creator_providers.py -q; echo EXIT=$?
53 passed
EXIT=True
```

Alle Pytest-Läufe zeigten nur die bereits bestehende Starlette-TestClient-Deprecation und eine
OpenTelemetry-Import-Metadaten-Deprecation.

Abschließende Verifikation ohne den potenziell langsamen `uv`-Wrapper:

```powershell
& 'C:\Users\User\Desktop\Laura\services\local-api\.venv\Scripts\python.exe' -m pytest tests/test_discovery.py -q; echo EXIT=$?
& 'C:\Users\User\Desktop\Laura\services\local-api\.venv\Scripts\python.exe' -m ruff check src/laura/short_creator/discovery.py src/laura/db/repos.py tests/test_discovery.py; echo RUFF_EXIT=$?
```

```text
3 passed, 1 skipped
EXIT=True
All checks passed!
RUFF_EXIT=True
```

Der gezielte Mypy-Lauf für die drei Dateien wurde nach 64 Sekunden ohne Ergebnis durch das
60-Sekunden-Command-Limit beendet; wegen der festgehaltenen Test-, Ruff- und Diff-Prüfungen
wurde er nicht erneut unbeschränkt gestartet.

## Selbstreview

- `get_asset_rough_cut` übernimmt exakt die bisherige Auswahl des neuesten Rough-Cuts und führt
  ausschließlich einen SELECT aus.
- `_scene_ranges` spiegelt die bestehende `_resolve_scene`-Komposition: Szenenreihenfolge über
  `order_index + 1`, Clips von `list_timeline_clips` und Source-Ranges aus
  `context._scene_src_ranges`; alle Ranges bleiben end-exclusive.
- Ein Asset ohne Rough-Cut wird einmalig mit `"no rough cut"` übersprungen und erzeugt keinen
  Timeline-Eintrag.
- Lexical-Treffer erhalten wie spezifiziert `1.0` je Treffer; Semantic-Scores werden als Float
  aggregiert. Das Ranking ist absteigend nach Asset-Score, Scene-Snippets sind pro Asset auf drei
  begrenzt und schließlich nach Scene-Nummer geordnet.
- `git diff --cached --check` war vor dem Commit sauber. Ausschließlich die drei vorgesehenen
  Produktions-/Testdateien wurden gestaged und committed.

## Verbleibende Hinweise

- Der lokale optionale Semantic-Modellcache ist unvollständig; deshalb konnte der echte
  Semantic-Test hier nicht ausgeführt werden, ohne einen externen Modelldownload zu erzwingen.
- Mypy konnte innerhalb des erlaubten 60-Sekunden-Limits nicht abschließen.

## Fix Review

Die Reviewbefunde wurden mit einem zweiten TDD-Zyklus behoben:

- Index-Akquisition und Query degradieren bei jeder Ausnahme zur lexikalischen Suche.
- Ein Semantic-Score `0.0` bleibt erhalten; nur `None` oder ein fehlender Score wird zu `1.0`.
- Discovery unterscheidet `"no rough cut"` von `"no scenes"`; ein direkter Test pinnt den
  zweiten Grund für einen vorhandenen Rough Cut ohne Scene-Marker.
- Der echte optionale Semantic-Test überspringt nur die lokal belegte
  `onnxruntime.capi.onnxruntime_pybind11_state.NoSuchFile`; beliebige Indexierungsfehler
  werden nicht länger übersprungen.

### RED

```powershell
& 'C:\Users\User\Desktop\Laura\services\local-api\.venv\Scripts\python.exe' -m pytest tests/test_discovery.py -q; echo EXIT=$?
```

```text
...FFFs
FAILED test_index_acquisition_failure_falls_back_to_lexical
FAILED test_semantic_zero_score_is_not_replaced_with_lexical_default
FAILED test_rough_cut_without_scenes_is_skipped_with_its_own_reason
EXIT=False
```

### GREEN

```powershell
& 'C:\Users\User\Desktop\Laura\services\local-api\.venv\Scripts\python.exe' -m pytest tests/test_discovery.py -q; echo EXIT=$?
& 'C:\Users\User\Desktop\Laura\services\local-api\.venv\Scripts\python.exe' -m ruff check src/laura/short_creator/discovery.py tests/test_discovery.py; echo RUFF_EXIT=$?
& 'C:\Users\User\Desktop\Laura\services\local-api\.venv\Scripts\python.exe' -m ruff format --check src/laura/short_creator/discovery.py tests/test_discovery.py; echo FORMAT_EXIT=$?
```

```text
6 passed, 1 skipped
EXIT=True
All checks passed!
RUFF_EXIT=True
2 files already formatted
FORMAT_EXIT=True
```

Der Skip bleibt ausschließlich der bekannte optionale FastEmbed/ONNX-Modellcache-Fall; die
beiden bestehenden Deprecation-Warnungen aus Starlette-TestClient und OpenTelemetry blieben
unverändert.

## Targeted mypy import suppression

Der optionale ONNX-Runtime-Import im echten Semantic-Test ist jetzt mit der engsten möglichen
Suppressionsregel versehen: `# type: ignore[import-untyped]` direkt auf der betroffenen
Import-Anweisung. Es gab keine Konfigurations- oder Verhaltensänderung.

```powershell
& 'C:\Users\User\Desktop\Laura\services\local-api\.venv\Scripts\python.exe' -m pytest tests/test_discovery.py -q
& 'C:\Users\User\Desktop\Laura\services\local-api\.venv\Scripts\python.exe' -m ruff check tests/test_discovery.py
$env:MYPYPATH = 'src'; & 'C:\Users\User\Desktop\Laura\services\local-api\.venv\Scripts\python.exe' -m mypy tests/test_discovery.py
```

```text
pytest: 6 passed, 1 skipped
ruff: All checks passed!
mypy: Success: no issues found in 1 source file
```

`MYPYPATH=src` stellt für den isolierten Worktree sicher, dass Mypy das lokale Source-Paket
statt des im vorgegebenen Interpreter editierbar installierten Nachbar-Checkouts analysiert.
Ohne diese Pfadauflösung entstehen sechs nicht zur Änderung gehörende `laura.*`
`import-untyped`-Meldungen; die ONNX-Meldung ist durch die gezielte Regel behoben.

## Fix Review

Die Reviewbefunde wurden mit einem zweiten TDD-Zyklus behoben:

- Die gesamte Semantic-Index-Akquisition und die Query liegen jetzt im best-effort-Fallback.
  Jede Ausnahme bei `get_index()` oder `query()` führt zur lexikalischen Suche.
- Ein vorhandener Semantic-Score `0.0` wird unverändert aggregiert; nur ein fehlender oder
  `None`-Score erhält den lexikalischen Default `1.0`.
- Read-only Discovery unterscheidet nun `"no rough cut"` von `"no scenes"`; der zweite Grund
  ist durch einen direkten Test für einen vorhandenen Rough Cut ohne Scene-Marker festgelegt.
- Der echte optionale Semantic-Test überspringt nur noch
  `onnxruntime.capi.onnxruntime_pybind11_state.NoSuchFile`, die lokal belegte fehlende
  FastEmbed-Modelldatei; beliebige Indexierungsfehler werden nicht länger verschluckt.

### RED

```powershell
& 'C:\Users\User\Desktop\Laura\services\local-api\.venv\Scripts\python.exe' -m pytest tests/test_discovery.py -q; echo EXIT=$?
```

```text
...FFFs
FAILED test_index_acquisition_failure_falls_back_to_lexical
FAILED test_semantic_zero_score_is_not_replaced_with_lexical_default
FAILED test_rough_cut_without_scenes_is_skipped_with_its_own_reason
EXIT=False
```

Die Fehler waren erwartet: Index-Akquisition wurde propagiert, `0.0` in `1.0` umgewandelt und
ein Rough Cut ohne Scenes als `"no rough cut"` bezeichnet.

### GREEN

```powershell
& 'C:\Users\User\Desktop\Laura\services\local-api\.venv\Scripts\python.exe' -m pytest tests/test_discovery.py -q; echo EXIT=$?
& 'C:\Users\User\Desktop\Laura\services\local-api\.venv\Scripts\python.exe' -m ruff check src/laura/short_creator/discovery.py tests/test_discovery.py; echo RUFF_EXIT=$?
& 'C:\Users\User\Desktop\Laura\services\local-api\.venv\Scripts\python.exe' -m ruff format --check src/laura/short_creator/discovery.py tests/test_discovery.py; echo FORMAT_EXIT=$?
```

```text
6 passed, 1 skipped
EXIT=True
All checks passed!
RUFF_EXIT=True
2 files already formatted
FORMAT_EXIT=True
```

Der Skip bleibt ausschließlich der bekannte optionale FastEmbed/ONNX-Modellcache-Fall; die
beiden bestehenden Deprecation-Warnungen aus Starlette-TestClient und OpenTelemetry blieben
unverändert.
