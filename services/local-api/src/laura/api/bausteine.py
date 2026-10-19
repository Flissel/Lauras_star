"""Reel-Bausteine (Video-Claw, 17.10.2026).

``POST /projects/{pid}/stills``      Standbild-Clip aus einem Bild-Asset (mit Bewegung).
``POST /projects/{pid}/titelkarte``  Titelkarte (Farbflaeche, Text, optional Logo).

Beide stellen einen Job ein (``{"job_id"}``); ``GET /jobs/{id}`` liefert im Ergebnis
``asset_id`` des neuen Video-Assets. Geprueft wird hier, was ohne ffmpeg geht: Projekt,
Bild-Asset, Farben, Text passt auf die Karte.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from ..auth import Principal, require_permission
from ..bausteine.filter import umbrechen
from ..db import repos
from ..db.database import Database
from ..jobs.queues import queue_for
from ..jobs.runner import enqueue

router = APIRouter(tags=["bausteine"])
_FARBE = r"^#[0-9a-fA-F]{6}$"


class StandbildRequest(BaseModel):
    asset_id: str
    dauer_s: float = Field(default=3.0, ge=1.0, le=10.0)
    bewegung: Literal["ruhig", "zoom_rein", "zoom_raus", "schwenk"] = "zoom_rein"


class TitelkarteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=90)
    unterzeile: str | None = Field(default=None, max_length=60)
    hintergrund: str = Field(pattern=_FARBE)
    textfarbe: str = Field(default="#ffffff", pattern=_FARBE)
    logo_asset_id: str | None = None
    dauer_s: float = Field(default=2.5, ge=1.0, le=8.0)
    # Marke in Reels (C1): Anzeigeschrift der Marke (ID aus laura/assets/schriften); ohne = bisherige Schrift
    schrift: str | None = None

    @field_validator("schrift")
    @classmethod
    def _markenschrift(cls, v: str | None) -> str | None:
        from ..render import schriften
        if v is not None and v not in schriften.REGISTER:
            raise ValueError(f"unbekannte Schrift: {v}")
        return v


def _db(request: Request) -> Database:
    db: Database = request.app.state.db
    return db


def _projekt(db: Database, project_id: str) -> None:
    if repos.get_project(db, project_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")


def _bild(db: Database, project_id: str, asset_id: str) -> None:
    asset = repos.get_asset(db, asset_id)
    if asset is None or asset["project_id"] != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "asset not found")
    if asset["type"] != "image":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "asset ist kein Bild (Import fertig? Typ muss 'image' sein)")


def _einstellen(db: Database, kind: str, payload: dict[str, Any]) -> dict[str, str]:
    return {"job_id": enqueue(db, queue=queue_for(kind), kind=kind, payload=payload,
                              max_attempts=2)}


@router.post("/projects/{project_id}/stills", status_code=status.HTTP_202_ACCEPTED)
def standbild(
    project_id: str, body: StandbildRequest, request: Request,
    principal: Annotated[Principal, Depends(require_permission("timeline:edit"))],
) -> dict[str, str]:
    db = _db(request)
    _projekt(db, project_id)
    _bild(db, project_id, body.asset_id)
    return _einstellen(db, "baustein.standbild", {"project_id": project_id, **body.model_dump()})


@router.post("/projects/{project_id}/titelkarte", status_code=status.HTTP_202_ACCEPTED)
def titelkarte(
    project_id: str, body: TitelkarteRequest, request: Request,
    principal: Annotated[Principal, Depends(require_permission("timeline:edit"))],
) -> dict[str, str]:
    db = _db(request)
    _projekt(db, project_id)
    if body.logo_asset_id:
        _bild(db, project_id, body.logo_asset_id)
    try:
        umbrechen(body.text)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    return _einstellen(db, "baustein.titelkarte", {"project_id": project_id, **body.model_dump()})
