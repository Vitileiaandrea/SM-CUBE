"""Web app di prova FM 7000: camera Creality o immagini, logica robot vera.

Lancio:  python -m fm7000.live.server
Browser: http://localhost:8080
"""

import base64
import os
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from fm7000.live.planner import LivePlan, LivePlanner
from fm7000.live.segmentation import segment_calibrated

app = FastAPI(title="FM 7000 - prova dal vivo")
planner = LivePlanner()
_plans: list[LivePlan] = []

# cartella di immagini usata come nastro simulato
IMAGE_DIR = Path(os.environ.get("FM7000_IMMAGINI", Path.home() / "fette"))
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

CUP_COLOR = (0, 220, 255)
OFF_CUP_COLOR = (150, 150, 150)
SLICE_COLORS = [(0, 255, 0), (0, 165, 255), (255, 200, 0), (255, 0, 200)]


class Frame(BaseModel):
    immagine: str | None = None
    file: str | None = None
    spessore_mm: float = 25.0
    lato_piano_mm: float = 420.0
    fetta_mm: float | None = None


class Conferma(BaseModel):
    indice: int


def _annotate(image: np.ndarray, plans: list[LivePlan]) -> str:
    vis = image.copy()
    for order, plan in enumerate(plans):
        det = plan.detection
        color = SLICE_COLORS[order % len(SLICE_COLORS)]
        thickness = 4 if order == 0 else 2
        cv2.drawContours(vis, [det.contour_px], -1, color, thickness)
        cx, cy = int(det.center_px[0]), int(det.center_px[1])
        cv2.putText(
            vis,
            f"{order + 1}",
            (cx - 12, cy + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            color,
            4,
        )
        label = f"{plan.meat_slice.meat_type.name} {det.fat_percentage * 100:.0f}% gr"
        cv2.putText(
            vis, label, (cx - 70, cy + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

        if order != 0:
            continue
        # ventose disegnate dove cadono davvero: offset presa + rotazione fetta
        radius_px = max(3, int(15.0 / det.mm_per_px))
        for r, c, dx_mm, dy_mm in planner.gripper_selector.cup_layout_on_slice_mm(
            plan.gripper.pick_offset_x_mm,
            plan.gripper.pick_offset_y_mm,
            plan.candidate.rotation_deg,
            plan.meat_slice,
        ):
            px = int(cx + dx_mm / det.mm_per_px)
            py = int(cy + dy_mm / det.mm_per_px)
            if plan.gripper.cup_pattern[r, c] > 0:
                cv2.circle(vis, (px, py), radius_px, CUP_COLOR, 3)
                cv2.circle(vis, (px, py), 3, CUP_COLOR, -1)
            else:
                cv2.circle(vis, (px, py), radius_px, OFF_CUP_COLOR, 1)

    ok, buffer = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return base64.b64encode(buffer).decode("ascii")


def _load_image(frame: Frame) -> np.ndarray | None:
    if frame.file:
        path = IMAGE_DIR / Path(frame.file).name
        return cv2.imread(str(path)) if path.exists() else None
    if not frame.immagine:
        return None
    try:
        data = base64.b64decode(frame.immagine.split(",")[-1])
    except (ValueError, TypeError):
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


@app.get("/immagini")
def immagini() -> dict:
    if not IMAGE_DIR.is_dir():
        return {"cartella": str(IMAGE_DIR), "file": []}
    names = sorted(
        p.name for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXT
    )
    return {"cartella": str(IMAGE_DIR), "file": names}


@app.post("/analizza")
def analizza(frame: Frame) -> dict:
    global _plans
    image = _load_image(frame)
    if image is None:
        return {"ok": False, "errore": "immagine non leggibile"}

    detections, _, error = segment_calibrated(
        image,
        board_width_mm=frame.lato_piano_mm,
        slice_long_side_mm=frame.fetta_mm,
    )
    if error:
        return {"ok": False, "errore": error, "stato": planner.state()}

    _plans = planner.plan_all(detections, frame.spessore_mm)
    if not _plans:
        return {
            "ok": False,
            "errore": "nessun deposito eseguibile: chiudere lo strato",
            "stato": planner.state(),
        }

    return {
        "ok": True,
        "fette": [plan.to_dict(planner) for plan in _plans],
        "immagine": _annotate(image, _plans),
        "stato": planner.state(),
    }


@app.post("/conferma")
def conferma(body: Conferma) -> dict:
    for plan in _plans:
        if plan.detection.index == body.indice:
            return planner.confirm(plan)
    return {"ok": False, "errore": "fetta non piu in elenco: rianalizza"}


@app.post("/chiudi-strato")
def chiudi_strato() -> dict:
    return {"ok": True, "stato": planner.close_layer()}


@app.post("/reset")
def reset() -> dict:
    _plans.clear()
    return {"ok": True, "stato": planner.reset()}


@app.get("/stato")
def stato() -> dict:
    return planner.state()


@app.get("/")
def home() -> HTMLResponse:
    page = Path(__file__).with_name("index.html").read_text(encoding="utf-8")
    return HTMLResponse(page)


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
