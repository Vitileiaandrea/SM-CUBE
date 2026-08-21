"""Scontornatura fette da immagine: sfondo blu, esclusione verde, grasso HSV.

La fetta va appoggiata su un piano blu di dimensione nota: la larghezza del
piano da' la scala mm/pixel, tutto cio' che sta fuori dal blu viene ignorato.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# sfondo blu (piano di appoggio)
BLUE_LOW = np.array([85, 30, 30])
BLUE_HIGH = np.array([135, 255, 255])

# verde da escludere: pinze e utensili verdi dentro il piano
GREEN_LOW = np.array([35, 40, 40])
GREEN_HIGH = np.array([85, 255, 255])

# grasso: chiaro e poco saturo
FAT_LOW = np.array([0, 0, 160])
FAT_HIGH = np.array([40, 80, 255])

MIN_AREA_PX = 2000
MIN_CONVEXITY = 0.35
MAX_ASPECT_RATIO = 7.0
BOARD_WIDTH_MM = 420.0
# oltre questa misura il contorno non e' una fetta sola: sono fette a contatto
MAX_SLICE_MM = 205.0


@dataclass
class SliceDetection2D:
    """Fetta misurata sull'immagine, in mm rispetto al centro del piano."""

    index: int
    contour_px: np.ndarray
    contour_mm: list[list[float]]
    center_px: tuple[float, float]
    center_mm: tuple[float, float]
    width_mm: float
    length_mm: float
    area_mm2: float
    fat_percentage: float
    shape_mask: np.ndarray
    fat_map: np.ndarray
    resolution_mm: float
    mm_per_px: float


def _board_mask(
    hsv: np.ndarray, board_width_mm: float
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Maschera del piano blu, maschera blu grezza e scala mm/pixel."""
    h, w = hsv.shape[:2]
    blue = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel, iterations=4)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # le fette possono spezzare il blu in piu' regioni: il piano e' l'inviluppo
    big = [c for c in contours if cv2.contourArea(c) > h * w * 0.01]
    if not big:
        return None
    board = cv2.convexHull(np.vstack(big))
    if cv2.contourArea(board) < h * w * 0.05:
        return None

    rect = cv2.minAreaRect(board)
    board_px = max(rect[1][0], rect[1][1])
    if board_px < 100:
        return None

    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled, [board], -1, 255, -1)
    return filled, blue, board_width_mm / float(board_px)


def _meat_mask(hsv: np.ndarray, board: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Dentro il piano: tutto cio' che non e' blu ne verde e' carne."""
    mask = cv2.bitwise_and(cv2.bitwise_not(blue), board)

    green = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    green = cv2.dilate(
        green, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2
    )
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(green))

    mask = cv2.erode(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def _rasterize(
    contour_px: np.ndarray,
    fat_px: np.ndarray,
    bbox: tuple[int, int, int, int],
    mm_per_px: float,
    resolution_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Contorno reale -> griglia 5 mm: forma e frazione di grasso per cella."""
    x, y, w, h = bbox
    cells_x = max(1, round(w * mm_per_px / resolution_mm))
    cells_y = max(1, round(h * mm_per_px / resolution_mm))

    filled = np.zeros(fat_px.shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour_px], -1, 255, -1)
    crop = filled[y:y + h, x:x + w].astype(np.float32) / 255.0
    fat_crop = (fat_px[y:y + h, x:x + w] > 0).astype(np.float32)

    # media d'area: ogni cella riceve la frazione di pixel carne / grasso
    shape = cv2.resize(crop, (cells_x, cells_y), interpolation=cv2.INTER_AREA)
    fat = cv2.resize(fat_crop, (cells_x, cells_y), interpolation=cv2.INTER_AREA)

    # assi immagine (righe = Y) -> assi cubo (indice 0 = X)
    shape_mask = (shape.T >= 0.5).astype(np.float32)
    fat_map = np.clip(fat.T, 0.0, 1.0) * shape_mask
    if not np.any(shape_mask > 0):
        shape_mask[:] = 1.0
    return shape_mask, fat_map


def segment_calibrated(
    image: np.ndarray,
    resolution_mm: float = 5.0,
    board_width_mm: float = BOARD_WIDTH_MM,
    slice_long_side_mm: float | None = None,
) -> tuple[list["SliceDetection2D"], np.ndarray, str | None]:
    """Come `segment_frame`, ma la scala puo' venire dalla fetta invece che dal piano.

    Utile sulle foto senza riferimento nel campo: si dichiara quanto misura il
    lato lungo della fetta piu' grande e la scala si ricava da lì.
    """
    detections, vis, error = segment_frame(image, resolution_mm, board_width_mm)
    if error or slice_long_side_mm is None or not detections:
        return detections, vis, error

    measured = max(max(d.width_mm, d.length_mm) for d in detections)
    if measured <= 0:
        return detections, vis, error
    corrected = board_width_mm * float(slice_long_side_mm) / measured
    return segment_frame(image, resolution_mm, corrected)


def _split_touching(contour: np.ndarray, mm_per_px: float) -> list[np.ndarray]:
    """Separa fette a contatto con watershed sulla mappa delle distanze."""
    x, y, w, h = cv2.boundingRect(contour)
    if max(w, h) * mm_per_px <= MAX_SLICE_MM:
        return [contour]

    blob = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(blob, [contour], -1, 255, -1, offset=(-x, -y))
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    _, cores = cv2.threshold(dist, 0.45 * dist.max(), 255, cv2.THRESH_BINARY)
    n_cores, markers = cv2.connectedComponents(cores.astype(np.uint8))
    if n_cores <= 2:
        return [contour]

    markers = markers + 1
    markers[blob == 0] = 0
    color = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)

    parts: list[np.ndarray] = []
    for label in range(2, n_cores + 1):
        piece = np.where(markers == label, 255, 0).astype(np.uint8)
        found, _ = cv2.findContours(
            piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in found:
            if cv2.contourArea(c) >= MIN_AREA_PX:
                parts.append(c + np.array([[[x, y]]], dtype=c.dtype))
    return parts or [contour]


def segment_frame(
    image: np.ndarray,
    resolution_mm: float = 5.0,
    board_width_mm: float = BOARD_WIDTH_MM,
) -> tuple[list[SliceDetection2D], np.ndarray, str | None]:
    """Trova le fette sul piano blu. Ritorna (fette, immagine annotata, errore).

    `board_width_mm` e' il lato lungo noto del piano blu: da' la scala mm/pixel.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    board_data = _board_mask(hsv, board_width_mm)
    if board_data is None:
        return [], image, "sfondo blu non rilevato"

    board, blue, mm_per_px = board_data
    meat = _meat_mask(hsv, board, blue)
    contours, _ = cv2.findContours(meat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fat_px = cv2.bitwise_and(cv2.inRange(hsv, FAT_LOW, FAT_HIGH), meat)
    board_moments = cv2.moments(board)
    board_cx = board_moments["m10"] / max(board_moments["m00"], 1)
    board_cy = board_moments["m01"] / max(board_moments["m00"], 1)

    pieces: list[np.ndarray] = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_AREA_PX:
            pieces.extend(_split_touching(cnt, mm_per_px))

    detections: list[SliceDetection2D] = []
    for cnt in pieces:
        area_px = cv2.contourArea(cnt)
        if area_px < MIN_AREA_PX:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0 or area_px / hull_area < MIN_CONVEXITY:
            continue
        rect_w, rect_h = cv2.minAreaRect(cnt)[1]
        if min(rect_w, rect_h) <= 0:
            continue
        if max(rect_w, rect_h) / min(rect_w, rect_h) > MAX_ASPECT_RATIO:
            continue

        # contorno reale, non l'inviluppo: la forma concava va conservata
        outline = cv2.approxPolyDP(cnt, 1.5, True)
        x, y, w, h = cv2.boundingRect(outline)
        shape_mask, fat_map = _rasterize(
            outline, fat_px, (x, y, w, h), mm_per_px, resolution_mm
        )
        cx, cy = x + w / 2.0, y + h / 2.0
        active = shape_mask > 0
        fat_pct = float(np.mean(fat_map[active])) if np.any(active) else 0.0

        pts = outline.reshape(-1, 2)
        if len(pts) > 60:
            pts = pts[np.linspace(0, len(pts) - 1, 60, dtype=int)]
        contour_mm = [
            [
                round(float((p[0] - cx) * mm_per_px), 1),
                round(float((p[1] - cy) * mm_per_px), 1),
            ]
            for p in pts
        ]

        detections.append(
            SliceDetection2D(
                index=len(detections),
                contour_px=outline,
                contour_mm=contour_mm,
                center_px=(round(cx, 1), round(cy, 1)),
                center_mm=(
                    round(float((cx - board_cx) * mm_per_px), 1),
                    round(float((cy - board_cy) * mm_per_px), 1),
                ),
                width_mm=round(shape_mask.shape[0] * resolution_mm, 1),
                length_mm=round(shape_mask.shape[1] * resolution_mm, 1),
                area_mm2=round(area_px * mm_per_px ** 2, 1),
                fat_percentage=round(fat_pct, 3),
                shape_mask=shape_mask,
                fat_map=fat_map,
                resolution_mm=resolution_mm,
                mm_per_px=mm_per_px,
            )
        )

    if not detections:
        return [], image, "nessuna fetta sul piano blu"
    return detections, image, None
