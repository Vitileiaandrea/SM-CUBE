"""Geometria del contorno fetta: spigoli e lati dritti per la presa.

La presa non e' libera a caso: si scegle uno spigolo della fetta e il lato piu'
dritto che parte da quello spigolo, poi la fetta si gira perche' quel lato sia
parallelo al lato della griglia ventose. Cosi' la ventosa primaria va sullo
spigolo (10 mm dentro il bordo) e le altre del perimetro corrono lungo il lato
dritto, sempre a 10 mm dal bordo: sono loro che guidano la fetta a parete.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CornerFit:
    """Spigolo della fetta con il suo lato dritto."""

    # rotazione da applicare alla fetta perche' il lato dritto sia parallelo
    # all'asse della griglia (gradi, verso antiorario)
    align_deg: float
    # vertice dello spigolo in mm (riga, colonna) rispetto alla maschera
    corner_mm: tuple[float, float]
    # lunghezza del lato dritto in mm
    edge_len_mm: float
    # scarto medio del contorno dalla retta del lato, in mm (0 = perfetto)
    straightness_mm: float
    # angolo interno dello spigolo in gradi (90 = spigolo quadro)
    corner_deg: float
    score: float


def corner_fits(
    mask: np.ndarray, resolution_mm: float, max_fits: int = 6
) -> list[CornerFit]:
    """Spigoli utili della fetta, dal migliore al peggiore.

    Uno spigolo e' buono se e' vicino a 90 gradi e almeno uno dei due lati che
    ci arrivano e' lungo e dritto: e' quello che si appoggia allo spigolo del
    cubo facendo due pareti in un colpo.
    """
    poly = _polygon(mask, resolution_mm)
    if poly is None or len(poly) < 3:
        return []

    fits: list[CornerFit] = []
    n = len(poly)
    for k in range(n):
        prev_p, corner, next_p = poly[k - 1], poly[k], poly[(k + 1) % n]
        inner = _inner_angle(prev_p, corner, next_p)
        for other in (next_p, prev_p):
            vec = other - corner
            length = float(np.hypot(*vec))
            if length < 30.0:
                continue
            dev = _straightness(mask, resolution_mm, corner, other)
            # spigolo quadro e lato lungo e dritto: e' la presa che porta la
            # fetta a parete su tutta la linea di ventose
            score = (
                length
                - abs(inner - 90.0) * 1.5
                - dev * 6.0
            )
            fits.append(
                CornerFit(
                    align_deg=_align_deg(vec),
                    corner_mm=(float(corner[0]), float(corner[1])),
                    edge_len_mm=length,
                    straightness_mm=dev,
                    corner_deg=inner,
                    score=score,
                )
            )
    fits.sort(key=lambda f: f.score, reverse=True)
    return fits[:max_fits]


def _polygon(mask: np.ndarray, res: float) -> np.ndarray | None:
    """Contorno della fetta approssimato a poligono, in mm (riga, colonna)."""
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return None
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    # tolleranza pari a una cella e mezza: tiene i lati veri e butta il rumore
    approx = cv2.approxPolyDP(contour, 1.5, True)
    pts = approx.reshape(-1, 2)[:, ::-1].astype(float) * res
    return pts


def _inner_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    n1, n2 = np.hypot(*v1), np.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _align_deg(vec: np.ndarray) -> float:
    """Rotazione che porta il lato parallelo all'asse colonne della griglia."""
    ang = float(np.degrees(np.arctan2(vec[0], vec[1])))
    return round(-ang, 1)


def _straightness(
    mask: np.ndarray, res: float, start: np.ndarray, end: np.ndarray
) -> float:
    """Scarto medio del bordo dalla retta start-end, in mm."""
    vec = end - start
    length = float(np.hypot(*vec))
    if length < 1e-6:
        return 0.0
    unit = vec / length
    normal = np.array([-unit[1], unit[0]])
    steps = max(4, int(length / max(res, 1.0)))
    devs = []
    for t in np.linspace(0.0, 1.0, steps):
        point = start + vec * t
        devs.append(abs(_edge_offset(mask, res, point, normal)))
    return float(np.mean(devs))


def _edge_offset(
    mask: np.ndarray, res: float, point: np.ndarray, normal: np.ndarray
) -> float:
    """Distanza dal punto al bordo vero lungo la normale, in mm."""
    inside = _sample(mask, point, res)
    for sign in (1.0, -1.0):
        for step in range(12):
            probe = point + normal * sign * step * res
            if _sample(mask, probe, res) != inside:
                return step * res
    return 0.0


def _sample(mask: np.ndarray, point: np.ndarray, res: float) -> bool:
    i = round(point[0] / res)
    j = round(point[1] / res)
    if not (0 <= i < mask.shape[0] and 0 <= j < mask.shape[1]):
        return False
    return bool(mask[i, j] > 0)
