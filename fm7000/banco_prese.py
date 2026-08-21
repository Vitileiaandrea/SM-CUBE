"""Banco da 1000 prese: caratterizza la presa su tutte le variabili fetta.

Le fette reali segmentate dalle foto vengono riproposte in ordine casuale e
con orientamenti diversi, cosi' il planner le incontra in stati di cubo
diversi: e' il modo per vedere dove la regola di presa tiene e dove cede.
"""

import glob
import random

import cv2
import numpy as np

from fm7000.live.planner import LivePlanner
from fm7000.live.segmentation import segment_calibrated

N = 1000
NEEDED = 10.0
CUP_R = 15.0
SPACING = 50.0


def sporgenza(plan):
    g = plan.gripper
    ms = plan.meat_slice.rotate(plan.candidate.rotation_deg)
    mask = ms.shape_mask > 0
    res = ms.resolution_mm
    cells = np.argwhere(mask)
    ci = cells[:, 0].mean() + g.pick_offset_x_mm / res
    cj = cells[:, 1].mean() + g.pick_offset_y_mm / res
    half = (1.5 * SPACING + CUP_R) / res
    return max(
        max(0.0, ci - half - cells[:, 0].min()),
        max(0.0, cells[:, 0].max() - (ci + half)),
        max(0.0, cj - half - cells[:, 1].min()),
        max(0.0, cells[:, 1].max() - (cj + half)),
    ) * res


def carica():
    dets = []
    for f in sorted(glob.glob("/home/ubuntu/fette/*")):
        img = cv2.imread(f)
        if img is None:
            continue
        d, _, _ = segment_calibrated(img, slice_long_side_mm=180.0)
        dets.extend(d)
    return dets


def main():
    dets = carica()
    rng = random.Random(7)
    print(f"fette reali disponibili: {len(dets)}")

    planner = LivePlanner()
    rows = []
    cubi = 0
    for k in range(N):
        if k and k % 50 == 0:
            print(f"  ... {k}/{N} prese")
        det = dets[rng.randrange(len(dets))]
        plan = planner.plan(det, rng.choice([20.0, 25.0, 30.0, 35.0, 40.0]))
        if plan is None:
            planner.close_layer()
            plan = planner.plan(det, 30.0)
        if plan is None:
            planner.reset()
            cubi += 1
            continue
        g = plan.gripper
        ms = plan.meat_slice
        cells = np.argwhere(ms.shape_mask > 0)
        taglia = max(
            cells[:, 0].max() - cells[:, 0].min(),
            cells[:, 1].max() - cells[:, 1].min(),
        ) * ms.resolution_mm
        rows.append(
            {
                "zona": plan.candidate.zone.value,
                "taglia": taglia,
                "area": float(cells.shape[0]) * ms.resolution_mm**2 / 100.0,
                "ventose": g.active_cups,
                "sporgenza": sporgenza(plan),
                "primaria": g.min_clearance_mm,
                "labbro": g.min_coverage,
                "rot": abs(plan.candidate.rotation_deg),
            }
        )
        planner.confirm(plan)
        if planner.cube_state.is_full:
            planner.reset()
            cubi += 1
    print(f"prese pianificate: {len(rows)} su {N}  (cubi chiusi: {cubi})")

    ventose = np.array([r["ventose"] for r in rows])
    sp = np.array([r["sporgenza"] for r in rows])
    pr = np.array([r["primaria"] for r in rows])
    print(
        f"ventose attive: media {ventose.mean():.2f} min {ventose.min()} "
        f"max {ventose.max()}  >=4: {(ventose >= 4).mean() * 100:.0f}%"
    )
    print(
        f"contenimento (<=10 mm): {(sp <= 10.5).mean() * 100:.0f}%  "
        f"media {sp.mean():.1f} mm  p95 {np.percentile(sp, 95):.1f} mm"
    )
    print(
        f"perimetro >=10 mm: {(pr >= 9.5).mean() * 100:.0f}%  "
        f"media {pr.mean():.1f} mm  peggiore {pr.min():.1f} mm"
    )
    lb = np.array([r["labbro"] for r in rows])
    print(
        f"labbro sulla carne (peggiore ventosa attiva): media {lb.mean() * 100:.0f}%"
        f"  minimo {lb.min() * 100:.0f}%  sotto meta': {(lb < 0.5).mean() * 100:.0f}%"
    )

    print("\nper taglia fetta:")
    bins = [(0, 100), (100, 130), (130, 160), (160, 250)]
    for lo, hi in bins:
        sel = [r for r in rows if lo <= r["taglia"] < hi]
        if not sel:
            continue
        v = np.array([r["ventose"] for r in sel])
        s = np.array([r["sporgenza"] for r in sel])
        p = np.array([r["primaria"] for r in sel])
        print(
            f"  {lo:3d}-{hi:3d} mm  n={len(sel):4d}  ventose {v.mean():.2f}  "
            f"sporgenza {s.mean():5.1f}  primaria ok {(p >= 9.5).mean() * 100:3.0f}%"
        )

    print("\nper zona:")
    for zona in sorted({r["zona"] for r in rows}):
        sel = [r for r in rows if r["zona"] == zona]
        v = np.array([r["ventose"] for r in sel])
        s = np.array([r["sporgenza"] for r in sel])
        p = np.array([r["primaria"] for r in sel])
        print(
            f"  {zona:20s} n={len(sel):4d}  ventose {v.mean():.2f}  "
            f"sporgenza {s.mean():5.1f}  primaria ok {(p >= 9.5).mean() * 100:3.0f}%"
        )

    print("\nper rotazione della presa:")
    for lo, hi in [(0, 30), (30, 60), (60, 90), (90, 181)]:
        sel = [r for r in rows if lo <= r["rot"] < hi]
        if not sel:
            continue
        v = np.array([r["ventose"] for r in sel])
        p = np.array([r["primaria"] for r in sel])
        print(
            f"  {lo:3d}-{hi:3d} deg  n={len(sel):4d}  ventose {v.mean():.2f}  "
            f"primaria ok {(p >= 9.5).mean() * 100:3.0f}%"
        )

    peggio = [r for r in rows if r["primaria"] < 9.5]
    if peggio:
        t = np.array([r["taglia"] for r in peggio])
        print(
            f"\ncasi con primaria sotto i 10 mm: {len(peggio)} "
            f"(taglia media {t.mean():.0f} mm, ventose "
            f"{np.mean([r['ventose'] for r in peggio]):.1f})"
        )


if __name__ == "__main__":
    main()
