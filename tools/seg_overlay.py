"""분할 점검: TUM 프레임 몇 장에 분할 마스크를 덮어 그리고, ObjectSAM 이면 SigLIP 2 이름과 구조물 판정을 같이 적는다.

  python tools/seg_overlay.py data/tum/rgbd_dataset_freiburg3_long_office_household [--seg models/ObjectSAM-416.pt] [--conf 0.5]
"""
import argparse
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
import scene_graph as SG

ap = argparse.ArgumentParser()
ap.add_argument('seq')
ap.add_argument('--seg', default='models/ObjectSAM-416.pt')
ap.add_argument('--conf', type=float, default=0.5)
ap.add_argument('--frames', default='0,650,1300,1950')
ap.add_argument('--out', default='docs/results/seg_overlay.png')
a = ap.parse_args()
seg = YOLO(a.seg)
agn = len(seg.names) == 1
emb = None
if agn:
    emb = SG.Embedder()
    emb.set_labels(SG.OBJECT_NAMES, SG.STRUCTURAL_NAMES)
rows = [l.split() for l in open(os.path.join(a.seq, 'rgb.txt')) if l[0] != '#']
tiles = []
rng = np.random.default_rng(0)
for fi in map(int, a.frames.split(',')):
    img = cv2.imread(os.path.join(a.seq, rows[fi][1]))
    H, W = img.shape[:2]
    r = seg(img, imgsz=416 if agn else 640, conf=a.conf, retina_masks=True, verbose=False)[0]
    vis = img.copy()
    n = 0 if r.masks is None else len(r.boxes)
    masks, crops = [], []
    for k in range(n):
        m = r.masks.data[k].cpu().numpy() > 0.5
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        x1, y1, x2, y2 = r.boxes.xyxy[k].cpu().numpy().astype(int)
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(min(x2, W), x1 + 1), max(min(y2, H), y1 + 1)
        masks.append((m, (x1, y1, x2, y2)))
        crops.append((img[y1:y2, x1:x2], m[y1:y2, x1:x2]))
    labs = emb.labels(crops) if agn else [(r.names[int(r.boxes.cls[k])], float(r.boxes.conf[k]), False) for k in range(n)]
    for (m, (x1, y1, x2, y2)), (name, p, st) in zip(masks, labs):
        drop = st or m.mean() > 0.35
        col = (0, 0, 255) if drop else tuple(int(c) for c in rng.integers(60, 255, 3))
        vis[m] = (0.45 * vis[m] + 0.55 * np.array(col)).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, col, 2)
        cv2.putText(vis, f"{'X ' if drop else ''}{name} {p:.2f}", (x1 + 2, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 3)
        cv2.putText(vis, f"{'X ' if drop else ''}{name} {p:.2f}", (x1 + 2, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    cv2.putText(vis, f'frame {fi}: {n} masks', (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(vis, f'frame {fi}: {n} masks', (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    tiles.append(np.hstack([img, vis]))
cv2.imwrite(a.out, np.vstack(tiles))
print('saved', a.out)
