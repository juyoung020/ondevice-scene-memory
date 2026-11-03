"""깊이 모델 비교: TUM RGB-D 의 Kinect 깊이를 정답으로 YOLO26s-depth vs Depth Anything V2 Metric-Indoor-Small.

  python tools/depth_compare.py data/tum/rgbd_dataset_freiburg3_long_office_household [--step 130]

프레임마다: 배율(예측/정답 중앙값), 절대 상대오차(AbsRel, 배율 그대로), 프레임 배율을 맞춘 뒤 AbsRel, 추론 시간.
그림: docs/results/depth_compare.png (RGB | Kinect | YOLO | DA2, 같은 색 범위 0–4 m)
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('seq')
ap.add_argument('--step', type=int, default=130)
ap.add_argument('--out', default='docs/results/depth_compare.png')
a = ap.parse_args()

from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

yolo = YOLO('models/yolo26s-depth.pt')
da_dir = 'models/depth-anything-v2-metric-indoor-small'
proc = AutoImageProcessor.from_pretrained(da_dir)
da = AutoModelForDepthEstimation.from_pretrained(da_dir).cuda().eval()
print('params  yolo', sum(p.numel() for p in yolo.model.parameters()) / 1e6, 'M   da2', sum(p.numel() for p in da.parameters()) / 1e6, 'M')


def run_yolo(img):
    return yolo(img, imgsz=768, verbose=False)[0].depth.data.cpu().numpy()


@torch.no_grad()
def run_da(img):
    x = proc(images=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), return_tensors='pt').to('cuda')
    return da(**x).predicted_depth[0].float().cpu().numpy()


rgb = [l.split() for l in open(os.path.join(a.seq, 'rgb.txt')) if l[0] != '#']
dpt = [l.split() for l in open(os.path.join(a.seq, 'depth.txt')) if l[0] != '#']
td = np.array([float(x[0]) for x in dpt])
stats = {'yolo': [], 'da2': []}
rows = []
for i in range(0, len(rgb), a.step):
    img = cv2.imread(os.path.join(a.seq, rgb[i][1]))
    j = int(np.argmin(abs(td - float(rgb[i][0]))))
    g = cv2.imread(os.path.join(a.seq, dpt[j][1]), -1).astype(np.float32) / 5000.0
    m = (g > 0.3) & (g < 5)
    preds = {}
    for name, fn in (('yolo', run_yolo), ('da2', run_da)):
        fn(img)  # warm
        torch.cuda.synchronize(); t0 = time.time()
        p = fn(img)
        torch.cuda.synchronize(); ms = (time.time() - t0) * 1000
        p = cv2.resize(p, (g.shape[1], g.shape[0]))
        s = float(np.median(p[m] / g[m]))
        stats[name].append((s, float(np.median(np.abs(p[m] - g[m]) / g[m])), float(np.median(np.abs(p[m] / s - g[m]) / g[m])), ms))
        preds[name] = p
    print(i, '  '.join(f'{k}: scale {v[-1][0]:.2f} absrel {v[-1][1]:.2f} fit {v[-1][2]:.2f}' for k, v in stats.items()))
    if len(rows) < 4 and i % (a.step * 5) == 0:
        cm = lambda d: cv2.applyColorMap(np.clip(d / 4.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        gg = cm(g); gg[g <= 0] = 0
        rows.append(np.hstack([img, gg, cm(preds['yolo']), cm(preds['da2'])]))

print()
for k, v in stats.items():
    v = np.array(v)
    print(f'{k:5s} scale median {np.median(v[:, 0]):.2f} range {v[:, 0].min():.2f}-{v[:, 0].max():.2f} (spread std {v[:, 0].std():.2f})  '
          f'absrel {v[:, 1].mean():.3f}  absrel after per-frame scale {v[:, 2].mean():.3f}  {np.median(v[:, 3]):.0f} ms (RTX)')
img = np.vstack(rows)
hdr = np.full((40, img.shape[1], 3), 255, np.uint8)
for x, t in zip(range(0, img.shape[1], 640), ['RGB', 'Kinect (truth)', 'YOLO26s-depth', 'Depth Anything V2 S']):
    cv2.putText(hdr, t, (x + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
cv2.imwrite(a.out, np.vstack([hdr, img]))
print('saved', a.out)
