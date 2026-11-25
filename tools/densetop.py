"""densetop.py <poses_final> <out.png> [zlo zhi]: 바닥 기준 높이 띠(기본 0.6–1.3 m: 책상 · 모니터) 점 밀도를 위에서(2 cm 칸, 로그 밝기)"""
import sys, os, json, numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
P = np.loadtxt(sys.argv[1]); C = os.environ["CAM"]
zlo, zhi = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.6, 1.3)
ci = json.load(open(C + "/calib.json"))["color_intrinsics"]; fx, fy, cx, cy = ci["fx"], ci["fy"], ci["ppx"], ci["ppy"]
dt = {round(float(l.split()[0]), 4): l.split()[1] for l in open(C + "/depth.txt") if l[0] != "#"}
vv, uu = np.mgrid[4:480:8, 4:640:8]; pts = []
for i in range(0, len(P), 3):
    p = dt.get(round(P[i, 0], 4))
    if not p: continue
    d = np.asarray(Image.open(C + "/" + p))[vv, uu].astype(float) / 1000.0; m = (d > 0.3) & (d < float(os.environ.get("ZMAX", 4.0)))
    x = np.stack([(uu[m] - cx) * d[m] / fx, (vv[m] - cy) * d[m] / fy, d[m]], 1)
    pts.append(R.from_quat(P[i, 4:8]).apply(x) + P[i, 1:4])
a = np.concatenate(pts); fl = np.percentile(a[:, 2], 3); a = a[(a[:, 2] > fl + zlo) & (a[:, 2] < fl + zhi)]
lo = np.percentile(a[:, :2], 0.5, 0) - 0.3; hi = np.percentile(a[:, :2], 99.5, 0) + 0.3; c = 0.02
h, _, _ = np.histogram2d(a[:, 0], a[:, 1], bins=[int((hi[0] - lo[0]) / c), int((hi[1] - lo[1]) / c)], range=[[lo[0], hi[0]], [lo[1], hi[1]]])
img = (255 - 255 * np.log1p(h) / np.log1p(h.max())).astype(np.uint8).T[::-1]
Image.fromarray(img).save(sys.argv[2]); print(img.shape, "floor", round(fl, 2))
