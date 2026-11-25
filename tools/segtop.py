"""segtop.py <poses_final> <out_prefix> [seg=300]: 끝 자세를 장면 seg 장씩 구간으로 나눠
① 구간마다 단독 위에서 본 점(높이 0.3–1.6 m 위 바닥 기준, 벽 · 책상) 그림 ② 연속 구간 겹쳐 그림(앞 빨강 · 뒤 초록)
③ 연속 구간 겹친 곳 2D ICP(방위 + 이동)로 뒤 구간을 앞 구간에 맞추는 데 필요한 어긋남을 출력"""
import sys, os, json, numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
P = np.loadtxt(sys.argv[1]); OUT = sys.argv[2]; SEG = int(sys.argv[3]) if len(sys.argv) > 3 else 300
C = os.environ.get("CAM", os.path.expanduser("~/rec_dl/ds_1118_121652/cam"))
ci = json.load(open(C + "/calib.json"))["color_intrinsics"]; fx, fy, cx, cy = ci["fx"], ci["fy"], ci["ppx"], ci["ppy"]
dt = {round(float(l.split()[0]), 4): l.split()[1] for l in open(C + "/depth.txt") if l[0] != "#"}
allt = sorted(float(l.split()[0]) for l in open(C + "/rgb.txt") if l[0] != "#")
fidx = np.searchsorted(allt, P[:, 0] - 1e-4)
S = 8; vv, uu = np.mgrid[S // 2:480:S, S // 2:640:S]
def cloud(i):
    p = dt.get(round(P[i, 0], 4))
    if not p: return None
    d = np.asarray(Image.open(C + "/" + p))[vv, uu].astype(float) / 1000.0
    m = (d > 0.3) & (d < 3.0)
    x = np.stack([(uu[m] - cx) * d[m] / fx, (vv[m] - cy) * d[m] / fy, d[m]], 1)
    return R.from_quat(P[i, 4:8]).apply(x) + P[i, 1:4]
segs = {}
for i in range(0, len(P), 3):
    c = cloud(i)
    if c is not None: segs.setdefault(fidx[i] // SEG, []).append(c)
segs = {k: np.concatenate(v) for k, v in segs.items()}
allp = np.concatenate(list(segs.values())); floor = np.percentile(allp[:, 2], 3)
segs = {k: v[(v[:, 2] > floor + 0.3) & (v[:, 2] < floor + 1.6)] for k, v in segs.items()}
lo = np.percentile(allp[:, :2], 1, 0) - 0.5; hi = np.percentile(allp[:, :2], 99, 0) + 0.5; sc = 60
W, H = (hi - lo) * sc
def draw(layers, path, txt):
    im = Image.new("RGB", (int(W), int(H) + 20), "white"); dr = ImageDraw.Draw(im)
    for pts, col in layers:
        q = ((pts[:, :2] - lo) * sc).astype(int)
        for x, y in q[::2]: im.putpixel((min(x, int(W) - 1), min(max(int(H) - 1 - y, 0), int(H) - 1)), col)
    dr.text((4, int(H) + 4), txt, fill="black"); im.save(path)
def icp2(a, b):  # b 를 a 에 맞춤(2D)
    ta = cKDTree(a); th = 0; tx = np.zeros(2); n = 0
    for th_ in (0.3, 0.3, 0.15, 0.15, 0.08, 0.08):
        c, s = np.cos(th), np.sin(th); bb = b @ np.array([[c, s], [-s, c]]) + tx
        d, j = ta.query(bb); m = d < th_; n = m.sum()
        if n < 200: return None
        p, q = bb[m], a[j[m]]; mp, mq = p.mean(0), q.mean(0)
        Hm = (p - mp).T @ (q - mq); dth = np.arctan2(Hm[0, 1] - Hm[1, 0], Hm[0, 0] + Hm[1, 1])
        c2, s2 = np.cos(dth), np.sin(dth); Rm = np.array([[c2, s2], [-s2, c2]])
        th += dth; tx = (tx - mp) @ Rm + mq if False else tx @ Rm + (mq - mp @ Rm)
    return np.degrees(th), np.linalg.norm(tx), n
ks = sorted(segs)
for k in ks:
    draw([(segs[k], (0, 0, 0))], f"{OUT}_seg{k}.png", f"frames {k*SEG}-{k*SEG+SEG-1}")
for a, b in zip(ks, ks[1:]):
    draw([(segs[a], (220, 0, 0)), (segs[b], (0, 160, 0))], f"{OUT}_pair{a}{b}.png", f"red {a*SEG}- green {b*SEG}-")
    r = icp2(segs[a][::4, :2], segs[b][::4, :2])
    print(f"구간 {a*SEG}-{a*SEG+SEG-1} ↔ {b*SEG}-{b*SEG+SEG-1}: " + ("겹침 부족" if r is None else f"뒤 구간 어긋남 방위 {r[0]:+.1f}° 이동 {r[1]:.2f} m (맞은 점 {r[2]})"))
draw([(segs[k], tuple(int(v) for v in np.array([[230,25,75],[60,180,75],[0,130,200],[245,130,48],[145,30,180],[70,240,240],[240,50,230],[128,128,0],[0,0,128],[170,110,40],[128,0,0],[0,128,128],[0,0,0]])[k % 13])) for k in ks], f"{OUT}_all.png", "all segments colored")
