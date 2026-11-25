"""mancheck.py <poses_final> : 맨해튼 방위(refine::manhattan 과 같은 수평 법선 4 배 각)가 벽 없는 장면에도 걸리나.
장면 10 장마다 깊이(8 화소 격자)로 법선 → 세계 수평 법선(|n.z|<0.15) 수, 4 배 각 집중도, 전체 기준과 차.
'벽 없음' = 수평 법선 점이 모두 1.8 m 안(가까운 책상 · 모니터 · 의자만 보임). CAM 환경변수 = 녹화 cam 폴더"""
import sys, os, json, numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
P = np.loadtxt(sys.argv[1]); C = os.environ["CAM"]
ci = json.load(open(C + "/calib.json"))["color_intrinsics"]; fx, fy, cx, cy = ci["fx"], ci["fy"], ci["ppx"], ci["ppy"]
dt = {round(float(l.split()[0]), 4): l.split()[1] for l in open(C + "/depth.txt") if l[0] != "#"}
vv, uu = np.mgrid[4:480:8, 4:640:8]
rows = []
for i in range(0, len(P), 10):
    p = dt.get(round(P[i, 0], 4))
    if not p: continue
    d = np.asarray(Image.open(C + "/" + p))[vv, uu].astype(float) / 1000.0
    X = np.stack([(uu - cx) * d / fx, (vv - cy) * d / fy, d], -1)
    dx = X[1:-1, 2:] - X[1:-1, :-2]; dy = X[2:, 1:-1] - X[:-2, 1:-1]; z = d[1:-1, 1:-1]
    ok = (z > 0.3) & (z < 3.5) & (np.abs(dx[..., 2]) < 0.05 * z + 0.02) & (np.abs(dy[..., 2]) < 0.05 * z + 0.02)
    for a in (d[1:-1, 2:], d[1:-1, :-2], d[2:, 1:-1], d[:-2, 1:-1]): ok &= (a > 0.3) & (a < 3.5)
    n = np.cross(dx, dy)[ok]; n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    w = R.from_quat(P[i, 4:8]).apply(n); h = np.abs(w[:, 2]) < 0.15
    if h.sum() < 50: rows.append((P[i, 0], h.sum(), 0, 0, 0)); continue
    a = 4 * np.arctan2(w[h, 1], w[h, 0]); m = np.exp(1j * a).mean()
    rows.append((P[i, 0], h.sum(), abs(m), np.angle(m), z[ok][h].max()))
r = np.array(rows); eng = r[:, 1] >= 50
ref = np.angle((r[eng, 2] ** 2 * np.exp(1j * r[eng, 3])).sum())
dev = np.degrees(np.abs(np.angle(np.exp(1j * (r[:, 3] - ref))))) / 4
nowall = eng & (r[:, 4] < 1.8)
print(f"장면 {len(r)}, 걸림(수평 법선 ≥50, refine 의 160×120 격자 200 점과 같음) {eng.sum()} ({eng.mean()*100:.0f}%)")
for nm, m in (("벽 보임", eng & ~nowall), ("벽 없음(1.8 m 안만)", nowall)):
    if m.sum(): print(f"{nm}: {m.sum()} 장, 집중도 중앙 {np.median(r[m,2]):.2f}, 기준과 차 중앙 {np.median(dev[m]):.1f}° · p90 {np.percentile(dev[m],90):.1f}°")
