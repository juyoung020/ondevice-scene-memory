"""영상 + 카메라 포즈로 지도 갱신·물체 기억을 돌려 sgview 기억 폴더를 쓴다(학습 없음, 추론만).

  python tools/run_memory.py --advio data/advio/advio-15 --out data/memory/advio15 [--pose arkit|gt] [--every 10]
  viewer/sgview/target/release/sgview data/memory/advio15        → http://localhost:8080 (돌아가는 중에도 갱신)

한 키프레임:
  1) 포즈: 카메라+IMU 추적(ARKit = 아이폰 VIO; 우리 VIO 가 생기면 그 출력으로 바꾼다). 정답(gt)은 평가용.
  2) 분할: YOLO26s-seg(COCO 사전학습, 이름 있음) → 물체 마스크
  3) 깊이: YOLO26s-depth(미터)
  4) 지도: 깊이 점 → 바닥 높이 띠는 빈칸, 장애물 높이 띠는 막힌 칸(5 cm 격자), 카메라→점 사이는 빈칸
  5) 물체 기억: 마스크 화소 → 3D 점 → 같은 이름·가까운 물체면 합치고, 아니면 새 물체.
     기억 속 물체가 화면 안·가려지지 않은 거리인데 3 번 연속 안 보이면 "gone".
끝에 정답 포즈와 비교한 위치 오차(ATE)를 낸다.
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

# ---------------------------------------------------------------- ADVIO 읽기


def read_csv(path):
    return np.loadtxt(path, delimiter=',', ndmin=2)


def quat_to_R(q):  # q = (w, x, y, z)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp_pose(tab, t):
    """tab: [t, x, y, z, qw, qx, qy, qz] 정렬됨 → t 에서 보간한 (p, R)."""
    i = np.searchsorted(tab[:, 0], t)
    i = min(max(i, 1), len(tab) - 1)
    a, b = tab[i - 1], tab[i]
    u = 0.0 if b[0] == a[0] else float(np.clip((t - a[0]) / (b[0] - a[0]), 0, 1))
    p = a[1:4] * (1 - u) + b[1:4] * u
    qa, qb = a[4:8] / np.linalg.norm(a[4:8]), b[4:8] / np.linalg.norm(b[4:8])
    if np.dot(qa, qb) < 0:
        qb = -qb
    q = qa * (1 - u) + qb * u
    return p, quat_to_R(q)


def load_advio(d):
    fr = read_csv(os.path.join(d, 'iphone', 'frames.csv'))
    ark = read_csv(os.path.join(d, 'iphone', 'arkit.csv'))
    gt = read_csv(os.path.join(d, 'ground-truth', 'pose.csv') if os.path.exists(os.path.join(d, 'ground-truth', 'pose.csv')) else os.path.join(d, 'ground-truth', 'poses.csv'))
    return fr, ark, gt


# ---------------------------------------------------------------- 기하


def umeyama(src, dst, scale=False):
    """dst ≈ s·R·src + t (점 대응). 위치 오차 평가용."""
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / xs.var(0).sum() if scale else 1.0
    return s, R, md - s * R @ ms


def voxel_down(pts, cols, v):
    if len(pts) == 0:
        return pts, cols
    k = np.floor(pts / v).astype(np.int64)
    _, idx = np.unique(k, axis=0, return_index=True)
    return pts[idx], cols[idx]


def write_ply(path, pts, cols):
    with open(path, 'wb') as f:
        f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n'
                 'property float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        rec = np.zeros(len(pts), dtype=[('p', '<f4', 3), ('c', 'u1', 3)])
        rec['p'], rec['c'] = pts, cols
        f.write(rec.tobytes())


# ---------------------------------------------------------------- 지도·물체 기억


class Grid:
    """2D 점유 격자(x, y 평면, z 위). 막힘·빔 횟수로 판정, 처음 보는 영역은 자란다."""

    def __init__(self, res=0.05):
        self.res = res
        self.hit = {}
        self.free = {}

    def add(self, cam_xy, occ_xy, floor_xy):
        """키프레임마다 칸 하나에 한 번씩만 센다(막힘·빔 둘 다) — 점 수가 많은 쪽이 이기지 않게."""
        r = self.res
        occ = {tuple(c) for c in np.floor(occ_xy / r).astype(np.int64).tolist()}
        free = {tuple(c) for c in np.floor(floor_xy / r).astype(np.int64).tolist()}
        # 카메라 → 장애물 칸 사이는 빈칸(장애물 칸 자체는 빼고)
        cx, cy = np.floor(cam_xy / r).astype(np.int64)
        for x, y in occ:
            n = max(abs(x - cx), abs(y - cy))
            for k in range(n - 1):
                free.add((int(cx + (x - cx) * k / n), int(cy + (y - cy) * k / n)))
        free -= occ
        for c in occ:
            self.hit[c] = self.hit.get(c, 0) + 1
        for c in free:
            self.free[c] = self.free.get(c, 0) + 1

    def write(self, out):
        keys = list(self.hit) + list(self.free)
        if not keys:
            return None
        a = np.array(keys)
        x0, y0 = a.min(0) - 5
        x1, y1 = a.max(0) + 5
        w, h = int(x1 - x0 + 1), int(y1 - y0 + 1)
        img = np.full((h, w), 205, np.uint8)
        for (x, y), n in self.free.items():
            img[h - 1 - (y - y0), x - x0] = 254
        for (x, y), n in self.hit.items():
            if n >= 2 and n >= self.free.get((x, y), 0):  # 막힘으로 본 키프레임이 빔으로 본 것 이상
                img[h - 1 - (y - y0), x - x0] = 0
        with open(os.path.join(out, 'map.pgm.tmp'), 'wb') as f:
            f.write(f'P5\n{w} {h}\n255\n'.encode())
            f.write(img.tobytes())
        os.replace(os.path.join(out, 'map.pgm.tmp'), os.path.join(out, 'map.pgm'))
        return {'width': w, 'height': h, 'resolution': self.res, 'origin': [float(x0 * self.res), float(y0 * self.res)]}


class Memory:
    def __init__(self, out):
        self.out = out
        os.makedirs(os.path.join(out, 'objects'), exist_ok=True)
        self.objs = []
        self.next_id = 0

    def match(self, name, c, ext):
        best, bd = None, 1e9
        for o in self.objs:
            if o['name'] != name:
                continue
            d = np.linalg.norm(o['pos'] - c)
            gate = max(0.35, 0.5 * float(np.linalg.norm(np.maximum(o['extent'], ext))))
            if d < gate and d < bd:
                best, bd = o, d
        return best

    def update(self, name, score, pts, cols, crop, stamp):
        c = np.median(pts, 0)
        lo, hi = np.percentile(pts, 5, 0), np.percentile(pts, 95, 0)
        ext = np.maximum(hi - lo, 0.05)
        o = self.match(name, c, ext)
        if o is None:
            o = {'id': self.next_id, 'name': name, 'first_pos': c.copy(), 'n_obs': 0, 'score': 0.0, 'miss': 0,
                 'pts': np.zeros((0, 3)), 'cols': np.zeros((0, 3), np.uint8), 'crop_area': 0, 'state': 'seen'}
            self.next_id += 1
            self.objs.append(o)
        p, cl = voxel_down(np.vstack([o['pts'], pts]), np.vstack([o['cols'], cols]), 0.02)
        if len(p) > 4000:
            sel = np.random.default_rng(0).choice(len(p), 4000, replace=False)
            p, cl = p[sel], cl[sel]
        o['pts'], o['cols'] = p, cl
        o['pos'] = np.median(p, 0)
        o['extent'] = np.maximum(np.percentile(p, 95, 0) - np.percentile(p, 5, 0), 0.05)
        o['n_obs'] += 1
        o['score'] = max(o['score'], score)
        o['last_seen'] = stamp
        o['miss'] = 0
        o['state'] = 'seen'
        o['dirty'] = True
        if crop is not None and crop[0].shape[0] * crop[0].shape[1] > o['crop_area']:
            o['crop_area'] = crop[0].shape[0] * crop[0].shape[1]
            o['crop'] = crop
        return o

    def check_missing(self, seen_ids, K, Rwc, pwc, depth, shape):
        """화면 안, 앞에 가린 것 없는 거리인데 이번에 못 본 물체 → miss, 3 번이면 gone."""
        h, w = shape
        for o in self.objs:
            if o['id'] in seen_ids or o['state'] == 'gone':
                continue
            pc = Rwc.T @ (o['pos'] - pwc)
            if pc[2] < 0.3 or pc[2] > 4.0:
                continue
            u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
            if not (0.1 * w < u < 0.9 * w and 0.1 * h < v < 0.9 * h):
                continue
            dz = depth[int(v), int(u)]
            if dz < pc[2] - 0.4:  # 앞에 다른 것이 가림
                continue
            o['miss'] += 1
            if o['miss'] >= 3:
                o['state'] = 'gone'

    def write(self, stamp, pose, traj, grid):
        objs = []
        for o in self.objs:
            oid = o['id']
            base = f'objects/O{oid}'
            if o.get('dirty'):
                write_ply(os.path.join(self.out, base + '_points.ply'), o['pts'].astype(np.float32), o['cols'])
                if 'crop' in o:
                    rgb, mask, dep = o['crop']
                    cv2.imwrite(os.path.join(self.out, base + '_rgb.png'), rgb)
                    cv2.imwrite(os.path.join(self.out, base + '_mask.png'), mask)
                    cv2.imwrite(os.path.join(self.out, base + '_depth.png'), dep)
                o['dirty'] = False
                o['stamp'] = o.get('stamp', 0) + 1
            e = {'id': oid, 'name': o['name'], 'state': o['state'], 'pos': o['pos'].tolist(),
                 'first_pos': o['first_pos'].tolist(), 'extent': o['extent'].tolist(), 'n_obs': o['n_obs'],
                 'score': float(o['score']), 'last_seen': float(o['last_seen']), 'structural': False,
                 'points': {'path': base + '_points.ply', 'stamp': o['stamp'], 'n': int(len(o['pts'])), 'voxel': 0.02}}
            if 'crop' in o:
                e['rgbd'] = {'rgb': base + '_rgb.png', 'mask': base + '_mask.png', 'depth': base + '_depth.png'}
            objs.append(e)
        nodes = [{'id': f'A{i}', 'kind': 'agent', 'pos': p, 't': t} for i, (t, p) in enumerate(traj)]
        view = {'stamp': float(stamp), 'pose': pose, 'grid': grid, 'objects': objs, 'rooms': [],
                'graph': {'nodes': nodes, 'edges': []}, 'robot': 'handheld camera'}
        tmp = os.path.join(self.out, 'view.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(view, f)
        os.replace(tmp, os.path.join(self.out, 'view.json'))


# ---------------------------------------------------------------- 실행


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--advio', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pose', default='arkit', choices=['arkit', 'gt'])
    ap.add_argument('--every', type=int, default=10, help='키프레임 간격(프레임, 60 fps 기준 10 = 6 Hz)')
    ap.add_argument('--max', type=int, default=0, help='키프레임 수 상한(0 = 끝까지)')
    ap.add_argument('--seg', default='models/yolo26s-seg.pt')
    ap.add_argument('--depth', default='models/yolo26s-depth.pt')
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--calib', default='', help='fx,fy,cx,cy (세로 영상 화소). 비우면 ADVIO 13-17 묶음 값')
    a = ap.parse_args()
    from ultralytics import YOLO
    seg = YOLO(a.seg)
    dep = YOLO(a.depth)
    fr, ark, gt = load_advio(a.advio)
    track = ark if a.pose == 'arkit' else gt
    fx, fy, cx, cy = (map(float, a.calib.split(',')) if a.calib else (1082.4, 1084.4, 364.68, 643.31))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    os.makedirs(a.out, exist_ok=True)
    mem, grid = Memory(a.out), Grid()
    cap = cv2.VideoCapture(os.path.join(a.advio, 'iphone', 'frames.mov'))
    traj, est, ref = [], [], []
    # 좌표: ARKit/ADVIO 세계는 y 위(중력 반대). 카메라는 OpenGL(x 오른쪽, y 위, −z 앞).
    # 뷰어는 z 위 → W = (x, −z, y). 우리 역투영은 OpenCV 카메라(x 오른쪽, y 아래, z 앞) → GL = diag(1, −1, −1)·CV.
    to_view = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    cv2gl = np.diag([1.0, -1.0, -1.0])
    floor_z = None
    t0 = time.time()
    nkf = 0
    fi = -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        fi += 1
        if fi % a.every or fi >= len(fr):
            continue
        t = fr[fi, 0]
        if t < track[0, 0] or t > track[-1, 0]:
            continue
        if img.shape[1] > img.shape[0]:  # 가로로 읽히면 세로로(보정값은 세로 화소)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        H, W = img.shape[:2]
        p, R = slerp_pose(track, t)
        Rwc = to_view @ R @ cv2gl            # OpenCV 카메라 → 뷰어 세계
        pwc = to_view @ p
        pg, Rg = slerp_pose(gt, t)
        est.append(pwc)
        ref.append(to_view @ pg)
        # 깊이(미터)·분할
        d = dep(img, imgsz=768, verbose=False)[0].depth.data.cpu().numpy()
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
        r = seg(img, imgsz=640, conf=a.conf, retina_masks=True, verbose=False)[0]
        # 지도: 성긴 깊이 점
        ys, xs = np.mgrid[8:H:16, 8:W:16]
        z = d[ys, xs].ravel()
        good = (z > 0.2) & (z < 6.0)
        Pc = np.stack([(xs.ravel() - cx) / fx * z, (ys.ravel() - cy) / fy * z, z], 1)[good]
        Pw = Pc @ Rwc.T + pwc
        if floor_z is None:
            floor_z = float(np.percentile(Pw[:, 2], 3))
        else:
            floor_z = 0.95 * floor_z + 0.05 * float(np.percentile(Pw[:, 2], 3))
        hgt = Pw[:, 2] - floor_z
        grid.add(pwc[:2], Pw[(hgt > 0.15) & (hgt < 1.8)][:, :2], Pw[hgt < 0.08][:, :2])
        # 물체
        seen = set()
        if r.masks is not None:
            names = r.names
            for k in range(len(r.boxes)):
                name = names[int(r.boxes.cls[k])]
                if name == 'person':
                    continue
                m = r.masks.data[k].cpu().numpy() > 0.5
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                m = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0  # 가장자리 깊이 번짐 제거
                vv, uu = np.nonzero(m)
                if len(vv) < 80:
                    continue
                if len(vv) > 2500:
                    s = np.random.default_rng(fi).choice(len(vv), 2500, replace=False)
                    vv, uu = vv[s], uu[s]
                zz = d[vv, uu]
                med = np.median(zz)
                keep = (zz > 0.2) & (np.abs(zz - med) < 0.25 * med + 0.1)
                if keep.sum() < 50 or med > 5.0:
                    continue
                vv, uu, zz = vv[keep], uu[keep], zz[keep]
                pc = np.stack([(uu - cx) / fx * zz, (vv - cy) / fy * zz, zz], 1)
                pw = pc @ Rwc.T + pwc
                cols = img[vv, uu][:, ::-1].copy()
                x1, y1, x2, y2 = r.boxes.xyxy[k].cpu().numpy().astype(int)
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
                crop = (img[y1:y2, x1:x2].copy(), (m[y1:y2, x1:x2] * 255).astype(np.uint8),
                        np.clip(d[y1:y2, x1:x2] * 1000, 0, 65535).astype(np.uint16))
                o = mem.update(name, float(r.boxes.conf[k]), pw, cols, crop, t)
                seen.add(o['id'])
        mem.check_missing(seen, K, Rwc, pwc, d, (H, W))
        yaw = math.atan2(Rwc[1, 2], Rwc[0, 2])  # 카메라 앞(z)의 바닥 방향
        traj.append((float(t), [float(pwc[0]), float(pwc[1]), float(floor_z)]))
        nkf += 1
        if nkf % 5 == 0 or nkf == 1:
            g = grid.write(a.out)
            mem.write(t, [float(pwc[0]), float(pwc[1]), yaw], traj, g)
            print(f'kf {nkf} t {t:.1f}s objects {len(mem.objs)} ({sum(o["state"] == "gone" for o in mem.objs)} gone) '
                  f'{(time.time() - t0) / nkf:.2f} s/kf', flush=True)
        if a.max and nkf >= a.max:
            break
    g = grid.write(a.out)
    mem.write(traj[-1][0], [float(pwc[0]), float(pwc[1]), yaw], traj, g)
    est, ref = np.array(est), np.array(ref)
    s, Rr, tr = umeyama(est, ref)
    ate = np.sqrt(((est @ Rr.T * s + tr - ref) ** 2).sum(1).mean())
    path = np.linalg.norm(np.diff(ref, axis=0), axis=1).sum()
    names = {}
    for o in mem.objs:
        names[o['name']] = names.get(o['name'], 0) + 1
    summary = {'keyframes': nkf, 'pose': a.pose, 'ate_rmse_m': float(ate), 'path_m': float(path),
               'ate_percent_of_path': float(100 * ate / max(path, 1e-6)), 'objects': len(mem.objs),
               'gone': sum(o['state'] == 'gone' for o in mem.objs), 'by_name': names}
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
