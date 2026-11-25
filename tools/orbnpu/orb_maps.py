"""ORB-SLAM3 ORB 추출의 합성곱 꼴 부분(FAST 점수 S · 7×7 가우시안 블러)을
층 모자이크 한 장에 대해 계산하는 모델. 시험 · 내보내기 도구(저장소 규칙상 파이썬 허용).

쓰기(~/sfm/bin/python, CPU 만):
  orb_maps.py export <out.onnx>                   ONNX(opset 13) 내보내기
  orb_maps.py run <mosdir> <outdir> <n> [--fq] [--start=k]   모자이크 k..n → m%04d.bin([2][H][W] f32: S, blur)
     --fq: 8 비트 흉내(텐서마다 비대칭 u8, 보정 = 고르게 8 장의 최소 · 최대, 무게 채널마다 대칭 int8)
"""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

OFF = [(0,3),(1,3),(2,2),(3,1),(3,0),(3,-1),(2,-2),(1,-3),(0,-3),(-1,-3),(-2,-2),(-3,-1),(-3,0),(-3,1),(-2,2),(-1,3)]  # (dx, dy)
HP = 15
UMAX = None


def umax():
    # ORBextractor 생성자와 같은 원형 조각 반폭
    import math
    u = [0] * (HP + 1)
    vmax = int(math.floor(HP * math.sqrt(2.0) / 2 + 1))
    vmin = int(math.ceil(HP * math.sqrt(2.0) / 2))
    hp2 = HP * HP
    for v in range(vmax + 1):
        u[v] = int(round(math.sqrt(hp2 - v * v)))
    v0 = 0
    for v in range(HP, vmin - 1, -1):
        while u[v0] == u[v0 + 1]:
            v0 += 1
        u[v] = v0
        v0 += 1
    return u


def gauss7():
    import cv2
    k = cv2.getGaussianKernel(7, 2.0, cv2.CV_64F)[:, 0]
    return np.outer(k, k).astype(np.float32)


class OrbMaps(nn.Module):
    """출력 S(FAST 점수, 127 에서 포화) · blur(7×7 가우시안). 판 1(모멘트 m10/m01 포함 · 차이값 ±255)은 8 비트 흉내에서
    각도 오차가 커서(평균 2.4°, 해밍 8.5) 방향은 CPU 에 남기고 이 판(2)으로 바꿨다 — docs/orb3_port.md "포터 NPU"."""

    def __init__(self):
        super().__init__()
        wd = torch.zeros(32, 1, 7, 7)
        for k, (dx, dy) in enumerate(OFF):
            wd[k, 0, 3, 3] += 1; wd[k, 0, 3 + dy, 3 + dx] -= 1      # 가운데 − 이웃
            wd[16 + k, 0, 3, 3] -= 1; wd[16 + k, 0, 3 + dy, 3 + dx] += 1  # 이웃 − 가운데
        self.register_buffer("wd", wd)
        for s in (1, 2, 4, 8):  # 두 묶음(16 채널)마다 채널을 s 만큼 돌리는 1×1 합성곱
            p = torch.zeros(32, 32, 1, 1)
            for g in (0, 16):
                for i in range(16):
                    p[g + i, g + (i + s) % 16] = 1
            self.register_buffer(f"r{s}", p)
        for g in (0, 16):  # 묶음마다 채널 하나 고르기(Slice 대신 1×1 합성곱)
            sel = torch.zeros(1, 32, 1, 1); sel[0, g] = 1
            self.register_buffer(f"sel{g}", sel)
        g = gauss7()
        gi = np.round(g / g.max() * 127)  # 정수 무게(최대 127): int8 무게 눈금에 정확히 맞음
        self.register_buffer("wg", torch.from_numpy((gi / gi.sum()).astype(np.float32))[None, None])
        self.q = None  # 8 비트 흉내 훅

    def Q(self, t, name):
        return self.q(t, name) if self.q else t

    def forward(self, x):
        Q = self.Q
        # 모든 값이 [0,127] · 차이 [-127,127] 안: 8 비트 눈금 ≈ 1
        mn = lambda a, b, n: Q(a - Q(F.relu(Q(a - b, n + "s")), n + "r"), n)
        mx = lambda a, b, n: Q(b + Q(F.relu(Q(a - b, n + "s")), n + "r"), n)
        r = lambda t, s, n: Q(F.conv2d(t, getattr(self, f"r{s}")), n)
        d = Q(F.relu(Q(F.conv2d(x, self.wd, padding=3), "d")), "dp")  # 이웃이 어둡/밝은 정도(음수는 0: 호 최솟값이 양수일 때만 구석)
        d = Q(127 - Q(F.relu(Q(127 - d, "c0")), "c1"), "dc")
        m2 = mn(d, r(d, 1, "d1"), "m2")
        m4 = mn(m2, r(m2, 2, "m2r"), "m4")
        m8 = mn(m4, r(m4, 4, "m4r"), "m8")
        m9 = mn(m8, r(d, 8, "d8"), "m9")
        a = mx(m9, r(m9, 8, "a8"), "a")
        b = mx(a, r(a, 4, "b4"), "b")
        c = mx(b, r(b, 2, "c2_"), "c")
        e = mx(c, r(c, 1, "e1"), "e")
        s = mx(Q(F.conv2d(e, self.sel0), "p"), Q(F.conv2d(e, self.sel16), "n"), "S")
        blur = Q(F.conv2d(x, self.wg, padding=3), "blur")
        return s, blur


def load_mos(d, i, W, H):
    a = np.fromfile(f"{d}/f{i:04d}.raw", np.uint8).reshape(H, W)
    return torch.from_numpy(a.astype(np.float32))[None, None]


class FQ:
    """텐서마다 최소 · 최대 보정 후 비대칭 u8 흉내"""
    def __init__(self):
        self.rng, self.cal = {}, True

    def __call__(self, t, name):
        if self.cal:
            lo, hi = float(t.min()), float(t.max())
            o = self.rng.get(name, (0.0, 0.0))
            self.rng[name] = (min(o[0], lo), max(o[1], hi))
            return t
        lo, hi = self.rng[name]
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        sc = (hi - lo) / 255.0 or 1.0
        zp = round(-lo / sc)
        return (torch.clamp(torch.round(t / sc) + zp, 0, 255) - zp) * sc


def main():
    cmd = sys.argv[1]
    m = OrbMaps().eval()
    if cmd == "export":
        out = sys.argv[2]
        x = torch.zeros(1, 1, 1120, 1280)
        torch.onnx.export(m, x, out, input_names=["mosaic"], output_names=["S", "blur"], opset_version=13, dynamo=False)
        print(out)
        return
    if cmd == "run":
        md, od, n = sys.argv[2], sys.argv[3], int(sys.argv[4])
        os.makedirs(od, exist_ok=True)
        W, H = map(int, open(f"{md}/layout.txt").readline().split()[:2])
        torch.set_num_threads(8)
        with torch.no_grad():
            if "--fq" in sys.argv:
                fq = FQ(); m.q = fq
                # 무게 흉내: 채널마다 대칭 int8
                for nm in ("wd", "wg", "r1", "r2", "r4", "r8", "sel0", "sel16"):
                    w = getattr(m, nm)
                    mxv = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-12)
                    setattr(m, nm, torch.round(w / mxv * 127) * mxv / 127)
                for i in range(0, 573, 573 // 8):  # 보정 8 장
                    m(load_mos(md, i, W, H))
                fq.cal = False
            st = next((int(a[8:]) for a in sys.argv if a.startswith("--start=")), 0)
            for i in range(st, n):
                s, b = m(load_mos(md, i, W, H))
                np.concatenate([s[0], b[0]], 0).astype(np.float32).tofile(f"{od}/m{i:04d}.bin")
        return


if __name__ == "__main__":
    main()
