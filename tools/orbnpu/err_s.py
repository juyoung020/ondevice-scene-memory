"""NPU(흉내) S 지도와 정확한 FAST 점수의 차 · 후보 비율. 쓰기: err_s.py <mosdir> <mapdir> <n> [margin...]
정확한 S = orbnpu_bench fast_score_cpu 와 같은 정의(127 에서 자름: 모델과 같음). 층 안쪽(테두리 제외)만."""
import sys, numpy as np

OFF = [(0,3),(1,3),(2,2),(3,1),(3,0),(3,-1),(2,-2),(1,-3),(0,-3),(-1,-3),(-2,-2),(-3,-1),(-3,0),(-3,1),(-2,2),(-1,3)]


def exact_s(img):
    H, W = img.shape
    c = img[3:H-3, 3:W-3].astype(np.int16)
    d = np.stack([c - img[3+dy:H-3+dy, 3+dx:W-3+dx].astype(np.int16) for dx, dy in OFF])
    best = np.zeros_like(c)
    for k in range(16):
        idx = [(k + j) % 16 for j in range(9)]
        a = d[idx]
        best = np.maximum(best, np.maximum(a.min(0), -a.max(0)))
    out = np.zeros((H, W), np.int16)
    out[3:H-3, 3:W-3] = np.minimum(best, 127)
    return out


def main():
    md, pd, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    margins = [int(a) for a in sys.argv[4:]] or [0, 1, 2, 3, 4]
    lines = open(f"{md}/layout.txt").read().split("\n")
    W, H = map(int, lines[0].split()[:2])
    slots = [tuple(map(int, l.split())) for l in lines[1:] if l.strip()]
    errs, tot, cand = [], 0, {m: 0 for m in margins}
    for i in range(n):
        img = np.fromfile(f"{md}/f{i:04d}.raw", np.uint8).reshape(H, W)
        S = np.fromfile(f"{pd}/m{i:04d}.bin", np.float32, H * W).reshape(H, W)
        Sh = np.clip(np.rint(S), 0, 255)
        for x, y, w, h in slots:
            e = 19
            a = exact_s(img[y:y+h, x:x+w])[e:h-e, e:w-e]
            b = Sh[y:y+h, x:x+w][e:h-e, e:w-e]
            errs.append(np.abs(a - b).max()); tot += a.size
            for m in margins:
                cand[m] += int((b >= 8 - m).sum())
            if i == 0 and x == 0:
                dd = (b - a).ravel()
                print("층0 차 분포:", {int(k): int(v) for k, v in zip(*np.unique(dd, return_counts=True))})
    print(f"최대 |S_npu - S| = {max(errs)}")
    for m in margins:
        print(f"여유 {m}: 정확 계산 후보 {cand[m] / tot * 100:.1f} % 화소")


if __name__ == "__main__":
    main()
