"""두 점 · 기술자 덤프(orbnpu_bench dump/maps 형식) 비교. 쓰기: cmp_kp.py <기준.bin> <비교.bin>
보고: 점 수, 같은 자리(x, y, 층) 비율, 같은 자리 점의 방향 차 · 반응 차 · 기술자 해밍 거리, 기술자 완전 같음 비율."""
import sys, numpy as np


def load(p):
    b = open(p, "rb").read()
    o, fr = 0, []
    while o < len(b):
        n = int(np.frombuffer(b, np.int32, 1, o)[0]); o += 4
        k = np.frombuffer(b, np.float32, n * 5, o).reshape(n, 5); o += n * 20
        d = np.frombuffer(b, np.uint8, n * 32, o).reshape(n, 32); o += n * 32
        fr.append((k, d))
    return fr


def main():
    A, B = load(sys.argv[1]), load(sys.argv[2])
    na = nb = same = 0
    dang, dresp, ham = [], [], []
    for (ka, da), (kb, db) in zip(A, B):
        na += len(ka); nb += len(kb)
        key = {}
        for i, k in enumerate(ka):
            key.setdefault((round(k[0], 2), round(k[1], 2), int(k[4])), i)
        for j, k in enumerate(kb):
            i = key.get((round(k[0], 2), round(k[1], 2), int(k[4])))
            if i is None:
                continue
            same += 1
            a = abs(ka[i][2] - k[2]) % 360; dang.append(min(a, 360 - a))
            dresp.append(abs(ka[i][3] - k[3]))
            ham.append(int(np.unpackbits(da[i] ^ db[j]).sum()))
    dang, dresp, ham = map(np.array, (dang, dresp, ham))
    print(f"장면 {len(A)}  점 {na} → {nb}  같은 자리 {same / max(na, 1) * 100:.2f} %")
    if same:
        print(f"같은 자리 점: 방향 차 평균 {dang.mean():.3f}° (95 % {np.percentile(dang, 95):.3f}°, 최대 {dang.max():.2f}°)  "
              f"반응 차 평균 {dresp.mean():.3f}  기술자 해밍 평균 {ham.mean():.2f}/256 (중앙 {np.median(ham):.0f}, 95 % {np.percentile(ham, 95):.0f})  "
              f"기술자 완전 같음 {np.mean(ham == 0) * 100:.1f} %")


if __name__ == "__main__":
    main()
