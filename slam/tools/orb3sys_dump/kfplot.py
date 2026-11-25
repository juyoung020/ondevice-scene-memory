#!/usr/bin/env python3
"""재생 결과 키프레임(.kf)을 지도별 색으로 위에서 본 그림(x-z) 여럿 나란히: kfplot.py 출력.png 이름=A.tsv.kf ..."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out, items = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
fig, axs = plt.subplots(1, len(items), figsize=(7 * len(items), 7))
for ax, (name, p) in zip(axs if len(items) > 1 else [axs], items):
    maps = {}
    for l in open(p):
        f = l.split("\t"); m = int(f[1]); v = [float(x) for x in f[2:14]]
        maps.setdefault(m, []).append((v[3], v[11]))
    for m, pts in sorted(maps.items()):
        ax.scatter([x for x, _ in pts], [z for _, z in pts], s=6, label=f"지도 {m} ({len(pts)})")
    ax.set_title(f"{name}: 키프레임 {sum(len(v) for v in maps.values())}, 지도 {len(maps)}")
    ax.set_aspect("equal"); ax.legend(fontsize=7); ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
plt.rcParams["font.family"] = "Noto Sans CJK KR"
fig.tight_layout(); fig.savefig(out, dpi=90)
