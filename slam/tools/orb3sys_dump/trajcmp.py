#!/usr/bin/env python3
"""두 재생 결과(orb3_sys_run TSV) 비교: 장면별 상태 · 지도 · 위치 차, 처음 갈린 장면, 추적 ms."""
import sys, math
def load(p):
    r = []
    for l in open(p):
        f = l.split('\t')
        r.append((f[0], int(f[1]), int(f[2]), [float(x) for x in f[3:15]], float(f[15])))
    return r
a, b = load(sys.argv[1]), load(sys.argv[2])
n = min(len(a), len(b))
first_bit = first_state = None
worst = (0.0, -1)
for i in range(n):
    x, y = a[i], b[i]
    if first_state is None and (x[1] != y[1] or x[2] != y[2]):
        first_state = i
    if first_bit is None and x[3] != y[3]:
        first_bit = i
    d = math.dist([x[3][3], x[3][7], x[3][11]], [y[3][3], y[3][7], y[3][11]])
    if d > worst[0]:
        worst = (d, i)
def ms(r):
    v = sorted(z[4] for z in r)
    return f"평균 {sum(v)/len(v):.2f} 중간 {v[len(v)//2]:.2f} 최대 {v[-1]:.2f}"
print(f"장면 {len(a)} / {len(b)}")
print(f"출력 9 자리 처음 다른 장면 {first_bit}, 상태·지도 처음 다른 장면 {first_state}")
print(f"위치 차 최대 {worst[0]:.4f} m (장면 {worst[1]})")
for k in (500, 1000, 2000, 3000, n - 1):
    if k < n:
        x, y = a[k], b[k]
        print(f"  장면 {k}: 상태 {x[1]}/{y[1]} 지도 {x[2]}/{y[2]} 위치 차 {math.dist([x[3][3],x[3][7],x[3][11]],[y[3][3],y[3][7],y[3][11]]):.4f}")
print("A 추적 ms", ms(a)); print("B 추적 ms", ms(b))
