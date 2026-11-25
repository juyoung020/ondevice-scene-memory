# kf2tum.py <x.tsv> <x.tsv.kf> <map> <out_gt>: 키프레임 끝 자세(그 지도)를 만든 장면 시각으로 TUM groundtruth
import sys, math
tsv, kff, mp, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
first = {}
for l in open(tsv):
    w = l.rstrip('\n').split('\t')
    r = int(w[-1])
    if r not in first: first[r] = w[0]
def quat(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1) * 2; return ((m[2][1]-m[1][2])/s, (m[0][2]-m[2][0])/s, (m[1][0]-m[0][1])/s, s/4)
    i = max(range(3), key=lambda k: m[k][k]); j, k = (i+1) % 3, (i+2) % 3
    s = math.sqrt(1 + m[i][i] - m[j][j] - m[k][k]) * 2
    q = [0]*3; q[i] = s/4; q[j] = (m[j][i]+m[i][j])/s; q[k] = (m[k][i]+m[i][k])/s
    return (q[0], q[1], q[2], (m[k][j]-m[j][k])/s)
n = 0
with open(out, 'w') as g:
    rows = []
    for l in open(kff):
        w = l.split('\t'); k, m = int(w[0]), int(w[1])
        if m != mp or k not in first: continue
        v = list(map(float, w[2:14])); M = [v[0:4], v[4:8], v[8:12]]
        q = quat(M); rows.append((float(first[k]), first[k], M, q))
    for _, t, M, q in sorted(rows):
        g.write(f"{t} {M[0][3]} {M[1][3]} {M[2][3]} {q[0]} {q[1]} {q[2]} {q[3]}\n"); n += 1
print(n)
