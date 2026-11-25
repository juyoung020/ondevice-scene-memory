#!/bin/bash
# ?ㅽ봽 ?ъ깮?쇰줈 ?먮낯 rgbd_tum 沅ㅼ쟻??留뚮뱾怨??뺣떟怨?ATE 鍮꾧탳: traj.sh <?ㅽ봽.bin> <?대쫫>
set -e
ORB=${ORB:-$HOME/slam-bench/src/ORB_SLAM3}
S=${SEQ:-$HOME/unify/data/rgbd_dataset_freiburg1_desk}
A=$ORB/Examples/RGB-D/associations/fr1_desk.txt
O=~/orbnpu_out/traj_$2; mkdir -p $O; cd $O
LD_PRELOAD=~/orbnpu_out/liborb3_replay.so ORBNPU_REPLAY=~/orbnpu_out/$1 \
  ${BIN:-$HOME/slam-bench/orb3-nopango/Examples/RGB-D/rgbd_tum} $ORB/Vocabulary/ORBvoc.txt $ORB/Examples/RGB-D/TUM1.yaml $S $A > run.log 2>&1 || true
~/sfm/bin/python - "$S/groundtruth.txt" CameraTrajectory.txt <<'EOF'
import sys, numpy as np
def rd(p):
    d = [l.split() for l in open(p) if l[0] != '#']
    return {float(r[0]): np.array(list(map(float, r[1:4]))) for r in d}
g, e = rd(sys.argv[1]), rd(sys.argv[2])
gt = np.array(sorted(g)); P, Q = [], []
for t, p in e.items():
    i = np.abs(gt - t).argmin()
    if abs(gt[i] - t) < 0.02: P.append(p); Q.append(g[gt[i]])
P, Q = np.array(P), np.array(Q)
mp, mq = P.mean(0), Q.mean(0); U, s, Vt = np.linalg.svd((Q - mq).T @ (P - mp))
D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt
err = np.linalg.norm((R @ (P - mp).T).T + mq - Q, axis=1)
print(f"?λ㈃ {len(P)}  ATE RMSE {np.sqrt((err**2).mean())*100:.2f} cm")
EOF
