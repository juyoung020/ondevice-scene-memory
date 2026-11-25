# ydrift.py <poses> <imu.csv>: 짧은 구간(0.2 s) 자세 회전 vs 자이로 적분 비교. R_ci 는 Kabsch 로 추정. 300 장 구간마다 세계 z 방위 누적 차(자세 - 자이로)
import sys,numpy as np
from scipy.spatial.transform import Rotation as R
P=np.loadtxt(sys.argv[1]); I=np.loadtxt(sys.argv[2],delimiter=",",comments="#")
it=I[:,0]*1e-9; w=I[:,1:4]
cw=np.concatenate([[[0,0,0]],np.cumsum(w[1:]*np.diff(it)[:,None],0)])
def gint(a,b):  # 자이로 소구간 회전(IMU 축), 작은 각 합성
    i,j=np.searchsorted(it,[a,b]); r=R.identity()
    for k in range(i,min(j,len(it)-1)): r=r*R.from_rotvec(w[k]*(it[k+1]-it[k]))
    return r
t=P[:,0]; Rw=R.from_quat(P[:,4:8])
pairs=[]
step=6
for k in range(0,len(t)-step,step):
    if t[k+step]-t[k]>0.3: continue
    rc=(Rw[k].inv()*Rw[k+step]).as_rotvec(); ri=gint(t[k],t[k+step]).as_rotvec()
    if np.linalg.norm(ri)>0.05: pairs.append((rc,ri,k))
A=np.array([p[1] for p in pairs]); B=np.array([p[0] for p in pairs])
H=A.T@B; U,S,Vt=np.linalg.svd(H); Rci=Vt.T@U.T
if np.linalg.det(Rci)<0: Vt[2]*=-1; Rci=Vt.T@U.T
res=np.linalg.norm(B-(Rci@A.T).T,axis=1); print("pairs",len(pairs),"resid median deg",np.degrees(np.median(res)))
# 세계 z 방위 비교: 연속 장면마다 자세 dyaw vs 자이로를 카메라→세계로 돌린 z 성분
dy_p=[];dy_g=[];idx=[]
for k in range(len(t)-1):
    if t[k+1]-t[k]>0.15: continue
    i0,i1=np.searchsorted(it,[t[k],t[k+1]])
    if i1>=len(it) or np.max(np.diff(it[max(i0-1,0):i1+1]))>0.02: continue
    rel=(Rw[k].inv()*Rw[k+1]).as_rotvec()
    dp=(Rw[k].as_matrix()@rel)[2]
    g=Rci@gint(t[k],t[k+1]).as_rotvec(); dg=(Rw[k].as_matrix()@g)[2]
    dy_p.append(dp);dy_g.append(dg);idx.append(k)
dy_p=np.array(dy_p);dy_g=np.array(dy_g);idx=np.array(idx)
for s in range(0,len(t),300):
    m=(idx>=s)&(idx<s+300)
    print("frames %4d-%4d n %3d  pose yaw %7.1f  gyro yaw %7.1f  diff %6.1f deg"%(s,s+299,m.sum(),np.degrees(dy_p[m].sum()),np.degrees(dy_g[m].sum()),np.degrees((dy_p[m]-dy_g[m]).sum())))
d=np.degrees(dy_p-dy_g); big=np.where(np.abs(d)>3)[0]
fr=np.loadtxt(__import__("os").path.expanduser("~/rec_dl/ds_1118_121652/cam/rgb.txt"),usecols=0)
print("steps |diff|>3deg:",len(big),"sum",d[big].sum())
for b in big[:60]:
    k=idx[b]; print("row %d frame %d t %.3f diff %.1f pose %.1f gyro %.1f"%(k,np.searchsorted(fr,t[k]-1e-4),t[k],d[b],np.degrees(dy_p[b]),np.degrees(dy_g[b])))
