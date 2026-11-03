"""ADVIO 한 시퀀스(아이폰 카메라 + IMU) → EuRoC 형식(ORB-SLAM3·OpenVINS 등 VIO 입력) + ORB-SLAM3 보정 YAML.

  python tools/advio_to_euroc.py data/advio/advio-15 data/euroc/advio15 [--every 2]

만드는 것:
  <out>/mav0/cam0/data/<ns>.png   흑백 세로 영상(720×1280), --every 프레임마다(2 = 30 fps)
  <out>/mav0/cam0/data.csv        #timestamp [ns],filename
  <out>/mav0/imu0/data.csv        #timestamp [ns],w_x,w_y,w_z [rad/s],a_x,a_y,a_z [m/s²]  (자이로 시각, 가속도는 선형 보간)
  <out>/times.txt                 프레임 시각(ns) 한 줄씩(ORB-SLAM3 EuRoC 예제 입력)
  <out>/orbslam3_advio.yaml       카메라·IMU 보정(ADVIO calibration/iphone-0N.yaml + README IMU 잡음)
보정 묶음(ADVIO README): 1–12 → iphone-02, 13–17 → iphone-03, 18–19 → iphone-01, 20–23 → iphone-04.
"""
import argparse
import os

import cv2
import numpy as np

CALIB = {  # fx, fy, cx, cy, k1, k2, p1, p2 (세로 720×1280 화소), T_cam_imu (Kalibr: p_cam = T·p_imu)
    'iphone-03': ((1082.4, 1084.4, 364.6778, 643.3080), (0.0366, 0.0803, 0.000783, -0.000215),
                  [[0.9999763379093255, -0.004079205042965442, -0.005539287650170447, -0.008977668364731128],
                   [-0.004066386342107199, -0.9999890330121858, 0.0023234365646622014, 0.07557012320238939],
                   [-0.00554870467502187, -0.0023008567036498766, -0.9999819588046867, -0.005545773942541918],
                   [0, 0, 0, 1]]),
}
# ADVIO README: 아이폰 IMU 잡음(연속 시간)·바이어스 걸음
IMU_NOISE = dict(NoiseGyro=2.4e-3, NoiseAcc=4.8e-3, GyroWalk=5.1e-5, AccWalk=2.1e-4, Frequency=100.0)


def batch_of(seq):
    n = int(''.join(c for c in os.path.basename(seq.rstrip('/\\')) if c.isdigit()))
    return 'iphone-02' if n <= 12 else 'iphone-03' if n <= 17 else 'iphone-01' if n <= 19 else 'iphone-04'


def write_yaml(path, calib, fps):
    (fx, fy, cx, cy), (k1, k2, p1, p2), T_cam_imu = calib
    T_b_c = np.linalg.inv(np.array(T_cam_imu))  # ORB-SLAM3 IMU.T_b_c1: 카메라 → 몸통(IMU)
    rows = ', '.join(f'{v:.10f}' for v in T_b_c.ravel())
    with open(path, 'w') as f:
        f.write(f"""%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: {fx}
Camera1.fy: {fy}
Camera1.cx: {cx}
Camera1.cy: {cy}
Camera1.k1: {k1}
Camera1.k2: {k2}
Camera1.p1: {p1}
Camera1.p2: {p2}
Camera.width: 720
Camera.height: 1280
Camera.newWidth: 720
Camera.newHeight: 1280
Camera.fps: {int(round(fps))}
Camera.RGB: 1
IMU.T_b_c1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [{rows}]
IMU.NoiseGyro: {IMU_NOISE['NoiseGyro']}
IMU.NoiseAcc: {IMU_NOISE['NoiseAcc']}
IMU.GyroWalk: {IMU_NOISE['GyroWalk']}
IMU.AccWalk: {IMU_NOISE['AccWalk']}
IMU.Frequency: {IMU_NOISE['Frequency']}
ORBextractor.nFeatures: 1500
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -3.5
Viewer.ViewpointF: 500.0
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('seq')
    ap.add_argument('out')
    ap.add_argument('--every', type=int, default=2)
    a = ap.parse_args()
    cam = os.path.join(a.out, 'mav0', 'cam0', 'data')
    os.makedirs(cam, exist_ok=True)
    os.makedirs(os.path.join(a.out, 'mav0', 'imu0'), exist_ok=True)
    fr = np.loadtxt(os.path.join(a.seq, 'iphone', 'frames.csv'), delimiter=',', ndmin=2)
    cap = cv2.VideoCapture(os.path.join(a.seq, 'iphone', 'frames.mov'))
    names, i = [], -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        i += 1
        if i % a.every or i >= len(fr):
            continue
        if img.shape[1] > img.shape[0]:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        ns = int(round(fr[i, 0] * 1e9))
        cv2.imwrite(os.path.join(cam, f'{ns}.png'), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        names.append(ns)
    with open(os.path.join(a.out, 'mav0', 'cam0', 'data.csv'), 'w') as f:
        f.write('#timestamp [ns],filename\n')
        f.writelines(f'{n},{n}.png\n' for n in names)
    with open(os.path.join(a.out, 'times.txt'), 'w') as f:
        f.writelines(f'{n}\n' for n in names)
    gy = np.loadtxt(os.path.join(a.seq, 'iphone', 'gyro.csv'), delimiter=',', ndmin=2)
    ac = np.loadtxt(os.path.join(a.seq, 'iphone', 'accelerometer.csv'), delimiter=',', ndmin=2)
    if np.median(np.linalg.norm(ac[:, 1:4], axis=1)) < 3:  # g 단위면 m/s² 로(README 는 g, 실제 파일은 m/s²)
        ac[:, 1:4] *= 9.80665
    t = gy[(gy[:, 0] >= ac[0, 0]) & (gy[:, 0] <= ac[-1, 0]), 0]
    g = np.stack([np.interp(t, gy[:, 0], gy[:, k]) for k in (1, 2, 3)], 1)
    acc = np.stack([np.interp(t, ac[:, 0], ac[:, k]) for k in (1, 2, 3)], 1)
    with open(os.path.join(a.out, 'mav0', 'imu0', 'data.csv'), 'w') as f:
        f.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],'
                'a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n')
        for k in range(len(t)):
            f.write(f'{int(round(t[k] * 1e9))},' + ','.join(f'{v:.6f}' for v in (*g[k], *acc[k])) + '\n')
    b = batch_of(a.seq)
    write_yaml(os.path.join(a.out, 'orbslam3_advio.yaml'), CALIB[b], 60.0 / a.every)
    print(f'{a.out}: {len(names)} frames, {len(t)} IMU samples, calibration {b}')


if __name__ == '__main__':
    main()
