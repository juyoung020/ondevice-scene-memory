//! 정답 궤적(TUM groundtruth.txt: 카메라 광학 중심 → 세계, 세계 z 위)으로 IMU 녹화(Kalibr imu0.csv)를 만든다: 실제 녹화 없이 IMU 융합 경로 시험.
//!   imu_synth <groundtruth.txt> <출력 imu0.csv> [--calib docs/imu_cam_calib.yaml] [--rate 200] [--seed 1]
//!             [--smooth-rot 0.008] [--smooth-pos 0.04] [--noise 1] [--offset 0] [--gyro-bias 0.008,-0.005,0.004] [--acc-bias 0.05,-0.03,0.08]
//! - 정답을 200 Hz 로 보간(위치 선형, 회전 slerp, 정답 간격이 50 ms 넘는 곳은 표본 없음) 뒤 가우스로 고름(모션 캡처 떨림이 미분에서 커지지 않게)
//! - IMU 자세 = 카메라 자세 × T_cam_imu⁻¹(Kalibr: p_cam = R_ci p_imu + t). 자이로 = 앞 표본 → 이 표본 사이 IMU 회전 / dt(표본 i 의 값이 (t_{i-1}, t_i] 동안: 위치 추적의 적분 약속과 같음)
//! - 가속도 = R_wiᵀ(IMU 위치 2 차 미분 + 중력 위 9.80665)
//! - 잡음(ICM-42688-P 자료표 수준, 배율 --noise): 자이로 0.00049 rad/s/√Hz · 치우침 걸음 0.00002 rad/s²/√Hz, 가속도 0.0069 m/s²/√Hz · 걸음 0.0002, 상수 치우침
//! - 시각 = 정답 시각 + timeshift_cam_imu(t_imu = t_cam + shift)
type V3 = [f64; 3];
type M3 = [[f64; 3]; 3];

fn mm(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
fn mt(a: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}
fn mv(a: &M3, v: &V3) -> V3 {
    std::array::from_fn(|i| (0..3).map(|k| a[i][k] * v[k]).sum())
}
fn exp(w: V3) -> M3 {
    let th = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    let k = [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]];
    let (a, b) = if th < 1e-9 { (1.0, 0.5) } else { (th.sin() / th, (1.0 - th.cos()) / (th * th)) };
    let k2 = mm(&k, &k);
    std::array::from_fn(|i| std::array::from_fn(|j| (i == j) as i32 as f64 + a * k[i][j] + b * k2[i][j]))
}
fn log(m: &M3) -> V3 {
    let c = ((m[0][0] + m[1][1] + m[2][2] - 1.0) * 0.5).clamp(-1.0, 1.0);
    let th = c.acos();
    let k = if th < 1e-7 { 0.5 } else { 0.5 * th / th.sin() };
    [k * (m[2][1] - m[1][2]), k * (m[0][2] - m[2][0]), k * (m[1][0] - m[0][1])]
}
fn quat_m(q: [f64; 4]) -> M3 {
    let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    let (x, y, z, w) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// 재현되는 정규 난수(SplitMix64 + Box–Muller)
struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        let (a, b) = (self.u().max(1e-300), self.u());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
}

/// Kalibr yaml 의 T_cam_imu(회전 · 이동) · timeshift_cam_imu
fn read_calib(path: &str) -> (M3, V3, f64) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut rows: Vec<Vec<f64>> = vec![];
    let (mut in_t, mut shift) = (false, 0.0);
    for l in text.lines() {
        let s = l.trim();
        if s.starts_with("T_cam_imu:") {
            in_t = true;
            continue;
        }
        if in_t && s.starts_with('-') && rows.len() < 3 {
            rows.push(s.trim_start_matches('-').trim().trim_matches(|c| c == '[' || c == ']').split(',').map(|x| x.trim().parse().unwrap()).collect());
            continue;
        }
        in_t = false;
        if let Some(v) = s.strip_prefix("timeshift_cam_imu:") {
            shift = v.trim().parse().unwrap();
        }
    }
    let r = std::array::from_fn(|i| std::array::from_fn(|j| rows[i][j]));
    (r, [rows[0][3], rows[1][3], rows[2][3]], shift)
}

fn arg<'a>(a: &'a [String], k: &str) -> Option<&'a str> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).map(|s| s.as_str())
}
fn v3(s: &str) -> V3 {
    let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect();
    [v[0], v[1], v[2]]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("imu_synth <groundtruth.txt> <출력 imu0.csv> [--calib yaml] [--rate 200] [--seed 1] [--smooth-rot 0.008] [--smooth-pos 0.04] [--noise 1] [--offset 0] [--gyro-bias x,y,z] [--acc-bias x,y,z]");
        std::process::exit(2);
    }
    let calib = arg(&a, "--calib").unwrap_or("docs/imu_cam_calib.yaml");
    let hz: f64 = arg(&a, "--rate").map(|s| s.parse().unwrap()).unwrap_or(200.0);
    let mut rng = Rng(arg(&a, "--seed").map(|s| s.parse().unwrap()).unwrap_or(1));
    let s_rot: f64 = arg(&a, "--smooth-rot").map(|s| s.parse().unwrap()).unwrap_or(0.008);
    let s_pos: f64 = arg(&a, "--smooth-pos").map(|s| s.parse().unwrap()).unwrap_or(0.04);
    let ns: f64 = arg(&a, "--noise").map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let bg0 = arg(&a, "--gyro-bias").map(v3).unwrap_or([0.008, -0.005, 0.004]);
    let ba0 = arg(&a, "--acc-bias").map(v3).unwrap_or([0.05, -0.03, 0.08]);
    // 정답 시계 → 영상 시계(초): 정답 시각 + offset = 영상 시각
    let offset: f64 = arg(&a, "--offset").map(|s| s.parse().unwrap()).unwrap_or(0.0);
    let (r_ci, t_ci, shift) = read_calib(calib);

    // 정답 읽기(사원수 부호를 앞 것과 같은 쪽으로)
    let text = std::fs::read_to_string(&a[1]).unwrap_or_else(|e| panic!("{}: {e}", a[1]));
    let mut gt: Vec<(f64, V3, [f64; 4])> = vec![];
    for v in common::tum::rows(&text).filter(|v| v.len() >= 8) {
        let mut q = [v[4], v[5], v[6], v[7]];
        if let Some(p) = gt.last() {
            if (0..4).map(|i| p.2[i] * q[i]).sum::<f64>() < 0.0 {
                q = q.map(|x| -x);
            }
        }
        gt.push((v[0], [v[1], v[2], v[3]], q));
    }
    // 200 Hz 칸에 보간: (시각, 카메라 회전, 카메라 위치) 또는 빈칸(정답 간격 50 ms 넘음)
    let dt = 1.0 / hz;
    let (t0, t1) = (gt[0].0, gt[gt.len() - 1].0);
    let n = ((t1 - t0) / dt).floor() as usize + 1;
    let mut grid: Vec<Option<(M3, V3)>> = Vec::with_capacity(n);
    for k in 0..n {
        let t = t0 + k as f64 * dt;
        let j = gt.partition_point(|g| g.0 <= t).clamp(1, gt.len() - 1);
        let (g0, g1) = (&gt[j - 1], &gt[j]);
        if g1.0 - g0.0 > 0.05 || t < g0.0 - 1e-9 || t > g1.0 + 1e-9 {
            grid.push(None);
            continue;
        }
        let u = ((t - g0.0) / (g1.0 - g0.0)).clamp(0.0, 1.0);
        let (r0, r1) = (quat_m(g0.2), quat_m(g1.2));
        let d = log(&mm(&mt(&r0), &r1));
        let r = mm(&r0, &exp(d.map(|x| x * u)));
        grid.push(Some((r, std::array::from_fn(|i| g0.1[i] + u * (g1.1[i] - g0.1[i])))));
    }
    // 가우스 고르기(빈칸 넘어서는 안 섞음): 회전은 그 칸 접공간에서, 위치는 그대로
    let gauss = |sig: f64| -> Vec<f64> {
        let m = (3.0 * sig / dt).ceil() as isize;
        (-m..=m).map(|i| (-0.5 * (i as f64 * dt / sig).powi(2)).exp()).collect()
    };
    let (kr, kp) = (gauss(s_rot), gauss(s_pos));
    let smooth = |k: usize| -> Option<(M3, V3)> {
        let (r, _) = grid[k]?;
        let (mut acc_r, mut wr) = ([0.0; 3], 0.0);
        let mr = (kr.len() / 2) as isize;
        for (i, w) in kr.iter().enumerate() {
            let j = k as isize + i as isize - mr;
            if let Some(Some((rj, _))) = (j >= 0).then(|| grid.get(j as usize)).flatten() {
                let l = log(&mm(&mt(&r), rj));
                for c in 0..3 {
                    acc_r[c] += w * l[c];
                }
                wr += w;
            }
        }
        let (mut acc_p, mut wp) = ([0.0; 3], 0.0);
        let mp = (kp.len() / 2) as isize;
        for (i, w) in kp.iter().enumerate() {
            let j = k as isize + i as isize - mp;
            if let Some(Some((_, pj))) = (j >= 0).then(|| grid.get(j as usize)).flatten() {
                for c in 0..3 {
                    acc_p[c] += w * pj[c];
                }
                wp += w;
            }
        }
        Some((mm(&r, &exp(acc_r.map(|x| x / wr))), acc_p.map(|x| x / wp)))
    };
    let sm: Vec<Option<(M3, V3)>> = (0..n).map(smooth).collect();
    // IMU 자세: R_wi = R_wc R_ci, 위치 = p_c + R_wc t_ci
    let imu: Vec<Option<(M3, V3)>> = sm.iter().map(|o| o.map(|(r, p)| (mm(&r, &r_ci), { let d = mv(&r, &t_ci); [p[0] + d[0], p[1] + d[1], p[2] + d[2]] }))).collect();

    // 세계 z 가 위인지 확인용: 카메라 아래(광학 y) 축의 세계 z 평균(위를 보는 일 없는 손 녹화면 음수)
    let ydown: f64 = sm.iter().flatten().map(|(r, _)| r[2][1]).sum::<f64>() / sm.iter().flatten().count() as f64;

    let (gs, gw) = (0.00049 * hz.sqrt() * ns, 0.00002 * dt.sqrt() * ns);
    let (as_, aw) = (0.0069 * hz.sqrt() * ns, 0.0002 * dt.sqrt() * ns);
    let (mut bg, mut ba) = (bg0, ba0);
    let mut out = String::from("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n");
    let mut written = 0usize;
    let mut wmax = 0f64;
    for k in 1..n.saturating_sub(1) {
        let (Some((r0, _)), Some((r1, p1)), Some((_, pm)), Some((_, pp))) = (imu[k - 1], imu[k], imu[k - 1], imu[k + 1]) else { continue };
        let w = log(&mm(&mt(&r0), &r1)).map(|x| x / dt);
        wmax = wmax.max((w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt());
        let aw_: V3 = std::array::from_fn(|i| (pp[i] - 2.0 * p1[i] + pm[i]) / (dt * dt) + if i == 2 { 9.80665 } else { 0.0 });
        let acc = mv(&mt(&r1), &aw_);
        for i in 0..3 {
            bg[i] += gw * rng.n();
            ba[i] += aw * rng.n();
        }
        let gy: V3 = std::array::from_fn(|i| w[i] + bg[i] + gs * rng.n());
        let ac: V3 = std::array::from_fn(|i| acc[i] + ba[i] + as_ * rng.n());
        let t = t0 + k as f64 * dt + shift + offset;
        out += &format!("{:.0},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}\n", t * 1e9, gy[0], gy[1], gy[2], ac[0], ac[1], ac[2]);
        written += 1;
    }
    std::fs::write(&a[2], out).unwrap_or_else(|e| panic!("{}: {e}", a[2]));
    eprintln!(
        "imu_synth: 정답 {} 줄 {:.1} 초 → IMU {written} 표본({hz} Hz, 빈칸 {}), 시각 보정 {shift:+.4} 초, 카메라 아래 축의 세계 z 평균 {ydown:+.2}(음수 = 세계 z 위), 최대 각속도 {:.0} °/s",
        gt.len(),
        t1 - t0,
        n - 2 - written,
        wmax.to_degrees()
    );
}
