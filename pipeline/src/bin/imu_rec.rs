//! IMU 녹화(보드): ICM-42688-P 200 Hz 표본을 Kalibr imu0.csv 꼴로 — `rsd_rec` 과 함께 돌려 카메라 + IMU 녹화를 만든다.
//!   imu_rec [/dev/i2c-1] <출력 csv> [초=60]
//! 줄: 시각(ns, 카메라 장면과 같은 시계), 자이로 xyz(rad/s), 가속도 xyz(m/s²). 재생: `ondevice slam --rsdrec <폴더> --imu-csv <csv>`
fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("imu_rec <장치(/dev/i2c-1)> <출력 csv> [초=60]");
        std::process::exit(1);
    }
    let secs: f64 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(60.0);
    #[cfg(target_os = "linux")]
    {
        use std::io::Write;
        let ring = slam::imu::Ring::new();
        if let Err(e) = slam::imu::spawn_reader(&a[1], ring.clone()) {
            eprintln!("imu_rec: {e}");
            std::process::exit(1);
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(&a[2]).unwrap_or_else(|e| panic!("{}: {e}", a[2])));
        let _ = writeln!(f, "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]");
        let (t0, mut n) = (std::time::Instant::now(), 0u64);
        while t0.elapsed().as_secs_f64() < secs {
            std::thread::sleep(std::time::Duration::from_millis(100));
            ring.drain(|s| {
                n += 1;
                let _ = writeln!(f, "{},{},{},{},{},{},{}", (s.t * 1e9).round() as u64, s.w[0], s.w[1], s.w[2], s.a[0], s.a[1], s.a[2]);
            });
        }
        let _ = f.flush();
        eprintln!("imu_rec: 표본 {n} 개({:.0} 개/초) → {}", n as f64 / secs, a[2]);
    }
    #[cfg(not(target_os = "linux"))]
    eprintln!("imu_rec: 리눅스 I2C 전용({secs} 초)");
}
