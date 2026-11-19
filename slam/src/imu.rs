//! IMU(ICM-42688-P, I2C): 읽기 스레드 · 표본 링 · Kalibr 보정 · 녹화 재생. 융합은 위치 백엔드(ORB-SLAM3 스테레오-관성)가 한다.
//!   - 읽기 스레드(같은 프로세스): 200 Hz 로 레지스터를 읽어 락프리 링버퍼(생산자 하나 · 소비자 하나)에 넣는다
//!   - 소비자(위치 백엔드)는 장면마다 그 장면 시각까지의 표본을 꺼내 넘긴다(`Ring::take_until`)
//! 시각: 장면 시각(rsdriver = 카메라 HW 시계 + 첫 장면에서 고정한 UNIX 시각 오프셋)과 같은 UNIX 시계. t_imu = t_cam + timeshift_cam_imu(Kalibr).
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// 한 표본(IMU 축): 시각(UNIX 초, IMU 시계), 자이로 rad/s, 가속도 m/s²(넷째 칸 0)
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub struct Sample {
    pub w: [f32; 4],
    pub a: [f32; 4],
    pub t: f64,
}

const CAP: u64 = 2048; // 200 Hz 로 10 초
const MASK: u64 = CAP - 1;

/// 생산자 하나(읽기 스레드) · 소비자 하나(위치 백엔드) 링버퍼. 가득 차면 새 표본을 버림(읽지 않은 칸을 덮어쓰지 않음)
pub struct Ring {
    slots: Box<[UnsafeCell<Sample>]>,
    /// 다음에 쓸 번호(생산자만 씀)
    head: AtomicU64,
    /// 아직 쓰는 가장 오래된 번호(소비자만 씀)
    tail: AtomicU64,
    pub dropped: AtomicU64,
    pub read_fail: AtomicU64,
    /// 읽기 단에서 버린 표본 수(Bad 순서: 무효 값 · 포화 · 튐 · 같은 값 반복 · 느린 읽기)
    pub bad: [AtomicU64; 5],
    pub closed: AtomicBool,
}

// SAFETY: 칸 [tail, head) 는 소비자만 읽고 생산자는 그 밖(head 자리)에만 쓴다. head 는 Release 로 내고 Acquire 로 읽는다
unsafe impl Sync for Ring {}
unsafe impl Send for Ring {}

impl Ring {
    pub fn new() -> Arc<Ring> {
        Arc::new(Ring {
            slots: (0..CAP).map(|_| UnsafeCell::new(Sample::default())).collect(),
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            read_fail: AtomicU64::new(0),
            bad: Default::default(),
            closed: AtomicBool::new(false),
        })
    }

    /// 생산자: 한 표본 넣기(가득 찼으면 버리고 false)
    pub fn push(&self, s: Sample) -> bool {
        let h = self.head.load(Ordering::Relaxed);
        if h - self.tail.load(Ordering::Acquire) >= CAP {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        // SAFETY: h 칸은 [tail, head) 밖이라 소비자가 읽지 않음
        unsafe { *self.slots[(h & MASK) as usize].get() = s };
        self.head.store(h + 1, Ordering::Release);
        true
    }

    /// 소비자: [tail, head) 를 두 조각 슬라이스로 빌림(링이 감긴 자리에서 나뉨)
    fn view(&self) -> (u64, &[Sample], &[Sample]) {
        let h = self.head.load(Ordering::Acquire);
        let from = self.tail.load(Ordering::Relaxed).min(h);
        let n = (h - from) as usize;
        let s0 = (from & MASK) as usize;
        let first = n.min(CAP as usize - s0);
        // SAFETY: [from, h) 칸은 생산자가 다 쓴 뒤(Acquire)이고 tail 을 넘기 전까지 덮어쓰지 않음. UnsafeCell<Sample> 와 Sample 은 같은 배치
        let base = self.slots.as_ptr() as *const Sample;
        unsafe { (from, std::slice::from_raw_parts(base.add(s0), first), std::slice::from_raw_parts(base, n - first)) }
    }

    /// 소비자: IMU 시각 t 까지의 표본을 차례로 꺼냄(그 뒤 표본은 다음 장면 몫으로 남김)
    pub fn take_until(&self, t: f64, mut f: impl FnMut(&Sample)) {
        let (from, a, b) = self.view();
        let mut k = 0u64;
        for s in a.iter().chain(b.iter()) {
            if s.t > t {
                break;
            }
            f(s);
            k += 1;
        }
        self.tail.store(from + k, Ordering::Release);
    }

    /// 쌓인 표본을 모두 꺼냄(녹화 도구 `imu_rec`)
    pub fn drain(&self, f: impl FnMut(&Sample)) {
        self.take_until(f64::INFINITY, f);
    }
}

/// IMU → 카메라 보정(Kalibr, 카메라 = 컬러): p_cam = R_ci · p_imu + t_ci, t_imu = t_cam + shift
#[derive(Clone, Debug)]
pub struct Calib {
    /// T_cam_imu 3×4 행 우선
    pub t_cam_imu: [f64; 12],
    pub shift: f64,
}

/// 저장소에 둔 보정 결과(내부 파라미터는 공장값을 쓰고 이 파일에서는 T_cam_imu · timeshift 만 읽음)
pub const DEFAULT_CALIB: &str = include_str!("../../docs/imu_cam_calib.yaml");

impl Calib {
    pub fn parse(text: &str) -> Result<Calib, String> {
        let mut rows: Vec<[f64; 4]> = Vec::new();
        let mut shift = None;
        let mut in_t = false;
        for l in text.lines() {
            let s = l.trim();
            if s.starts_with("T_cam_imu:") {
                in_t = true;
                continue;
            }
            if in_t && s.starts_with('-') && rows.len() < 3 {
                let v: Vec<f64> = s.trim_start_matches('-').trim().trim_matches(|c| c == '[' || c == ']').split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if v.len() != 4 {
                    return Err(format!("T_cam_imu 줄을 못 읽음: {s}"));
                }
                rows.push([v[0], v[1], v[2], v[3]]);
                continue;
            }
            in_t = false;
            if let Some(v) = s.strip_prefix("timeshift_cam_imu:") {
                shift = v.trim().parse::<f64>().ok();
            }
        }
        if rows.len() < 3 {
            return Err("T_cam_imu 가 없음".into());
        }
        let mut t = [0.0; 12];
        for (i, r) in rows.iter().enumerate() {
            t[i * 4..i * 4 + 4].copy_from_slice(r);
        }
        Ok(Calib { t_cam_imu: t, shift: shift.ok_or("timeshift_cam_imu 가 없음")? })
    }
}

/// 읽기 단에서 버리는 표본 종류(Ring::bad 칸 번호)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bad {
    /// 칩의 무효 값(-32768) · 바이트가 모두 0 · 모두 0xFF(버스 오류)
    Invalid = 0,
    /// 한 축이라도 ±2000 °/s · ±16 g 끝(실제 값을 모름)
    Saturated = 1,
    /// 앞 표본과 견줘 5 ms 사이 물리적으로 불가능한 변화(자이로 15 rad/s · 가속도 8 g 넘게)
    Jump = 2,
    /// 여섯 값이 앞 표본과 똑같음(레지스터가 새 값으로 안 바뀜)
    Repeat = 3,
    /// 읽기에 4 ms 넘게 걸림(버스 막힘 · 다시 시도: 시각을 모름)
    Slow = 4,
}

/// 레지스터 12 바이트(가속도 xyz · 자이로 xyz, 빅 엔디언 i16) 검사. prev = 앞에서 받아들인 표본의 원값, jumps = 튐으로 연달아 버린 수(3 번 넘으면 새 기준으로 받아들임)
pub fn check_raw(b: &[u8], prev: Option<&[i16; 6]>, jumps: u32) -> Result<[i16; 6], Bad> {
    let v: [i16; 6] = std::array::from_fn(|i| i16::from_be_bytes([b[2 * i], b[2 * i + 1]]));
    if v.contains(&i16::MIN) || b[..12].iter().all(|&x| x == 0) || b[..12].iter().all(|&x| x == 0xFF) {
        return Err(Bad::Invalid);
    }
    if v.iter().any(|x| x.unsigned_abs() >= 32700) {
        return Err(Bad::Saturated);
    }
    if let Some(p) = prev {
        if *p == v {
            return Err(Bad::Repeat);
        }
        // 가속도 8 g = 16,384 LSB, 자이로 15 rad/s ≈ 14,080 LSB
        let d = |i: usize| (v[i] as i32 - p[i] as i32).abs();
        if jumps < 3 && ((0..3).any(|i| d(i) > 16_384) || (3..6).any(|i| d(i) > 14_080)) {
            return Err(Bad::Jump);
        }
    }
    Ok(v)
}

/// 재생: 카메라 장면 시각보다 이만큼(초) 앞선 IMU 표본까지 링에 있다고 본다(실시간은 장면이 찍힌 뒤 받기 · 처리까지 약 20 ms 걸리는 동안 IMU 는 계속 들어옴)
pub const REPLAY_LEAD: f64 = 0.02;

/// 녹화 IMU 재생(Kalibr imu0.csv: 시각 ns, 자이로 rad/s xyz, 가속도 m/s² xyz[, 그 밖 칸]). 읽기 단 거르기(같은 값 반복 · 시각 역행 · 튐)를
/// 실시간 읽기 스레드와 같은 규칙으로 하고, 남은 표본을 같은 링에 넣는다
pub struct CsvReplay {
    s: Arc<Vec<Sample>>,
    i: usize,
    /// 버린 표본(Bad 순서): 무효 0 · 포화 0 · 튐 · 같은 값 반복 · 시각 역행(느린 읽기 칸)
    pub bad: [u64; 5],
    shift: f64,
}

impl CsvReplay {
    pub fn load(path: &str, shift: f64) -> Result<CsvReplay, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let mut s = Vec::new();
        let mut bad = [0u64; 5];
        let (mut prev, mut jumps) = (None::<[f32; 6]>, 0u32);
        for l in text.lines() {
            let v: Vec<f64> = l.split(',').map_while(|x| x.trim().parse().ok()).collect();
            if v.len() < 7 {
                continue; // 머리 줄 · 빈 줄
            }
            let t = v[0] * 1e-9;
            let x = [v[1] as f32, v[2] as f32, v[3] as f32, v[4] as f32, v[5] as f32, v[6] as f32];
            if s.last().is_some_and(|p: &Sample| t <= p.t) {
                bad[Bad::Slow as usize] += 1;
                continue;
            }
            if let Some(p) = prev {
                if p == x {
                    bad[Bad::Repeat as usize] += 1;
                    continue;
                }
                // 실시간 check_raw 와 같은 한도: 5 ms 사이 자이로 15 rad/s · 가속도 8 g 넘는 변화는 3 번까지 버림
                let big = (0..3).any(|i| (x[i] - p[i]).abs() > 15.0) || (3..6).any(|i| (x[i] - p[i]).abs() > 8.0 * 9.80665);
                if big && jumps < 3 {
                    jumps += 1;
                    bad[Bad::Jump as usize] += 1;
                    continue;
                }
            }
            (prev, jumps) = (Some(x), 0);
            s.push(Sample { w: [x[0], x[1], x[2], 0.0], a: [x[3], x[4], x[5], 0.0], t });
        }
        if s.len() < 40 {
            return Err(format!("{path}: IMU 표본이 모자람({} 개)", s.len()));
        }
        Ok(CsvReplay { s: Arc::new(s), i: 0, bad, shift })
    }

    pub fn len(&self) -> usize {
        self.s.len()
    }

    /// (첫, 마지막) 표본 시각(IMU 시계)
    pub fn span(&self) -> (f64, f64) {
        (self.s[0].t, self.s[self.s.len() - 1].t)
    }

    /// 읽기 단 거른 수를 링 집계에 더함(끝 요약 줄에 실시간과 같은 꼴로 나옴)
    pub fn count_into(&self, ring: &Ring) {
        for (k, &n) in self.bad.iter().enumerate() {
            ring.bad[k].fetch_add(n, Ordering::Relaxed);
        }
    }

    /// IMU 시각 t 까지의 표본을 링에 넣는다
    pub fn feed(&mut self, ring: &Ring, t: f64) {
        while self.i < self.s.len() && self.s[self.i].t <= t {
            ring.push(self.s[self.i]);
            self.i += 1;
        }
    }

    /// 찍은 속도 재생(--rate R): 장면 재생 시계(첫 장면 시각, 그때 벽시계)에 맞춰 표본을 넣는 스레드(실시간 읽기 스레드 자리)
    pub fn spawn_timed(mut self, ring: Arc<Ring>, clock: Arc<std::sync::OnceLock<(f64, std::time::Instant)>>, rate: f64) {
        let _ = std::thread::Builder::new().name("imu".into()).spawn(move || {
            let (t0, w0) = loop {
                if let Some(c) = clock.get() {
                    break *c;
                }
                if ring.closed.load(Ordering::Relaxed) {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            };
            while self.i < self.s.len() && !ring.closed.load(Ordering::Relaxed) {
                // 카메라 시각 τ 의 표본은 벽시계 (τ − t0 − 앞섬) / 배속 에 들어온다
                let now = t0 + REPLAY_LEAD + w0.elapsed().as_secs_f64() * rate;
                self.feed(&ring, now + self.shift);
                std::thread::sleep(std::time::Duration::from_micros(2500));
            }
        });
    }
}

/// ICM-42688-P 레지스터 변환 배율(±16 g, ±2000 °/s)
pub const ACC_S: f32 = (16.0 * 9.80665 / 32768.0) as f32;
pub const GYR_S: f32 = (2000.0 / 32768.0 * std::f64::consts::PI / 180.0) as f32;

#[cfg(target_os = "linux")]
fn unix_now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// 읽기 스레드 시작: 칩을 찾고(0x68·0x69 WHO_AM_I 0x47) 200 Hz 로 설정한 뒤 링에 넣는다. 칩을 못 찾으면 Err
#[cfg(target_os = "linux")]
pub fn spawn_reader(dev: &str, ring: Arc<Ring>) -> Result<std::thread::JoinHandle<()>, String> {
    use std::io::{Read, Write};
    use std::os::fd::AsRawFd;
    unsafe extern "C" {
        fn ioctl(fd: i32, req: std::ffi::c_ulong, ...) -> i32;
    }
    const I2C_SLAVE: std::ffi::c_ulong = 0x0703;
    let mut f = std::fs::OpenOptions::new().read(true).write(true).open(dev).map_err(|e| format!("{dev}: {e} (I2C 켜기: sudo dtparam i2c_arm=on)"))?;
    let fd = f.as_raw_fd();
    let rd = |f: &mut std::fs::File, reg: u8, out: &mut [u8]| f.write(&[reg]).is_ok_and(|n| n == 1) && f.read(out).is_ok_and(|n| n == out.len());
    let mut addr = 0;
    for a in [0x69u16, 0x68] {
        let mut who = [0u8];
        // SAFETY: 열린 I2C 기술자에 주소 고르기
        if unsafe { ioctl(fd, I2C_SLAVE, a as std::ffi::c_ulong) } == 0 && rd(&mut f, 0x75, &mut who) && who[0] == 0x47 {
            addr = a;
            break;
        }
    }
    if addr == 0 {
        return Err(format!("{dev}: ICM-42688-P 를 찾지 못함(0x68·0x69)"));
    }
    let wr = |f: &mut std::fs::File, reg: u8, v: u8| f.write(&[reg, v]).is_ok_and(|n| n == 2);
    wr(&mut f, 0x11, 0x01); // 소프트 리셋
    std::thread::sleep(std::time::Duration::from_millis(5));
    wr(&mut f, 0x4F, 0x07); // 자이로 ±2000 °/s, 200 Hz
    wr(&mut f, 0x50, 0x07); // 가속도 ±16 g, 200 Hz
    wr(&mut f, 0x4E, 0x0F); // 자이로·가속도 켬(저잡음)
    std::thread::sleep(std::time::Duration::from_millis(50));
    common::info!("IMU: {dev} 주소 0x{addr:02x} ICM-42688-P, 200 Hz");
    Ok(std::thread::Builder::new()
        .name("imu".into())
        .spawn(move || {
            // 실시간 우선순위(권한이 없으면 nice 만)
            unsafe extern "C" {
                fn sched_setscheduler(pid: i32, policy: i32, param: *const i32) -> i32;
            }
            let prio = 50i32;
            // SAFETY: 이 스레드 SCHED_FIFO(1)
            if unsafe { sched_setscheduler(0, 1, &prio) } != 0 {
                common::threads::apply(-10, 0);
            }
            // 코어 0(I2C 인터럽트 자리): 처리 루프 코어(마지막)에서 띄워도 따로
            common::threads::pin(1);
            // 시각: 단조 시계 + 시작 때 고정한 UNIX 오프셋(카메라 시각과 같은 방식, NTP 가 시스템 시계를 옮겨도 뒤로 가거나 튀지 않음)
            let (base_unix, base) = (unix_now(), std::time::Instant::now());
            let now_s = || base_unix + base.elapsed().as_secs_f64();
            let period = 0.005;
            let mut next = now_s();
            let mut b = [0u8; 16];
            let (mut prev, mut jumps, mut t_last) = (None::<[i16; 6]>, 0u32, f64::MIN);
            while !ring.closed.load(Ordering::Relaxed) {
                let ta = now_s();
                let ok = f.write(&[0x1Fu8]).is_ok_and(|n| n == 1) && f.read(&mut b[..12]).is_ok_and(|n| n == 12);
                let tb = now_s();
                if ok {
                    let t = 0.5 * (ta + tb);
                    let chk = if tb - ta > 0.004 || t <= t_last { Err(Bad::Slow) } else { check_raw(&b, prev.as_ref(), jumps) };
                    match chk {
                        Ok(v) => {
                            (prev, jumps, t_last) = (Some(v), 0, t);
                            let (a, w) = crate::simd::imu_raw(&b, ACC_S, GYR_S);
                            ring.push(Sample { w, a, t });
                        }
                        Err(k) => {
                            jumps += (k == Bad::Jump) as u32;
                            ring.bad[k as usize].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                } else {
                    ring.read_fail.fetch_add(1, Ordering::Relaxed);
                }
                next += period;
                let now = now_s();
                // 밀렸으면 몰아 읽지 않음(레지스터에는 최신 값 하나뿐: 같은 값을 다른 시각으로 여러 번 쓰게 됨)
                if next < now {
                    next = now;
                }
                if next > now {
                    std::thread::sleep(std::time::Duration::from_secs_f64(next - now));
                }
            }
        })
        .map_err(|e| e.to_string())?)
}

#[cfg(not(target_os = "linux"))]
pub fn spawn_reader(dev: &str, _ring: Arc<Ring>) -> Result<std::thread::JoinHandle<()>, String> {
    Err(format!("{dev}: IMU 읽기는 리눅스 I2C 전용"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(ring: &Ring, t0: f64, n: usize) {
        for i in 0..n {
            assert!(ring.push(Sample { w: [0.0; 4], a: [0.0, 0.0, 9.80665, 0.0], t: t0 + i as f64 * 0.005 }));
        }
    }

    #[test]
    fn calib_parse() {
        let c = Calib::parse(DEFAULT_CALIB).unwrap();
        assert!((c.shift + 0.02584).abs() < 1e-4);
        // 회전이 직교, 이동은 수 cm
        let r = |i: usize, j: usize| c.t_cam_imu[i * 4 + j];
        let d: f64 = (0..3).map(|k| r(0, k) * r(1, k)).sum();
        let n: f64 = (0..3).map(|k| r(0, k) * r(0, k)).sum();
        assert!(d.abs() < 1e-4 && (n - 1.0).abs() < 1e-4);
        assert!((0..3).all(|i| r(i, 3).abs() < 0.1));
    }

    /// 장면 시각까지만 꺼내고 나머지는 남김. 링이 감긴 자리(두 조각)에서도 차례대로
    #[test]
    fn take_until_keeps_later_samples() {
        let ring = Ring::new();
        feed(&ring, 0.0, CAP as usize - 10);
        let mut n = 0;
        ring.take_until((CAP as f64 - 20.0) * 0.005, |_| n += 1);
        assert_eq!(n, CAP as usize - 19);
        feed(&ring, (CAP - 10) as f64 * 0.005, 30); // 감김
        let mut ts = Vec::new();
        ring.take_until(f64::INFINITY, |s| ts.push(s.t));
        assert_eq!(ts.len(), 9 + 30);
        assert!(ts.windows(2).all(|w| w[1] > w[0]));
    }

    /// 가득 차면 새 표본을 버린다(읽지 않은 칸을 덮어쓰지 않음)
    #[test]
    fn ring_full_drops_new() {
        let ring = Ring::new();
        feed(&ring, 0.0, CAP as usize);
        assert!(!ring.push(Sample::default()));
        assert_eq!(ring.dropped.load(Ordering::Relaxed), 1);
        let mut first = None;
        ring.drain(|s| {
            first.get_or_insert(s.t);
        });
        assert_eq!(first, Some(0.0));
    }

    /// 읽기 단 거르기: 무효 · 포화 · 반복 · 튐(3 번 연달아면 새 기준)
    #[test]
    fn raw_filter() {
        let enc = |v: [i16; 6]| -> [u8; 12] { let mut b = [0u8; 12]; for i in 0..6 { b[2 * i..2 * i + 2].copy_from_slice(&v[i].to_be_bytes()); } b };
        let p = [10, -20, 2048, 3, -4, 5];
        assert_eq!(check_raw(&enc(p), None, 0), Ok(p));
        assert_eq!(check_raw(&[0u8; 12], None, 0), Err(Bad::Invalid));
        assert_eq!(check_raw(&[0xFFu8; 12], None, 0), Err(Bad::Invalid));
        assert_eq!(check_raw(&enc([i16::MIN, 0, 2048, 0, 0, 1]), None, 0), Err(Bad::Invalid));
        assert_eq!(check_raw(&enc([0, 0, 2048, 32767, 0, 1]), None, 0), Err(Bad::Saturated));
        assert_eq!(check_raw(&enc(p), Some(&p), 0), Err(Bad::Repeat));
        let j = [10, -20, 2048, 20000, -4, 5];
        assert_eq!(check_raw(&enc(j), Some(&p), 0), Err(Bad::Jump));
        assert_eq!(check_raw(&enc(j), Some(&p), 3), Ok(j));
    }

    /// 녹화 재생: 같은 여섯 값 · 시각 역행은 버리고, 넣을 시각까지만 링에 넣음
    #[test]
    fn csv_replay_filter_and_feed() {
        let p = std::env::temp_dir().join(format!("imu_csv_{}.csv", std::process::id()));
        let mut s = String::from("#timestamp [ns],wx,wy,wz,ax,ay,az\n");
        for i in 0..100u64 {
            let t = 1_000_000_000_000 + i * 5_000_000;
            let v = if (50..53).contains(&i) { 49 } else { i }; // 50 – 52 는 49 와 같은 값(밀려 반복 기록)
            s += &format!("{t},{},0,0,0,0,9.8,{}\n", v as f64 * 0.001, t as f64 * 1e-9);
        }
        s += "1000000100000,1,1,1,1,1,1\n"; // 시각 역행
        std::fs::write(&p, s).unwrap();
        let mut r = CsvReplay::load(p.to_str().unwrap(), 0.0).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(r.len(), 97);
        assert_eq!(r.bad[Bad::Repeat as usize], 3);
        assert_eq!(r.bad[Bad::Slow as usize], 1);
        let ring = Ring::new();
        r.feed(&ring, 1000.301);
        let mut n = 0;
        ring.drain(|_| n += 1);
        // 0 – 60 번 표본(1000.0 – 1000.3), 반복 3 개 뺌
        assert_eq!(n, 61 - 3);
    }
}
