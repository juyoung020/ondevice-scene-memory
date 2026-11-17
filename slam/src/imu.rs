//! IMU(ICM-42688-P, I2C) 융합: 가볍고 빠른 쪽만(전체 최적화 VIO 없음).
//!   - 읽기 스레드(같은 프로세스): 200 Hz 로 레지스터를 읽어 락프리 링버퍼(생산자 하나 · 소비자 하나)에 넣는다. 복사 없이 소비자가 칸을 슬라이스로 빌려 읽음
//!   - 장면 사이 자이로 사전적분 → 몸체 회전(광류 시작점 · 되찾기 추측 · 등속 예측의 회전 몫)
//!   - 가속도 저역 통과 → 중력 방향(첫 장면 지도 수평 맞춤, 기울기 진단)
//!   - 광류·되찾기·ICP 가 모두 실패한 짧은 구간(기본 1 초)은 IMU 회전으로 이어 붙임
//!   - 자이로 치우침: 시작할 때 가만히 있으면 평균, 그 뒤 광류가 잘 된 장면마다 영상 회전과의 차이로 조금씩 고침
//! 시각: 장면 시각(rsdriver = 카메라 HW 시계 + 첫 장면에서 고정한 UNIX 시각 오프셋)과 같은 UNIX 시계. t_imu = t_cam + timeshift_cam_imu(Kalibr).
//! 셈(사전적분 · 회전 보정 · 좌표 변환 · 레지스터 변환)은 simd.rs 의 NEON 판(보드)과 스칼라 판(PC)이 비트까지 같다.
use crate::register::rtf::Tf;
use crate::simd::{gyro_step, m34_mul, m34_t, m34_vec, so3_exp, M34, M34_I};
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

/// 생산자 하나(읽기 스레드) · 소비자 하나(위치 추적 스레드) 링버퍼. 가득 차면 새 표본을 버림(읽지 않은 칸을 덮어쓰지 않음)
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
    /// 장면 사이 200 Hz 위치(소비자 = 처리 루프만 만짐)
    fast: UnsafeCell<FastPose>,
    /// 뷰어가 있을 때만: 표본을 넣을 때마다 처리 루프를 깨움(장면 사이 위치를 그 자리에서 냄)
    pub waker: std::sync::OnceLock<Arc<common::rt::Waker>>,
}

/// 장면 사이 위치를 채우는 기준: 마지막으로 보낸 장면의 위치(지도 좌표 몸체 · 광학 변환) · 지도 좌표 속도 · 그때 IMU 상태
#[derive(Clone)]
pub struct Anchor {
    /// 장면 시각(카메라 시계)
    pub t: f64,
    pub body: Tf,
    pub optical: Tf,
    /// 지도 좌표 속도(m/s)
    pub vel: [f32; 3],
    m: M34,
    mt: M34,
    bias: [f32; 4],
    shift: f64,
}

/// 장면 사이 위치(처리 루프가 가짐): 위치 = 기준 위치 × 기준 이후 자이로 회전, 이동은 등속(0.1 초까지) 뒤 유지.
/// 표본은 링에서 바로 읽고(링의 소비자 = 처리 루프), 회전은 지난번 끝에서 새 표본까지만 이어 적분한다(표본마다 O(1))
#[derive(Default)]
pub struct FastPose {
    anchor: Option<Anchor>,
    /// 기준 뒤 이어 적분한 회전과 그 끝 시각(IMU 시계)
    acc: Option<(M34, f64)>,
}

/// 기준 이후 이만큼(초) 지나면 IMU 만으로 위치를 내지 않음(자이로 치우침이 쌓임)
pub const FAST_MAX: f64 = 1.0;
/// 등속으로 옮기는 최대 시간(초)
const FAST_VEL_MAX: f64 = 0.1;

impl FastPose {
    pub fn set_anchor(&mut self, a: Anchor) {
        self.acc = Some((M34_I, a.t + a.shift));
        self.anchor = Some(a);
    }

    /// 가장 새 표본 시각의 (카메라 시계 시각, 표본 시각(IMU 시계), 카메라 광학 → 지도). 기준이 없거나 낡았거나 새 표본이 없으면 None
    pub fn next(&mut self, s0: &[Sample], s1: &[Sample]) -> Option<(f64, f64, Tf)> {
        let a = self.anchor.as_ref()?;
        let s = *s1.last().or(s0.last())?;
        let t = s.t - a.shift;
        let dt = t - a.t;
        if !(0.0..=FAST_MAX).contains(&dt) {
            return None;
        }
        let (r0, t0) = self.acc?;
        if s.t <= t0 {
            return None; // 새 표본 없음(이미 낸 위치)
        }
        let r = m34_mul(&r0, &integrate(s0, s1, t0, s.t, &a.bias, gyro_step)?);
        self.acc = Some((r, s.t));
        let rb = m34_mul(&m34_mul(&a.m, &r), &a.mt);
        let mut b = a.body.mul(&tf_of(&rb, [0.0; 3]));
        let k = dt.min(FAST_VEL_MAX) as f32;
        b.0[3] += a.vel[0] * k;
        b.0[7] += a.vel[1] * k;
        b.0[11] += a.vel[2] * k;
        Some((t, s.t, b.mul(&a.optical)))
    }
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
            fast: UnsafeCell::new(FastPose::default()),
            waker: std::sync::OnceLock::new(),
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

    /// 소비자: [from, head) 를 두 조각 슬라이스로 빌림(링이 감긴 자리에서 나뉨). from 은 tail 이상
    fn view(&self, from: u64) -> (u64, &[Sample], &[Sample]) {
        let h = self.head.load(Ordering::Acquire);
        let from = from.max(self.tail.load(Ordering::Relaxed)).min(h);
        let n = (h - from) as usize;
        let s0 = (from & MASK) as usize;
        let first = n.min(CAP as usize - s0);
        // SAFETY: [from, h) 칸은 생산자가 다 쓴 뒤(Acquire)이고 tail 을 넘기 전까지 덮어쓰지 않음. UnsafeCell<Sample> 와 Sample 은 같은 배치
        let base = self.slots.as_ptr() as *const Sample;
        unsafe { (h, std::slice::from_raw_parts(base.add(s0), first), std::slice::from_raw_parts(base, n - first)) }
    }

    /// 소비자: from 앞 칸은 다 씀(생산자가 다시 씀)
    /// 장면 사이 위치 기준(소비자 = 처리 루프)
    pub fn set_fast_anchor(&self, a: Anchor) {
        // SAFETY: fast 는 소비자(처리 루프)만 만진다
        unsafe { (*self.fast.get()).set_anchor(a) }
    }

    /// 가장 새 표본까지의 장면 사이 위치(소비자 = 처리 루프). 돌려줌: (카메라 시계 시각, IMU 시각, 카메라 광학 → 지도)
    pub fn fast_next(&self) -> Option<(f64, f64, Tf)> {
        let (_, a, b) = self.view(0);
        // SAFETY: fast 는 소비자(처리 루프)만 만진다
        let r = unsafe { (*self.fast.get()).next(a, b) };
        if r.is_some() {
            FAST_N.fetch_add(1, Ordering::Relaxed);
        }
        r
    }

    fn release(&self, upto: u64) {
        let t = self.tail.load(Ordering::Relaxed);
        if upto > t {
            self.tail.store(upto.min(self.head.load(Ordering::Acquire)), Ordering::Release);
        }
    }
}

/// IMU → 카메라 보정(Kalibr): p_cam = R_ci · p_imu + t, t_imu = t_cam + shift
#[derive(Clone, Debug)]
pub struct Calib {
    pub r_ci: M34,
    pub shift: f64,
}

/// 저장소에 둔 보정 결과(내부 파라미터는 공장값을 쓰고 이 파일에서는 T_cam_imu · timeshift 만 읽음)
pub const DEFAULT_CALIB: &str = include_str!("../../docs/imu_cam_calib.yaml");

impl Calib {
    pub fn parse(text: &str) -> Result<Calib, String> {
        let mut rows: Vec<[f32; 4]> = Vec::new();
        let mut shift = None;
        let mut in_t = false;
        for l in text.lines() {
            let s = l.trim();
            if s.starts_with("T_cam_imu:") {
                in_t = true;
                continue;
            }
            if in_t && s.starts_with('-') && rows.len() < 3 {
                let v: Vec<f32> = s.trim_start_matches('-').trim().trim_matches(|c| c == '[' || c == ']').split(',').filter_map(|x| x.trim().parse().ok()).collect();
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
        let r_ci = [[rows[0][0], rows[0][1], rows[0][2], 0.0], [rows[1][0], rows[1][1], rows[1][2], 0.0], [rows[2][0], rows[2][1], rows[2][2], 0.0]];
        Ok(Calib { r_ci, shift: shift.ok_or("timeshift_cam_imu 가 없음")? })
    }
}

/// 놓친 뒤 다시 시작할 때 마지막 성공 장면부터 자이로 회전으로 자세를 이어 가는 최대 빈틈(초). 치우침 0.01 rad/s 가 남아도 1.7° 안
pub const RESTART_MAX: f64 = 3.0;

/// 적분 구간 안에서 허용하는 표본 간격(초, 200 Hz 의 4 칸)
pub const MAX_GAP: f64 = 0.02;

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

/// [t0, t1](IMU 시각) 자이로 사전적분: 표본 i 의 값이 (t_{i-1}, t_i] 동안 이어졌다고 봄. 두 조각 슬라이스를 그대로 읽음
/// 끝 쪽 표본이 30 ms 넘게 모자라거나 시작 쪽이 비면 None
pub fn integrate(s0: &[Sample], s1: &[Sample], t0: f64, t1: f64, bias: &[f32; 4], step: fn(&M34, &[f32; 4], &[f32; 4], f32) -> M34) -> Option<M34> {
    if !(t1 > t0) {
        return None;
    }
    let i0 = s0.partition_point(|s| s.t <= t0);
    let (a, b) = if i0 < s0.len() { (&s0[i0..], s1) } else { (&s1[s1.partition_point(|s| s.t <= t0)..], &s1[..0]) };
    let first = a.first().or(b.first())?;
    // 시작 앞 표본이 없고 첫 표본이 늦으면(스레드가 막 시작) 모름
    let had_before = i0 > 0 || (i0 == s0.len() && !s1.is_empty() && s1[0].t <= t0);
    if !had_before && first.t - t0 > 0.03 {
        return None;
    }
    let mut r = M34_I;
    let mut tp = t0;
    let mut last = first.w;
    // 구간 안에 표본 간격이 MAX_GAP 넘게 벌어졌으면(읽기 실패 · 버린 표본 · 스레드 밀림) 그 사이 회전을 모름: 적분하지 않음
    let mut ps = t0;
    for s in a.iter().chain(b.iter()) {
        if s.t.min(t1) - ps > MAX_GAP {
            return None;
        }
        ps = s.t;
        let end = s.t.min(t1);
        if end > tp {
            r = step(&r, &s.w, bias, (end - tp) as f32);
            tp = end;
        }
        last = s.w;
        if s.t >= t1 {
            break;
        }
    }
    if tp < t1 {
        if t1 - tp > 0.03 {
            return None;
        }
        r = step(&r, &last, bias, (t1 - tp) as f32);
    }
    Some(r)
}

/// 회전 행렬 → 회전 벡터(log)
pub fn so3_log(m: &M34) -> [f32; 4] {
    let c = ((m[0][0] + m[1][1] + m[2][2] - 1.0) * 0.5).clamp(-1.0, 1.0);
    let th = c.acos();
    let k = if th < 1e-4 { 0.5 } else { 0.5 * th / th.sin() };
    [k * (m[2][1] - m[1][2]), k * (m[0][2] - m[2][0]), k * (m[1][0] - m[0][1]), 0.0]
}

pub fn m34_of_tf(t: &Tf) -> M34 {
    let a = &t.0;
    [[a[0], a[1], a[2], 0.0], [a[4], a[5], a[6], 0.0], [a[8], a[9], a[10], 0.0]]
}

pub fn tf_of(r: &M34, t: [f32; 3]) -> Tf {
    Tf([r[0][0], r[0][1], r[0][2], t[0], r[1][0], r[1][1], r[1][2], t[1], r[2][0], r[2][1], r[2][2], t[2]])
}

/// 단위 벡터 a 를 b 로 돌리는 가장 작은 회전
pub fn rot_between(a: &[f32; 4], b: &[f32; 4]) -> M34 {
    let c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0.0];
    let s = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    let d = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0);
    if s < 1e-7 {
        return if d > 0.0 { M34_I } else { so3_exp([std::f32::consts::PI, 0.0, 0.0, 0.0]) };
    }
    let th = s.atan2(d);
    so3_exp([c[0] / s * th, c[1] / s * th, c[2] / s * th, 0.0])
}

/// 위치 추적 스레드 쪽(소비자) 융합 상태
pub struct Fusion {
    ring: Arc<Ring>,
    calib: Calib,
    /// IMU 축 → 몸체 축(= 광학→몸체 · IMU→광학), 그 전치
    m: M34,
    mt: M34,
    have_m: bool,
    /// 자이로 치우침(IMU 축)
    pub bias: [f32; 4],
    /// 가속도 저역 통과(IMU 축)
    acc_lp: [f32; 4],
    lp_init: bool,
    /// 저역 통과에 넣은 다음 표본 번호
    seen: u64,
    /// 영상이 다 실패했을 때 IMU 회전으로 이어 붙이는 최대 시간(초)
    pub bridge_max: f64,
    /// 시작할 때 가만히 있어 치우침을 평균으로 잡았는지
    pub bias_static: bool,
    pub n_bias_upd: u64,
    /// 표본 간격이 벌어져 회전을 내지 않은 장면 수
    pub n_gap: std::cell::Cell<u64>,
    /// 시작 때 잡은 치우침(되돌리기용) · 되돌린 수
    pub bias0: [f32; 4],
    pub n_bias_reset: u64,
}

impl Fusion {
    pub fn new(ring: Arc<Ring>, calib: Calib, bridge_max: f64) -> Fusion {
        Fusion { ring, calib, m: M34_I, mt: M34_I, have_m: false, bias: [0.0; 4], acc_lp: [0.0; 4], lp_init: false, seen: 0, bridge_max, bias_static: false, n_bias_upd: 0, n_gap: Default::default(), bias0: [0.0; 4], n_bias_reset: 0 }
    }

    /// (받은 표본 수, 링이 가득 차 버린 수, 읽기 실패 수)
    pub fn counts(&self) -> (u64, u64, u64) {
        (self.ring.head.load(Ordering::Relaxed), self.ring.dropped.load(Ordering::Relaxed), self.ring.read_fail.load(Ordering::Relaxed))
    }

    pub fn ring(&self) -> &Arc<Ring> {
        &self.ring
    }

    /// 장면마다 한 번: 새 표본을 저역 통과에 넣고, keep_from(장면 시각) 앞 표본은 놓아 줌
    pub fn poll(&mut self, local: &Tf, now: f64, keep_from: f64) {
        if !self.have_m {
            let l = m34_of_tf(local);
            self.m = m34_mul(&l, &self.calib.r_ci);
            self.mt = m34_t(&self.m);
            self.have_m = true;
        }
        let (h, a, b) = self.ring.view(self.seen);
        if !self.lp_init {
            // 처음: 지금까지 모인 표본(최대 1 초) 평균. 가만히 있으면(자이로 흔들림 작고 가속도 ≈ g) 자이로 평균 = 치우침
            let n = a.len() + b.len();
            if n < 40 {
                return;
            }
            let (mut sa, mut sw, mut sw2) = ([0f64; 3], [0f64; 3], 0f64);
            let tl = a.iter().chain(b.iter()).last().map(|s| s.t).unwrap_or(0.0);
            let mut k = 0usize;
            for s in a.iter().chain(b.iter()).filter(|s| s.t > tl - 1.0) {
                for i in 0..3 {
                    sa[i] += s.a[i] as f64;
                    sw[i] += s.w[i] as f64;
                    sw2 += (s.w[i] as f64).powi(2);
                }
                k += 1;
            }
            let kf = k as f64;
            let mw: Vec<f64> = sw.iter().map(|x| x / kf).collect();
            let var = sw2 / kf - mw.iter().map(|x| x * x).sum::<f64>();
            let g = (sa.iter().map(|x| (x / kf).powi(2)).sum::<f64>()).sqrt();
            self.acc_lp = [(sa[0] / kf) as f32, (sa[1] / kf) as f32, (sa[2] / kf) as f32, 0.0];
            if var < 0.03f64.powi(2) && (g - 9.80665).abs() < 0.5 {
                self.bias = [mw[0] as f32, mw[1] as f32, mw[2] as f32, 0.0];
                self.bias0 = self.bias;
                self.bias_static = true;
            }
            common::info!("IMU: 시작 표본 {k} 개, 중력 {g:.2} m/s², 자이로 치우침 {}", if self.bias_static { format!("{:.4} {:.4} {:.4} rad/s(가만히 있어 평균)", mw[0], mw[1], mw[2]) } else { "0(움직이는 중, 영상으로 고침)".into() });
            self.lp_init = true;
        } else {
            // 저역 통과(200 Hz 에서 시간 상수 약 0.5 초)
            for s in a.iter().chain(b.iter()) {
                for i in 0..3 {
                    self.acc_lp[i] += 0.01 * (s.a[i] - self.acc_lp[i]);
                }
            }
        }
        self.seen = h;
        // 놓아 줄 자리: 위치 추적이 다음에 적분을 시작할 시각(마지막 성공 장면)과 지금 − 이어 붙이기 한도 중 늦은 쪽 − 0.1 초
        let t_rel = keep_from.max(now - self.bridge_max.max(RESTART_MAX) - 0.2) + self.calib.shift - 0.1;
        let (_, a, b) = self.ring.view(0);
        let k = a.partition_point(|s| s.t < t_rel) + if a.last().is_some_and(|s| s.t < t_rel) { b.partition_point(|s| s.t < t_rel) } else { 0 };
        let tail = self.ring.tail.load(Ordering::Relaxed);
        self.ring.release(tail + k as u64);
    }

    /// 장면 시각 t0 → t1 사이 몸체 회전(t0 몸체 축에서 본 t1 몸체). 표본이 모자라면 None
    pub fn rotation(&self, t0: f64, t1: f64) -> Option<M34> {
        if !self.have_m || !self.lp_init {
            return None;
        }
        let (_, a, b) = self.ring.view(0);
        let Some(r) = integrate(a, b, t0 + self.calib.shift, t1 + self.calib.shift, &self.bias, gyro_step) else {
            self.n_gap.set(self.n_gap.get() + 1);
            return None;
        };
        Some(m34_mul(&m34_mul(&self.m, &r), &self.mt))
    }

    /// 장면 사이 위치를 채울 기준(몸체 → 지도, 몸체 → 광학, 지도 좌표 속도). 아직 IMU 를 못 맞췄으면 None
    pub fn anchor(&self, t: f64, body: Tf, optical: Tf, vel: [f32; 3]) -> Option<Anchor> {
        (self.have_m && self.lp_init).then(|| Anchor { t, body, optical, vel, m: self.m, mt: self.mt, bias: self.bias, shift: self.calib.shift })
    }

    /// 몸체 축의 위 방향(단위, 가속도 저역 통과 = 중력 반대)
    pub fn up_body(&self) -> Option<[f32; 4]> {
        if !self.have_m || !self.lp_init {
            return None;
        }
        let u = m34_vec(&self.m, &self.acc_lp);
        let n = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        (n > 5.0).then(|| [u[0] / n, u[1] / n, u[2] / n, 0.0])
    }

    /// 몸체 축의 위 방향을 가만히 있을 때만(최근 50 ms 표본: 자이로 0.2 rad/s 안, 가속도 크기 g ± 0.4 m/s²). 저역 통과와 달리 회전에 늦지 않음
    pub fn up_static(&self) -> Option<[f32; 4]> {
        if !self.have_m || !self.lp_init {
            return None;
        }
        let (_, a, b) = self.ring.view(0);
        let n = a.len() + b.len();
        if n < 10 {
            return None;
        }
        let mut s = [0f32; 4];
        for x in a.iter().chain(b.iter()).skip(n - 10) {
            let w2: f32 = (0..3).map(|i| (x.w[i] - self.bias[i]).powi(2)).sum();
            let g = (0..3).map(|i| x.a[i] * x.a[i]).sum::<f32>().sqrt();
            if w2 > 0.04 || (g - 9.80665).abs() > 0.4 {
                return None;
            }
            for i in 0..3 {
                s[i] += x.a[i];
            }
        }
        let u = m34_vec(&self.m, &s);
        let k = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        Some([u[0] / k, u[1] / k, u[2] / k, 0.0])
    }

    /// 광류가 잘 된 장면: IMU 회전(r_imu, 몸체)과 영상 회전(r_vis, 몸체)의 차이로 치우침을 조금 고침
    pub fn correct_bias(&mut self, r_imu: &M34, r_vis: &M34, dt: f64) {
        if !(dt > 0.005 && dt < 0.2) {
            return;
        }
        let e = so3_log(&m34_mul(&m34_t(r_imu), r_vis));
        let n = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
        if n > 0.05 {
            return; // 3° 넘게 다르면 영상 쪽이 틀렸을 수 있음
        }
        let ei = m34_vec(&self.mt, &e);
        let k = 0.01 / dt as f32; // 영상 회전 잡음(정지 때 장면당 ±0.5 °/s)이 치우침을 흔들지 않게 작게
        for i in 0..3 {
            self.bias[i] = (self.bias[i] - k * ei[i]).clamp(-0.1, 0.1);
        }
        self.n_bias_upd += 1;
    }

    /// 영상과 계속 어긋날 때: 영상으로 고쳐 온 치우침이 틀렸다고 보고 시작 때 값으로 되돌림
    pub fn reset_bias(&mut self) {
        self.bias = self.bias0;
        self.n_bias_reset += 1;
    }

    /// 지도 위(z)와 IMU 위의 사이각(도): pose = 몸체 → 지도
    pub fn tilt_deg(&self, pose: &Tf) -> f32 {
        match self.up_body() {
            Some(u) => {
                let w = m34_vec(&m34_of_tf(pose), &u);
                w[2].clamp(-1.0, 1.0).acos().to_degrees()
            }
            None => -1.0,
        }
    }
}

/// 재생: 카메라 장면 시각보다 이만큼(초) 앞선 IMU 표본까지 링에 있다고 본다(실시간은 장면이 찍힌 뒤 받기 · 처리까지 약 20 ms 걸리는 동안 IMU 는 계속 들어옴)
pub const REPLAY_LEAD: f64 = 0.02;

/// 녹화 IMU 재생(Kalibr imu0.csv: 시각 ns, 자이로 rad/s xyz, 가속도 m/s² xyz[, 그 밖 칸]). 읽기 단 거르기(같은 값 반복 · 시각 역행 · 튐)를
/// 실시간 읽기 스레드와 같은 규칙으로 하고, 남은 표본을 같은 링 · 장면 사이 위치 칸에 넣는다(Fusion 이하 경로는 실시간과 같음)
pub struct CsvReplay {
    s: Arc<Vec<Sample>>,
    i: usize,
    /// 버린 표본(Bad 순서): 무효 0 · 포화 0 · 튐 · 같은 값 반복 · 시각 역행(느린 읽기 칸)
    pub bad: [u64; 5],
    /// 이 구간(카메라 시각)의 표본은 넣지 않음(끊김 시험)
    pub drop: Vec<(f64, f64)>,
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
        Ok(CsvReplay { s: Arc::new(s), i: 0, bad, drop: Vec::new(), shift })
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

    fn dropped(&self, t_imu: f64) -> bool {
        let t = t_imu - self.shift;
        self.drop.iter().any(|&(a, b)| t >= a && t < b)
    }

    /// IMU 시각 t 까지의 표본을 링(과 장면 사이 위치 칸)에 넣는다. t − 2 초보다 오래된 표본은 건너뜀(위치 추적이 보는 범위는 최대 1.3 초 전)
    pub fn feed(&mut self, ring: &Ring, t: f64) {
        let mut pushed = false;
        while self.i < self.s.len() && self.s[self.i].t <= t {
            let s = self.s[self.i];
            self.i += 1;
            if s.t < t - 2.0 || self.dropped(s.t) {
                continue;
            }
            ring.push(s);
            pushed = true;
        }
        if let (true, Some(w)) = (pushed, ring.waker.get()) {
            w.wake();
        }
    }

    /// 장면 시각(카메라 시계) t 의 각속도(카메라 광학 축, rad/s): 끊김 시험의 흐림 길이용. r_ci = IMU → 카메라
    pub fn gyro_at(samples: &[Sample], r_ci: &M34, shift: f64, t: f64) -> Option<[f32; 4]> {
        let ti = t + shift;
        let k = samples.partition_point(|s| s.t < ti);
        let s = samples.get(k).or(samples.get(k.wrapping_sub(1)))?;
        ((s.t - ti).abs() < 0.02).then(|| m34_vec(r_ci, &s.w))
    }

    pub fn samples(&self) -> Arc<Vec<Sample>> {
        self.s.clone()
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

/// 낸 장면 사이 위치 수
pub static FAST_N: AtomicU64 = AtomicU64::new(0);

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
    let mut rd = |f: &mut std::fs::File, reg: u8, out: &mut [u8]| f.write(&[reg]).is_ok_and(|n| n == 1) && f.read(out).is_ok_and(|n| n == out.len());
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
                            if let Some(wk) = ring.waker.get() {
                                wk.wake(); // 뷰어가 있으면 처리 루프가 이 표본까지의 장면 사이 위치를 냄
                            }
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
    use crate::simd::gyro_step_scalar;

    fn feed(ring: &Ring, t0: f64, n: usize, w: [f32; 4]) {
        for i in 0..n {
            assert!(ring.push(Sample { w, a: [0.0, 0.0, 9.80665, 0.0], t: t0 + i as f64 * 0.005 }));
        }
    }

    #[test]
    fn calib_parse() {
        let c = Calib::parse(DEFAULT_CALIB).unwrap();
        assert!((c.shift + 0.02584).abs() < 1e-4);
        // 회전이 직교
        let p = m34_mul(&m34_t(&c.r_ci), &c.r_ci);
        assert!((p[0][0] - 1.0).abs() < 1e-4 && p[0][1].abs() < 1e-4);
    }

    /// 일정한 각속도: 적분 = Exp(ω·Δt), 링이 감긴 자리(두 조각)에서도 같음, NEON 판 = 스칼라 판
    #[test]
    fn integrate_constant_rate() {
        let ring = Ring::new();
        let w = [0.3, -1.2, 2.0, 0.0];
        // 링을 거의 끝까지 채우고 놓아 감긴 자리를 만든다
        feed(&ring, 0.0, CAP as usize - 100, w);
        ring.release(CAP - 150);
        feed(&ring, (CAP - 100) as f64 * 0.005, 300, w);
        let (_, a, b) = ring.view(0);
        assert!(!b.is_empty());
        let (t0, t1) = (a[a.len() - 20].t + 0.0012, a[a.len() - 20].t + 0.2003);
        let r = integrate(a, b, t0, t1, &[0.0; 4], gyro_step).unwrap();
        let rs = integrate(a, b, t0, t1, &[0.0; 4], gyro_step_scalar).unwrap();
        assert_eq!(r.iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>(), rs.iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>());
        let dt = (t1 - t0) as f32;
        let e = so3_exp([w[0] * dt, w[1] * dt, w[2] * dt, 0.0]);
        for i in 0..3 {
            for j in 0..3 {
                assert!((r[i][j] - e[i][j]).abs() < 2e-4, "{r:?} {e:?}");
            }
        }
        // 치우침을 빼면 그만큼 덜 돈다
        let z = integrate(a, b, t0, t1, &w, gyro_step).unwrap();
        assert!((z[0][0] - 1.0).abs() < 1e-6 && z[0][1].abs() < 1e-6);
        // 표본보다 뒤(30 ms 넘게)는 모름
        let tl = b.last().unwrap().t;
        assert!(integrate(a, b, tl - 0.1, tl + 0.05, &[0.0; 4], gyro_step).is_none());
    }

    /// 장면 사이 위치: 기준 위치 × 그 뒤 자이로 회전(이어 적분), 이동은 등속 0.1 초까지, 1 초 넘으면 멈춤
    #[test]
    fn fast_pose_between_frames() {
        let ring = Ring::new();
        let w = [0.0, 0.0, 1.0, 0.0]; // z 축 1 rad/s
        let push = |i: usize| assert!(ring.push(Sample { w, a: [0.0, 0.0, 9.80665, 0.0], t: 100.0 + i as f64 * 0.005 }));
        for i in 0..20 {
            push(i);
        }
        assert!(ring.fast_next().is_none()); // 기준 없음
        ring.set_fast_anchor(Anchor { t: 100.0, body: Tf::IDENTITY, optical: Tf::IDENTITY, vel: [1.0, 0.0, 0.0], m: M34_I, mt: M34_I, bias: [0.0; 4], shift: 0.0 });
        for i in 20..30 {
            push(i);
        }
        assert!(ring.fast_next().is_some()); // 중간에 한 번(이어 적분)
        for i in 30..41 {
            push(i);
        }
        let (t, _, p) = ring.fast_next().unwrap();
        assert!((t - 100.2).abs() < 1e-9);
        let yaw = p.0[4].atan2(p.0[0]);
        assert!((yaw - 0.2).abs() < 2e-3, "{yaw}");
        assert!((p.0[3] - 0.1).abs() < 1e-6 && p.0[7].abs() < 1e-6, "{:?}", p.0);
        for i in 41..241 {
            push(i);
        }
        assert!(ring.fast_next().is_none()); // 기준 뒤 1.2 초
    }

    /// 가득 차면 새 표본을 버린다(읽지 않은 칸을 덮어쓰지 않음)
    #[test]
    fn ring_full_drops_new() {
        let ring = Ring::new();
        feed(&ring, 0.0, CAP as usize, [0.0; 4]);
        assert!(!ring.push(Sample::default()));
        assert_eq!(ring.dropped.load(Ordering::Relaxed), 1);
        let (_, a, b) = ring.view(0);
        assert_eq!(a.len() + b.len(), CAP as usize);
        assert_eq!(a[0].t, 0.0);
    }

    /// 몸체 회전 · 위 방향 · 치우침 고치기: IMU 축 = 몸체 축이 되게 맞춘 보정으로 확인
    #[test]
    fn fusion_body_and_bias() {
        let ring = Ring::new();
        let w = [0.0, 0.0, 0.5, 0.0]; // 위(z) 축으로 0.5 rad/s, 진짜 치우침 0.02
        let wb = [0.0, 0.0, 0.52, 0.0];
        feed(&ring, 0.0, 400, wb);
        // 광학 → 몸체(OPTICAL)의 역을 R_ci 로 두면 IMU 축 = 몸체 축
        let l = m34_of_tf(&crate::register::rtf::OPTICAL);
        let mut f = Fusion::new(ring.clone(), Calib { r_ci: m34_t(&l), shift: 0.0 }, 1.0);
        f.poll(&crate::register::rtf::OPTICAL, 1.0, 0.0);
        let up = f.up_body().unwrap();
        assert!((up[2] - 1.0).abs() < 1e-5);
        // 움직이는 중이 아니라 가만히(흔들림 0)라 처음 평균 = 0.52 가 치우침이 됨 → 0 으로 되돌리고 영상으로 고치기
        f.bias = [0.0; 4];
        for k in 0..1000 {
            let (t0, t1) = (0.5 + k as f64 * 0.0, 0.533);
            let r = f.rotation(t0, t1).unwrap();
            let dt = (t1 - t0) as f32;
            let vis = so3_exp([w[0] * dt, w[1] * dt, w[2] * dt, 0.0]);
            f.correct_bias(&r, &vis, t1 - t0);
        }
        assert!((f.bias[2] - 0.02).abs() < 2e-3, "{:?}", f.bias);
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

    /// 녹화 재생: 같은 여섯 값 · 시각 역행은 버리고, 넣을 시각까지만 링에 넣음(끊김 구간은 뺌)
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
        r.drop = vec![(1000.0975, 1000.1975)];
        let ring = Ring::new();
        r.feed(&ring, 1000.301);
        let (_, a, b) = ring.view(0);
        // 0 – 60 번 표본(1000.0 – 1000.3), 반복 3 개 · 끊김 구간 20 개 뺌
        assert_eq!(a.len() + b.len(), 61 - 3 - 20);
        assert!(a.iter().all(|s| !(1000.0975..1000.1975).contains(&s.t)));
    }

    /// 구간 안 표본 간격이 벌어지면 적분하지 않음
    #[test]
    fn integrate_gap() {
        let ring = Ring::new();
        feed(&ring, 0.0, 40, [0.0, 0.0, 1.0, 0.0]);
        feed(&ring, 0.3, 40, [0.0, 0.0, 1.0, 0.0]);
        let (_, a, b) = ring.view(0);
        assert!(integrate(a, b, 0.05, 0.15, &[0.0; 4], gyro_step).is_some());
        assert!(integrate(a, b, 0.1, 0.4, &[0.0; 4], gyro_step).is_none());
    }

    #[test]
    fn level_rotation() {
        let up = [0.3f32, -0.2, 0.932, 0.0];
        let n = (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt();
        let up = [up[0] / n, up[1] / n, up[2] / n, 0.0];
        let r = rot_between(&up, &[0.0, 0.0, 1.0, 0.0]);
        let v = m34_vec(&r, &up);
        assert!((v[2] - 1.0).abs() < 1e-5);
    }
}
