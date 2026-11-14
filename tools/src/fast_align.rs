//! 깊이 → 컬러 정렬(depth-to-color registration). librealsense rs2::align 과 같은 결과를 더 적은 CPU 로 낸다.
//!
//! 입력: 깊이 u16 영상(깊이 내부 보정값), 컬러 내부 보정값, 깊이→컬러 외부 변환, 깊이 단위(m/값).
//! 출력: 컬러 화면 크기의 u16 깊이. 값은 원래 깊이 값(깊이 카메라 기준 z, 단위 그대로)이고, 여러 깊이 화소가
//!       한 컬러 화소에 겹치면 가장 가까운(작은) 값, 아무것도 안 오면 0.
//!
//! 방식([`Method`]):
//! - `Sdk`     — rs2::align 의 SIMD 경로(x86 SSE, ARM NEON: v2.59.1 src/proc/sse/sse-align.cpp,
//!   src/proc/neon/neon-align.cpp)와 같은 규칙. Windows SDK 와 Pi 소스 빌드(BUILD_WITH_NEON 기본 켬)가 쓰는 경로.
//!   깊이 화소의 왼쪽 위 모서리(x-0.5, y-0.5)를 컬러로 옮겨 반올림한 화소에 쓴다. 깊이의 각도당 화소가 컬러보다
//!   적을 때(또는 640×480→640×360 같은 특수 해상도)만 오른쪽 아래 모서리까지 옮겨 그 사각형을 채운다.
//!   컬러 왜곡은 MODIFIED_BROWN_CONRADY 일 때만 적용한다(SDK 와 같음, `Options::color_distortion` 으로 D455 의 IBC 도).
//! - `Generic` — rs2::align 의 일반(비 SIMD) 경로(src/proc/align.cpp align_images + rsutil). 두 모서리를 늘 옮겨
//!   사각형을 채우고, 사각형이 화면을 벗어나면 그 화소는 통째로 버리며, 컬러 왜곡을 rs2_project_point_to_pixel 대로 적용.
//!
//! 수치: 각 방식은 같은 순서의 f32 연산을 쓰므로 그 플랫폼의 SDK 와 비트 단위로 같은 결과를 낸다
//! (x86 은 SSE 순서 곱·더하기, aarch64 는 NEON 의 fma 순서). Rust 는 곱·더하기를 스스로 합치지 않으므로
//! fma 는 `mul_add` 로 쓴 곳만이다. tests/ 의 이식본(`rsref`)과 0 화소 차이로 확인한다.
//!
//! 빠르게 하는 방법:
//!   1) 모서리 역투영 방향 표: 깊이 왜곡이 없으면 열마다 x, 줄마다 y 만 둔다(SDK 는 화소마다 f32 2 개 × 모서리 2 개).
//!   2) 한 줄씩 계산 → 바로 흩뿌리기: 줄 하나 분량의 정수 좌표만 L1 에 두고 쓴다(SDK 는 영상 전체 int2 배열을 쓰고 다시 읽음).
//!   3) 계산 루프는 분기 없는 f32 식이라 자동 벡터화된다(x86 SSE, AVX2 가 있으면 8 칸, aarch64 NEON).
//!   4) 여러 스레드: 깊이 줄 덩어리를 짝수·홀수 두 번에 나눠 돌리고, 동시에 돈 덩어리가 쓴 컬러 줄 범위가 겹치면 한 스레드로 다시 한다.


/// rs2_distortion 과 같은 번호
pub mod dist {
    pub const NONE: i32 = 0;
    pub const MODIFIED_BROWN_CONRADY: i32 = 1;
    pub const INVERSE_BROWN_CONRADY: i32 = 2;
    pub const FTHETA: i32 = 3;
    pub const BROWN_CONRADY: i32 = 4;
    pub const KANNALA_BRANDT4: i32 = 5;
}

/// rs2_intrinsics 와 같은 배치(librealsense C API 에 그대로 넘김)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Intrinsics {
    pub width: i32,
    pub height: i32,
    pub ppx: f32,
    pub ppy: f32,
    pub fx: f32,
    pub fy: f32,
    /// [`dist`] 번호
    pub model: i32,
    pub coeffs: [f32; 5],
}

/// rs2_extrinsics 와 같은 배치: rotation 은 열 우선 3×3, translation 은 m
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Extrinsics {
    pub rotation: [f32; 9],
    pub translation: [f32; 3],
}

impl Default for Extrinsics {
    fn default() -> Self {
        Extrinsics { rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], translation: [0.0; 3] }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Method {
    #[default]
    Sdk,
    Generic,
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub method: Method,
    /// 줄 단위 나눔. 1 = 부른 스레드만
    pub threads: i32,
    /// Sdk 방식에서 늘 두 모서리 사각형을 채움(구멍이 줄어듦, SDK 와 결과가 달라짐)
    pub force_rect: bool,
    /// Sdk 방식에서 컬러 Inverse Brown-Conrady 왜곡도 rs2_project_point_to_pixel 식으로 적용(D455 컬러는 계수가 있음).
    /// SDK SIMD 경로는 이 왜곡을 무시하므로 켜면 SDK 와 결과가 달라지고, 정렬된 깊이가 실제 컬러 화소 위치에 맞는다.
    pub color_distortion: bool,
    /// true: SDK 와 같은 순서의 f32 연산(비트 단위 같은 결과). false: Sdk 방식에서 나눗셈 하나(1/Z)로 계산해
    /// 더 빠르지만, 반올림 경계에 걸린 드문 화소가 옆 화소로 옮겨진다(tests 에서 비율을 잰다).
    pub exact: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { method: Method::Sdk, threads: 1, force_rect: false, color_distortion: false, exact: true }
    }
}

/// rs2_fov 와 같은 계산(도)
pub fn fov_deg(i: &Intrinsics) -> [f32; 2] {
    [
        (f32::atan2(i.ppx + 0.5, i.fx) + f32::atan2(i.width as f32 - (i.ppx + 0.5), i.fx)) * 57.2957795,
        (f32::atan2(i.ppy + 0.5, i.fy) + f32::atan2(i.height as f32 - (i.ppy + 0.5), i.fy)) * 57.2957795,
    ]
}

/// aarch64 는 SDK NEON 경로와 같은 fma 순서, 그 밖은 SSE 경로와 같은 곱·더하기 순서
const NEON: bool = cfg!(target_arch = "aarch64");

fn is_special_resolution(d: &Intrinsics, to: &Intrinsics) -> bool {
    (d.width == 640 && d.height == 240 && to.width == 320 && to.height == 180)
        || (d.width == 640 && d.height == 480 && to.width == 640 && to.height == 360)
}

fn distortion_zero(i: &Intrinsics) -> bool {
    i.coeffs.iter().all(|c| c.abs() < f32::EPSILON)
}

/// 정수 변환: SDK 는 (u + 0.5) 를 0 쪽으로 자른다(cvttps / vcvtq). 범위 밖 값은 화면 밖이 되게 ±1e9 로 묶는다(NaN 은 -1e9).
#[inline(always)]
fn to_int(v: f32) -> i32 {
    let c = v.max(-1.0e9).min(1.0e9);
    // SAFETY: c 는 유한하고 i32 범위 안
    unsafe { c.to_int_unchecked::<i32>() }
}

// ---------------- 모서리 역투영 표 ----------------

/// Sdk: pre_compute_x_y_map(sse-align.cpp / neon-align.cpp)
fn sdk_corner(d: &Intrinsics, w: i32, h: i32, offset: f32) -> (f32, f32) {
    let px = w as f32 + offset;
    let py = h as f32 + offset;
    let mut x = (px - d.ppx) / d.fx;
    let mut y = (py - d.ppy) / d.fy;
    if d.model == dist::INVERSE_BROWN_CONRADY {
        let c = &d.coeffs;
        let r2 = x * x + y * y;
        let f = if NEON {
            1.0 + (c[0] + (c[1] + c[4] * r2) * r2) * r2
        } else {
            1.0 + c[0] * r2 + c[1] * r2 * r2 + c[4] * r2 * r2 * r2
        };
        let ux = x * f + 2.0 * c[2] * x * y + c[3] * (r2 + 2.0 * x * x);
        let uy = y * f + 2.0 * c[3] * x * y + c[2] * (r2 + 2.0 * y * y);
        x = ux;
        y = uy;
    }
    (x, y)
}

/// Generic: rs2_deproject_pixel_to_point(src/rs.cpp)의 깊이 1 일 때 x, y
fn generic_corner(i: &Intrinsics, w: i32, h: i32, offset: f32) -> Result<(f32, f32), String> {
    let px = w as f32 + offset;
    let py = h as f32 + offset;
    let mut x = (px - i.ppx) / i.fx;
    let mut y = (py - i.ppy) / i.fy;
    let (xo, yo) = (x, y);
    let c = &i.coeffs;
    if !distortion_zero(i) {
        if i.model == dist::INVERSE_BROWN_CONRADY {
            for _ in 0..10 {
                let r2 = x * x + y * y;
                let icdist = 1.0f32 / (1.0 + ((c[4] * r2 + c[1]) * r2 + c[0]) * r2);
                let xq = x / icdist;
                let yq = y / icdist;
                let delta_x = 2.0 * c[2] * xq * yq + c[3] * (r2 + 2.0 * xq * xq);
                let delta_y = 2.0 * c[3] * xq * yq + c[2] * (r2 + 2.0 * yq * yq);
                x = (xo - delta_x) * icdist;
                y = (yo - delta_y) * icdist;
            }
        }
        if i.model == dist::BROWN_CONRADY {
            for _ in 0..10 {
                let r2 = x * x + y * y;
                let icdist = 1.0f32 / (1.0 + ((c[4] * r2 + c[1]) * r2 + c[0]) * r2);
                let delta_x = 2.0 * c[2] * x * y + c[3] * (r2 + 2.0 * x * x);
                let delta_y = 2.0 * c[3] * x * y + c[2] * (r2 + 2.0 * y * y);
                x = (xo - delta_x) * icdist;
                y = (yo - delta_y) * icdist;
            }
        }
    }
    if i.model == dist::FTHETA || i.model == dist::KANNALA_BRANDT4 {
        return Err("fast_align: 깊이 왜곡 모델 FTheta/KannalaBrandt4 는 지원하지 않음".into());
    }
    Ok((x, y))
}

// ---------------- 본체 ----------------

#[derive(Default)]
struct Scratch {
    u0: Vec<i32>,
    v0: Vec<i32>,
    u1: Vec<i32>,
    v1: Vec<i32>,
    zf: Vec<f32>,
    vmin: i32,
    vmax: i32,
}

/// 정렬 계산에 쓰는 고정 값(보정·표). 스레드끼리 나눠 읽는다.
struct Core {
    d: Intrinsics,
    c: Intrinsics,
    e: Extrinsics,
    unit: f32,
    method: Method,
    exact: bool,
    rect: bool,
    /// 표가 열·줄로 나뉨(깊이 왜곡 없음)
    sep: bool,
    color_dist: bool,
    /// 표[모서리]: sep 이면 x 는 열 수, y 는 줄 수 길이, 아니면 화소 수 길이
    xt: [Vec<f32>; 2],
    yt: [Vec<f32>; 2],
    /// x86 에서 AVX2 가 있으면 8 칸 벡터 루프(같은 연산, 같은 결과)
    #[allow(dead_code)]
    avx2: bool,
}

/// 여러 스레드가 같은 출력 영상에 쓰기 위한 포인터(겹침은 process 가 확인하고 한 스레드로 다시 함)
#[derive(Clone, Copy)]
struct OutPtr(*mut u16);
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

// ---------------- 스레드 풀 ----------------
// 부른 스레드 + 일꾼 T-1 개가 0..count 를 나눠 돈다. 일꾼은 조건 변수로 깨운다(작업 하나가 0.1 ms 대라 깨우는 비용이 작아야 함).

type Job = *const (dyn Fn(usize) + Sync);

struct PoolState {
    job: Option<Job>,
    count: usize,
    generation: u64,
    /// job 을 받아 도는 중인 일꾼 수(run 은 0 이 될 때까지 기다린 뒤 job 을 지움)
    active: usize,
    stop: bool,
}
// SAFETY: job 포인터는 run 이 끝나기 전(active == 0 확인 전)에만 쓴다
unsafe impl Send for PoolState {}

struct PoolShared {
    m: std::sync::Mutex<PoolState>,
    cv: std::sync::Condvar,
    done_cv: std::sync::Condvar,
    next: std::sync::atomic::AtomicUsize,
}

struct Pool {
    sh: std::sync::Arc<PoolShared>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    fn new(n: usize) -> Pool {
        let sh = std::sync::Arc::new(PoolShared {
            m: std::sync::Mutex::new(PoolState { job: None, count: 0, generation: 0, active: 0, stop: false }),
            cv: std::sync::Condvar::new(),
            done_cv: std::sync::Condvar::new(),
            next: std::sync::atomic::AtomicUsize::new(0),
        });
        let workers = (0..n)
            .map(|_| {
                let sh = sh.clone();
                std::thread::spawn(move || {
                    let mut seen = 0u64;
                    loop {
                        let (job, count) = {
                            let mut st = sh.m.lock().unwrap();
                            while !st.stop && st.generation == seen {
                                st = sh.cv.wait(st).unwrap();
                            }
                            if st.stop {
                                return;
                            }
                            seen = st.generation;
                            let Some(job) = st.job else { continue };
                            st.active += 1;
                            (job, st.count)
                        };
                        // SAFETY: active 를 올린 동안 run 은 돌아가지 않으므로 job 이 살아 있음
                        Pool::work(&sh, unsafe { &*job }, count);
                        let mut st = sh.m.lock().unwrap();
                        st.active -= 1;
                        if st.active == 0 {
                            sh.done_cv.notify_all();
                        }
                    }
                })
            })
            .collect();
        Pool { sh, workers }
    }

    fn work(sh: &PoolShared, f: &(dyn Fn(usize) + Sync), count: usize) {
        loop {
            let i = sh.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if i >= count {
                return;
            }
            f(i);
        }
    }

    /// f(0..count) 를 일꾼 + 부른 스레드가 나눠 돌리고 다 끝날 때까지 기다림
    fn run(&self, count: usize, f: &(dyn Fn(usize) + Sync)) {
        // SAFETY: 수명만 지움. 아래에서 일꾼이 모두 놓을 때까지 기다린 뒤 job 을 지운다
        let job: Job = unsafe { std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), Job>(f as *const _) };
        {
            let mut st = self.sh.m.lock().unwrap();
            st.job = Some(job);
            st.count = count;
            self.sh.next.store(0, std::sync::atomic::Ordering::Relaxed);
            st.generation += 1;
        }
        self.sh.cv.notify_all();
        Pool::work(&self.sh, f, count);
        let mut st = self.sh.m.lock().unwrap();
        while st.active > 0 {
            st = self.sh.done_cv.wait(st).unwrap();
        }
        st.job = None;
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.sh.m.lock().unwrap().stop = true;
        self.sh.cv.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// 보정값이 같은 동안 다시 쓰는 정렬기. 생성 때 모서리 방향 표를 만들고 스레드를 띄운다.
pub struct DepthAligner {
    core: Core,
    threads: usize,
    fell_back: bool,
    /// 덩어리마다(여러 스레드), 또는 하나
    scratch: Vec<std::sync::Mutex<Scratch>>,
    pool: Option<Pool>,
}

impl DepthAligner {
    pub fn new(depth: &Intrinsics, color: &Intrinsics, d2c: &Extrinsics, depth_unit: f32, opt: &Options) -> Result<DepthAligner, String> {
        if depth.width <= 0 || depth.height <= 0 || color.width <= 0 || color.height <= 0 {
            return Err("fast_align: 영상 크기가 0".into());
        }
        let (rect, color_dist) = if opt.method == Method::Sdk {
            let color_dist = color.model == dist::MODIFIED_BROWN_CONRADY || (opt.color_distortion && color.model == dist::INVERSE_BROWN_CONRADY);
            let fov = fov_deg(depth);
            let (dpx, dpy) = (depth.width as f32 / fov[0], depth.height as f32 / fov[1]);
            let fov = fov_deg(color);
            let (cpx, cpy) = (color.width as f32 / fov[0], color.height as f32 / fov[1]);
            (opt.force_rect || dpx < cpx || dpy < cpy || is_special_resolution(depth, color), color_dist)
        } else {
            if color.model == dist::FTHETA || color.model == dist::KANNALA_BRANDT4 {
                return Err("fast_align: 컬러 왜곡 모델 FTheta/KannalaBrandt4 는 지원하지 않음".into());
            }
            (true, false)
        };
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        let mut core = Core {
            d: *depth,
            c: *color,
            e: *d2c,
            unit: depth_unit,
            method: opt.method,
            exact: opt.exact,
            rect,
            sep: true,
            color_dist,
            xt: [vec![], vec![]],
            yt: [vec![], vec![]],
            avx2,
        };
        core.build_tables()?;
        let t = opt.threads.max(1) as usize;
        let w = depth.width as usize;
        let scratch = (0..if t == 1 { 1 } else { 2 * t })
            .map(|_| std::sync::Mutex::new(Scratch { u0: vec![0; w], v0: vec![0; w], u1: vec![0; w], v1: vec![0; w], zf: vec![0.0; w], vmin: 0, vmax: 0 }))
            .collect();
        let pool = if t > 1 { Some(Pool::new(t - 1)) } else { None };
        Ok(DepthAligner { core, threads: t, fell_back: false, scratch, pool })
    }

    /// Sdk 방식에서 사각형(두 모서리)을 채우는지. Generic 은 늘 true.
    pub fn uses_rect(&self) -> bool {
        self.core.rect
    }
    /// 마지막 process 가 겹침 때문에 한 스레드로 다시 돌았는지(여러 스레드일 때)
    pub fn last_fell_back_serial(&self) -> bool {
        self.fell_back
    }

    /// depth: depth.width × depth.height, out: color.width × color.height(먼저 0 으로 채움)
    pub fn process(&mut self, depth: &[u16], out: &mut [u16]) {
        let h = self.core.d.height as usize;
        assert!(depth.len() >= h * self.core.d.width as usize, "fast_align: 깊이 버퍼가 작음");
        assert!(out.len() >= (self.core.c.width * self.core.c.height) as usize, "fast_align: 출력 버퍼가 작음");
        let n_out = (self.core.c.width * self.core.c.height) as usize;
        self.fell_back = false;
        let t = self.threads;
        let op = OutPtr(out.as_mut_ptr());
        if t == 1 || h < 4 * t {
            out[..n_out].fill(0);
            self.core.rows(depth, op, 0, h, self.scratch[0].get_mut().unwrap());
            return;
        }
        let nch = 2 * t;
        let chunk_y = |i: usize| h * i / nch;
        let core = &self.core;
        let scratch = &self.scratch;
        let pool = self.pool.as_ref().unwrap();
        // 0 채우기도 일꾼이 나눠서(앞 판의 차례 채우기 대신, 값은 같음)
        pool.run(t, &|j: usize| {
            let op = op;
            let (a, b) = (n_out * j / t, n_out * (j + 1) / t);
            // SAFETY: [a, b) 는 j 마다 겹치지 않는 출력 칸
            unsafe { std::slice::from_raw_parts_mut(op.0.add(a), b - a).fill(0) };
        });
        for pass in 0..2 {
            pool.run(t, &|j: usize| {
                let op = op;
                let i = 2 * j + pass;
                core.rows(depth, op, chunk_y(i), chunk_y(i + 1), &mut scratch[i].lock().unwrap());
            });
        }
        let scratch: Vec<(i32, i32)> = self.scratch.iter_mut().map(|s| {
            let s = s.get_mut().unwrap();
            (s.vmin, s.vmax)
        }).collect();
        // 같은 번에 동시에 돈 덩어리끼리 컬러 줄 범위가 겹쳤으면 경쟁이 있었을 수 있음 → 한 스레드로 다시
        'outer: for pass in 0..2 {
            let mut a = pass;
            while a < nch {
                let mut b = a + 2;
                while b < nch {
                    let ((amin, amax), (bmin, bmax)) = (scratch[a], scratch[b]);
                    if !(amax < amin || bmax < bmin) && amin <= bmax && bmin <= amax {
                        self.fell_back = true;
                        break 'outer;
                    }
                    b += 2;
                }
                a += 2;
            }
        }
        if self.fell_back {
            // 쓴 컬러 줄만 다시 0 으로(나머지 줄은 아직 0)
            let cw = self.core.c.width as usize;
            let (lo, hi) = scratch.iter().filter(|s| s.1 >= s.0).fold((usize::MAX, 0usize), |(lo, hi), s| (lo.min(s.0 as usize), hi.max(s.1 as usize + 1)));
            if lo < hi {
                out[lo * cw..hi * cw].fill(0);
            }
            self.core.rows(depth, op, 0, h, self.scratch[0].get_mut().unwrap());
        }
    }
}

/// 한 번만 쓸 때
pub fn align_depth_to_color(depth: &[u16], out: &mut [u16], di: &Intrinsics, ci: &Intrinsics, e: &Extrinsics, unit: f32, opt: &Options) -> Result<(), String> {
    let mut a = DepthAligner::new(di, ci, e, unit, opt)?;
    a.process(depth, out);
    Ok(())
}

impl Core {
    fn build_tables(&mut self) -> Result<(), String> {
        let (w, h) = (self.d.width as usize, self.d.height as usize);
        let mut fx: [Vec<f32>; 2] = [vec![0.0; w * h], vec![0.0; w * h]];
        let mut fy: [Vec<f32>; 2] = [vec![0.0; w * h], vec![0.0; w * h]];
        for k in 0..2 {
            let off = if k == 0 { -0.5 } else { 0.5 };
            for yy in 0..h {
                for xx in 0..w {
                    let (ox, oy) = if self.method == Method::Sdk {
                        sdk_corner(&self.d, xx as i32, yy as i32, off)
                    } else {
                        generic_corner(&self.d, xx as i32, yy as i32, off)?
                    };
                    fx[k][yy * w + xx] = ox;
                    fy[k][yy * w + xx] = oy;
                }
            }
        }
        // 나눌 수 있는지: x 가 줄마다 같고 y 가 열마다 같은지(비트 단위)
        let mut sep = true;
        'k: for k in 0..2 {
            for yy in 0..h {
                for xx in 0..w {
                    if fx[k][yy * w + xx].to_bits() != fx[k][xx].to_bits() || fy[k][yy * w + xx].to_bits() != fy[k][yy * w].to_bits() {
                        sep = false;
                        break 'k;
                    }
                }
            }
        }
        self.sep = sep;
        for k in 0..2 {
            if sep {
                self.xt[k] = fx[k][..w].to_vec();
                self.yt[k] = (0..h).map(|yy| fy[k][yy * w]).collect();
            } else {
                self.xt[k] = std::mem::take(&mut fx[k]);
                self.yt[k] = std::mem::take(&mut fy[k]);
            }
        }
        Ok(())
    }

    /// 한 줄, 한 모서리: 깊이 화소 → 컬러 정수 좌표.
    /// IDX: 화면 안이고 깊이가 있으면 컬러 화소 번호(v*W+u), 아니면 -1 을 uo 에(왼쪽 위 모서리만 쓸 때).
    ///      vo[0], vo[1] = 이 줄에서 쓴 번호의 최소(부호 없이)·최대.
    /// FAST: 나눗셈 하나(1/Z)와 줄마다 미리 곱한 회전 방향으로(SDK 와 반올림이 드물게 달라짐)
    #[inline(always)]
    fn corner_row_sdk<const SEP: bool, const DIST: bool, const IDX: bool, const FAST: bool>(&self, z: &[f32], y: usize, k: usize, uo: &mut [i32], vo: &mut [i32]) {
        let w = self.d.width as usize;
        let (cw, ch) = (self.c.width, self.c.height);
        let z = &z[..w];
        let uo = &mut uo[..w];
        let xs: &[f32] = if SEP { &self.xt[k][..w] } else { &self.xt[k][y * w..y * w + w] };
        let ys: &[f32] = if SEP { &self.xt[k][..w] } else { &self.yt[k][y * w..y * w + w] };
        let yrow = if SEP { self.yt[k][y] } else { 0.0 };
        let r = &self.e.rotation;
        let (r0, r1, r2, r3, r4, r5, r6, r7, r8) = (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]);
        let (t0, t1, t2) = (self.e.translation[0], self.e.translation[1], self.e.translation[2]);
        let (fx, fy, ppx, ppy) = (self.c.fx, self.c.fy, self.c.ppx, self.c.ppy);
        let cc = &self.c.coeffs;
        let (c0, c1, c2, c3, c4) = (cc[0], cc[1], cc[2], cc[3], cc[4]);
        let scale = self.unit;
        // FAST 용: 회전한 방향 a = R (x, y, 1) 의 줄 상수 부분
        let (kx, ky, kz) = (r3 * yrow + r6, r4 * yrow + r7, r5 * yrow + r8);
        let mut rmin = u32::MAX; // -1 은 부호 없이 가장 커서 최소에서 빠짐
        let mut rmax = -1i32;
        for x in 0..w {
            let depth = z[x] * scale; // z = 깊이 값(f32 로 정확히 표현됨)
            let (mut xx, mut yy);
            if FAST {
                let yv = if SEP { 0.0 } else { ys[x] };
                let ax = if SEP { r0 * xs[x] + kx } else { r0 * xs[x] + (r3 * yv + r6) };
                let ay = if SEP { r1 * xs[x] + ky } else { r1 * xs[x] + (r4 * yv + r7) };
                let az = if SEP { r2 * xs[x] + kz } else { r2 * xs[x] + (r5 * yv + r8) };
                let wv = 1.0 / (depth * az + t2);
                xx = (depth * ax + t0) * wv;
                yy = (depth * ay + t1) * wv;
            } else {
                let px = depth * xs[x];
                let py = depth * if SEP { yrow } else { ys[x] };
                let zz;
                if NEON {
                    xx = r0.mul_add(px, r3.mul_add(py, r6.mul_add(depth, t0)));
                    yy = r1.mul_add(px, r4.mul_add(py, r7.mul_add(depth, t1)));
                    zz = r2.mul_add(px, r5.mul_add(py, r8.mul_add(depth, t2)));
                } else {
                    xx = r0 * px + (r3 * py + (r6 * depth + t0));
                    yy = r1 * px + (r4 * py + (r7 * depth + t1));
                    zz = r2 * px + (r5 * py + (r8 * depth + t2));
                }
                xx /= zz;
                yy /= zz;
            }
            if DIST {
                // MODIFIED_BROWN_CONRADY
                if NEON {
                    let rr = xx * xx + yy * yy;
                    let f = rr.mul_add(rr.mul_add(rr.mul_add(c4, c1), c0), 1.0);
                    let (xf, yf) = (xx * f, yy * f);
                    xx = c3.mul_add(rr, xf * 2.0f32.mul_add(c3.mul_add(xf, c2 * yf), 1.0));
                    yy = c2.mul_add(rr, yf * 2.0f32.mul_add(c2.mul_add(yf, c3 * xf), 1.0));
                } else {
                    let rr = xx * xx + yy * yy;
                    let r3_ = c1 * (rr * rr) + c4 * (rr * (rr * rr));
                    let f = 1.0 + (c0 * rr + r3_);
                    let (xf, yf) = (xx * f, yy * f);
                    let r4_ = c3 * (rr + 2.0 * (xf * xf));
                    xx = xf + (2.0 * (c2 * (xf * yf)) + r4_);
                    let r5_ = c2 * (rr + 2.0 * (yf * yf));
                    yy = yf + (2.0 * (c3 * (xf * yf)) + r5_);
                }
            }
            let (u, v) = if NEON { (xx.mul_add(fx, ppx), yy.mul_add(fy, ppy)) } else { (xx * fx + ppx, yy * fy + ppy) };
            let (ui, vi) = (to_int(u + 0.5), to_int(v + 0.5));
            if IDX {
                // 분기·비교 없이(부호 비트): 화면 밖이거나 깊이 0 이면 -1, 아니면 vi*CW+ui. 묶은 좌표는 ±1e9 라 넘침 없음
                let zi = z[x] as i32;
                let bad = (ui | (cw - 1 - ui) | vi | (ch - 1 - vi) | (zi - 1)) >> 31;
                let id = ((vi as u32).wrapping_mul(cw as u32).wrapping_add(ui as u32) as i32) | bad;
                uo[x] = id;
                rmin = if (id as u32) < rmin { id as u32 } else { rmin };
                rmax = if id > rmax { id } else { rmax };
            } else {
                uo[x] = ui;
                vo[x] = vi;
            }
        }
        if IDX {
            vo[0] = rmin as i32;
            vo[1] = rmax;
        }
    }

    /// Generic: rs2_deproject → rs2_transform_point_to_point → rs2_project_point_to_pixel 순서 그대로
    #[inline(always)]
    fn corner_row_generic<const SEP: bool, const CMODEL: i32>(&self, z: &[f32], y: usize, k: usize, uo: &mut [i32], vo: &mut [i32]) {
        let w = self.d.width as usize;
        let z = &z[..w];
        let (uo, vo) = (&mut uo[..w], &mut vo[..w]);
        let xs: &[f32] = if SEP { &self.xt[k][..w] } else { &self.xt[k][y * w..y * w + w] };
        let ys: &[f32] = if SEP { &self.xt[k][..w] } else { &self.yt[k][y * w..y * w + w] };
        let yrow = if SEP { self.yt[k][y] } else { 0.0 };
        let r = &self.e.rotation;
        let (r0, r1, r2, r3, r4, r5, r6, r7, r8) = (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]);
        let (t0, t1, t2) = (self.e.translation[0], self.e.translation[1], self.e.translation[2]);
        let (fx, fy, ppx, ppy) = (self.c.fx, self.c.fy, self.c.ppx, self.c.ppy);
        let cc = &self.c.coeffs;
        let (c0, c1, c2, c3, c4) = (cc[0], cc[1], cc[2], cc[3], cc[4]);
        let scale = self.unit;
        for x in 0..w {
            let depth = scale * z[x];
            let p0 = depth * xs[x];
            let p1 = depth * if SEP { yrow } else { ys[x] };
            let p2 = depth;
            let xx0 = r0 * p0 + r3 * p1 + r6 * p2 + t0;
            let yy0 = r1 * p0 + r4 * p1 + r7 * p2 + t1;
            let zz0 = r2 * p0 + r5 * p1 + r8 * p2 + t2;
            let mut xx = xx0 / zz0;
            let mut yy = yy0 / zz0;
            if CMODEL == 1 {
                // MODIFIED / INVERSE BROWN_CONRADY
                let rr = xx * xx + yy * yy;
                let f = 1.0 + c0 * rr + c1 * rr * rr + c4 * rr * rr * rr;
                xx *= f;
                yy *= f;
                let dx = xx + 2.0 * c2 * xx * yy + c3 * (rr + 2.0 * xx * xx);
                let dy = yy + 2.0 * c3 * xx * yy + c2 * (rr + 2.0 * yy * yy);
                xx = dx;
                yy = dy;
            } else if CMODEL == 2 {
                // BROWN_CONRADY
                let rr = xx * xx + yy * yy;
                let f = 1.0 + c0 * rr + c1 * rr * rr + c4 * rr * rr * rr;
                let xf = xx * f;
                let yf = yy * f;
                let dx = xf + 2.0 * c2 * xx * yy + c3 * (rr + 2.0 * xx * xx);
                let dy = yf + 2.0 * c3 * xx * yy + c2 * (rr + 2.0 * yy * yy);
                xx = dx;
                yy = dy;
            }
            let u = xx * fx + ppx;
            let v = yy * fy + ppy;
            uo[x] = to_int(u + 0.5);
            vo[x] = to_int(v + 0.5);
        }
    }

    /// Sdk 핵심 루프 고르기: 표 모양(SEP) × 컬러 왜곡(DIST) × 빠른 식(FAST)
    #[inline(always)]
    fn sdk_row_impl<const IDX: bool>(&self, z: &[f32], y: usize, k: usize, uo: &mut [i32], vo: &mut [i32]) {
        let fast = !self.exact;
        match (self.sep, self.color_dist, fast) {
            (true, true, true) => self.corner_row_sdk::<true, true, IDX, true>(z, y, k, uo, vo),
            (true, true, false) => self.corner_row_sdk::<true, true, IDX, false>(z, y, k, uo, vo),
            (true, false, true) => self.corner_row_sdk::<true, false, IDX, true>(z, y, k, uo, vo),
            (true, false, false) => self.corner_row_sdk::<true, false, IDX, false>(z, y, k, uo, vo),
            (false, true, true) => self.corner_row_sdk::<false, true, IDX, true>(z, y, k, uo, vo),
            (false, true, false) => self.corner_row_sdk::<false, true, IDX, false>(z, y, k, uo, vo),
            (false, false, true) => self.corner_row_sdk::<false, false, IDX, true>(z, y, k, uo, vo),
            (false, false, false) => self.corner_row_sdk::<false, false, IDX, false>(z, y, k, uo, vo),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn sdk_row_avx2<const IDX: bool>(&self, z: &[f32], y: usize, k: usize, uo: &mut [i32], vo: &mut [i32]) {
        self.sdk_row_impl::<IDX>(z, y, k, uo, vo)
    }

    fn sdk_row<const IDX: bool>(&self, z: &[f32], y: usize, k: usize, uo: &mut [i32], vo: &mut [i32]) {
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            // SAFETY: AVX2 가 있는 것을 생성 때 확인함
            return unsafe { self.sdk_row_avx2::<IDX>(z, y, k, uo, vo) };
        }
        self.sdk_row_impl::<IDX>(z, y, k, uo, vo)
    }

    fn corner_row(&self, z: &[f32], y: usize, k: usize, uo: &mut [i32], vo: &mut [i32]) {
        if self.method == Method::Sdk {
            self.sdk_row::<false>(z, y, k, uo, vo);
        } else {
            let cm = if self.c.model == dist::MODIFIED_BROWN_CONRADY || self.c.model == dist::INVERSE_BROWN_CONRADY {
                1
            } else if self.c.model == dist::BROWN_CONRADY {
                2
            } else {
                0
            };
            match (self.sep, cm) {
                (true, 1) => self.corner_row_generic::<true, 1>(z, y, k, uo, vo),
                (true, 2) => self.corner_row_generic::<true, 2>(z, y, k, uo, vo),
                (true, _) => self.corner_row_generic::<true, 0>(z, y, k, uo, vo),
                (false, 1) => self.corner_row_generic::<false, 1>(z, y, k, uo, vo),
                (false, 2) => self.corner_row_generic::<false, 2>(z, y, k, uo, vo),
                (false, _) => self.corner_row_generic::<false, 0>(z, y, k, uo, vo),
            }
        }
    }

    /// 깊이 줄 [y0, y1) 를 정렬해 out 에 씀. 쓴 컬러 줄 범위를 sc.vmin/vmax 에 남김
    fn rows(&self, depth: &[u16], out: OutPtr, y0: usize, y1: usize, sc: &mut Scratch) {
        let w = self.d.width as usize;
        let (cw, ch) = (self.c.width, self.c.height);
        let out = out.0;
        // 비었으면 새 값, 아니면 가까운 값
        #[inline(always)]
        unsafe fn put(o: *mut u16, zz: u16) {
            unsafe {
                let cur = *o;
                *o = if cur != 0 && cur < zz { cur } else { zz };
            }
        }
        let (mut vmin, mut vmax) = (ch, -1i32);
        for y in y0..y1 {
            let z = &depth[y * w..y * w + w];
            for (f, &v) in sc.zf.iter_mut().zip(z) {
                *f = v as f32;
            }
            if !self.rect {
                // Sdk, 왼쪽 위 모서리만: 화소 번호까지 벡터 루프에서 내고 여기서는 흩뿌리기만
                let mut mm = [0i32; 2];
                self.sdk_row::<true>(&sc.zf, y, 0, &mut sc.u0, &mut mm);
                for (&i, &zz) in sc.u0.iter().zip(z) {
                    if i < 0 {
                        continue;
                    }
                    // SAFETY: i 는 0..CW*CH(화면 안만 남김)
                    unsafe { put(out.add(i as usize), zz) };
                }
                if mm[1] >= 0 {
                    vmin = vmin.min(mm[0] / cw);
                    vmax = vmax.max(mm[1] / cw);
                }
                continue;
            }
            {
                let Scratch { u0, v0, u1, v1, zf, .. } = sc;
                self.corner_row(zf, y, 0, u0, v0);
                self.corner_row(zf, y, 1, u1, v1);
            }
            let (u0, v0, u1, v1) = (&sc.u0, &sc.v0, &sc.u1, &sc.v1);
            if self.method == Method::Sdk {
                // move_depth_to_other: 화소마다 화면 안만
                for x in 0..w {
                    let zz = z[x];
                    if zz == 0 {
                        continue;
                    }
                    let (ya, yb) = (v0[x].max(0), v1[x].min(ch - 1));
                    let (xa, xb) = (u0[x].max(0), u1[x].min(cw - 1));
                    if ya > yb || xa > xb {
                        continue;
                    }
                    vmin = vmin.min(ya);
                    vmax = vmax.max(yb);
                    if yb - ya <= 1 && xb - xa <= 1 {
                        // 흔한 1×1 ~ 2×2: 루프 없이(쓰는 값은 최솟값이라 같은 칸을 두 번 써도 결과 같음)
                        unsafe {
                            let r0 = out.add(ya as usize * cw as usize + xa as usize);
                            let dx = (xb - xa) as usize;
                            put(r0, zz);
                            put(r0.add(dx), zz);
                            if yb > ya {
                                let r1 = r0.add(cw as usize);
                                put(r1, zz);
                                put(r1.add(dx), zz);
                            }
                        }
                        continue;
                    }
                    for oy in ya..=yb {
                        let row = unsafe { out.add(oy as usize * cw as usize) };
                        for ox in xa..=xb {
                            unsafe { put(row.add(ox as usize), zz) };
                        }
                    }
                }
            } else {
                // align_images: 사각형이 화면을 벗어나면 통째로 버림
                for x in 0..w {
                    let zz = z[x];
                    if zz == 0 {
                        continue;
                    }
                    if u0[x] < 0 || v0[x] < 0 || u1[x] >= cw || v1[x] >= ch {
                        continue;
                    }
                    if v0[x] > v1[x] || u0[x] > u1[x] {
                        continue;
                    }
                    for oy in v0[x]..=v1[x] {
                        let row = unsafe { out.add(oy as usize * cw as usize) };
                        for ox in u0[x]..=u1[x] {
                            unsafe { put(row.add(ox as usize), zz) };
                        }
                    }
                    vmin = vmin.min(v0[x]);
                    vmax = vmax.max(v1[x]);
                }
            }
        }
        sc.vmin = vmin;
        sc.vmax = vmax;
    }
}
