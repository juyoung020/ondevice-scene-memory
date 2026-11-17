//! 장면 한 장이 카메라 → 위치 → 분할 → 물체 기억 → 뷰어까지 가는 길의 계측(늘 켜 둠, 원자 값 더하기만).
//! - 복사 [`copy`]: 어디서 몇 바이트를 몇 번 옮겼는지(버퍼 → 버퍼 복사, 형식 바꾸며 옮기기 포함)
//! - 기다림 [`wait`]: 큐 · 락 · NPU 결과를 기다리며 멈춘 시간과 횟수. 락은 [`lock`] 이 먼저 바로 잡아 보고 막힐 때만 잰다
//! - 스레드 [`threads`]: 스레드 이름별 CPU 시간 · 깨어난 횟수(스스로 잠든 횟수 = voluntary 문맥 전환) · 밀려난 횟수.
//!   끝난 스레드는 [`track_thread`] 를 부른 스레드만 끝날 때 몫을 남긴다(리눅스만)
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

/// 복사 자리
#[derive(Clone, Copy)]
pub enum C {
    /// 분할 전처리: 컬러 → NPU 입력(640×640 레터박스, u8)
    SegLetterbox,
    /// PC ONNX: u8 NHWC → f32 NCHW
    SegNchw,
    /// NPU 출력을 DXRT 안 버퍼에서 우리 버퍼로(wait 의 memcpy)
    SegNpuOut,
    /// PC ONNX: 출력 텐서 → 후처리 버퍼
    SegOnnxOut,
    /// 사진 벡터 전처리: 물체 사진 → 인코더 입력(f32)
    EmbedIn,
    /// 사진 벡터 출력을 DXRT 안 버퍼에서 우리 버퍼로
    EmbedNpuOut,
    /// 사진 벡터 → 파일 바이트
    EmbedBytes,
    /// 사진 벡터 파일을 디스크에(메모리 기억 폴더인데도 쓰던 것)
    EmbedDisk,
    /// 물체 사진 자르기(컬러 · 깊이 mm · 마스크)
    Crop,
    /// 물체 사진 PNG 짜기에 넘김(Arc 나눠 씀: 바이트 0)
    CropShare,
    /// 깊이 u16 → 미터 f32(본 스레드, 장면마다)
    DepthMeters,
    /// 뷰 스레드에 넘기는 지나온 길 점
    ViewTraj,
}
pub const COPY_NAMES: [&str; 12] = [
    "seg_letterbox", "seg_nchw_f32", "seg_npu_out_memcpy", "seg_onnx_out", "embed_in", "embed_npu_out_memcpy", "embed_bytes", "embed_disk_write",
    "crop", "crop_png_share", "depth_meters", "view_traj",
];
static CN: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
static CB: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];

/// 복사 한 번(bytes 바이트)
#[inline]
pub fn copy(c: C, bytes: usize) {
    CN[c as usize].fetch_add(1, Relaxed);
    CB[c as usize].fetch_add(bytes as u64, Relaxed);
}

/// 기다림 자리
#[derive(Clone, Copy)]
pub enum W {
    /// 읽기: 다음 장면(위치 프로그램 큐)
    ReaderNext,
    /// 읽기: NPU 자리(함께 올린 수가 찼음)
    ReaderNpuSlot,
    /// 읽기: 분할 단계에 넘기기(앞 장면이 아직)
    ReaderHand,
    /// 읽기: 본 스레드가 앞 장면을 받아 갈 때까지
    ReaderTaken,
    /// 읽기: 다음 장면 고르기를 늦춤(추론 끝 무렵에 고르기)
    ReaderHold,
    /// 분할 받기: 다음 일
    SegIdle,
    /// 분할 받기: NPU 결과
    SegNpu,
    /// 분할 받기: 본 스레드에 넘기기
    SegHand,
    /// 본 스레드: 다음 장면
    MainFrame,
    /// 사진 벡터: 다음 일
    EmbedIdle,
    /// 사진 벡터: NPU 결과
    EmbedNpu,
    /// 락: 사진 벡터 이름표(본 스레드 · 사진 벡터 스레드)
    LockEmbed,
    /// 락: 본 스레드가 받아 간 수
    LockTaken,
    /// 락: 분할 단계 시간 합
    LockSegStats,
    /// 락: 기록 줄
    LockWriteQ,
    /// 락: 깊이 버퍼 줄
    LockDepthPool,
}
pub const WAIT_NAMES: [&str; 16] = [
    "reader_next", "reader_npu_slot", "reader_hand", "reader_taken", "reader_hold", "seg_idle", "seg_npu", "seg_hand", "main_frame", "embed_idle",
    "embed_npu", "lock_embed", "lock_taken", "lock_seg_stats", "lock_writeq", "lock_depth_pool",
];
static WN: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
static WT: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];

/// 기다림 한 번(c0 부터 지금까지)
#[inline]
pub fn wait(w: W, c0: Instant) {
    wait_ns(w, c0.elapsed().as_nanos() as u64);
}
#[inline]
pub fn wait_ns(w: W, ns: u64) {
    WN[w as usize].fetch_add(1, Relaxed);
    WT[w as usize].fetch_add(ns, Relaxed);
}

/// 일한 시간 자리(나눠 돌릴지 판단용)
#[derive(Clone, Copy)]
pub enum T {
    /// 본 스레드: 물체 하나의 마스크 × 깊이 → 세계 점(거르기 · 중앙값 포함)
    ObjPoints,
    /// 본 스레드: 물체 기억 찾기 · 갱신(사진 자르기 포함)
    ObjUpdate,
}
pub const TIME_NAMES: [&str; 2] = ["obj_points", "obj_update"];
static TN: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static TT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
#[inline]
pub fn time(t: T, c0: Instant) {
    TN[t as usize].fetch_add(1, Relaxed);
    TT[t as usize].fetch_add(c0.elapsed().as_nanos() as u64, Relaxed);
}
/// 일한 시간: [(이름, 횟수, 초)]
pub fn times() -> Vec<(&'static str, u64, f64)> {
    (0..TIME_NAMES.len()).map(|i| (TIME_NAMES[i], TN[i].load(Relaxed), TT[i].load(Relaxed) as f64 * 1e-9)).collect()
}

/// 락 잡기: 바로 잡히면 재지 않고, 막히면 막힌 시간을 남긴다
#[inline]
pub fn lock<T>(m: &Mutex<T>, w: W) -> MutexGuard<'_, T> {
    match m.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::Poisoned(e)) => panic!("{e}"),
        Err(std::sync::TryLockError::WouldBlock) => {
            let c = Instant::now();
            let g = m.lock().unwrap();
            wait(w, c);
            g
        }
    }
}

/// 다시 셈(리셋 · 실행마다)
pub fn reset() {
    for a in CN.iter().chain(&CB).chain(&WN).chain(&WT).chain(&TN).chain(&TT) {
        a.store(0, Relaxed);
    }
}

/// 복사: [(이름, 횟수, 바이트)]
pub fn copies() -> Vec<(&'static str, u64, u64)> {
    (0..COPY_NAMES.len()).map(|i| (COPY_NAMES[i], CN[i].load(Relaxed), CB[i].load(Relaxed))).collect()
}
/// 기다림: [(이름, 횟수, 초)]
pub fn waits() -> Vec<(&'static str, u64, f64)> {
    (0..WAIT_NAMES.len()).map(|i| (WAIT_NAMES[i], WN[i].load(Relaxed), WT[i].load(Relaxed) as f64 * 1e-9)).collect()
}

/// 스레드 하나의 몫
#[derive(Clone, Debug, Default)]
pub struct ThreadStat {
    pub name: String,
    /// 같은 이름 스레드 수
    pub n: u32,
    pub cpu_s: f64,
    /// 스스로 잠든 횟수(기다림 · 잠 → 다시 깨어남)
    pub wakeups: u64,
    /// 밀려난 횟수(다른 스레드에 코어를 뺏김)
    pub preempted: u64,
}

/// 끝난 스레드의 몫(이름별)
static GONE: Mutex<Vec<ThreadStat>> = Mutex::new(Vec::new());

#[cfg(target_os = "linux")]
fn read_task(dir: &str) -> Option<ThreadStat> {
    let name = std::fs::read_to_string(format!("{dir}/comm")).ok()?.trim().to_string();
    let ss = std::fs::read_to_string(format!("{dir}/schedstat")).ok()?;
    let cpu_ns: u64 = ss.split_whitespace().next()?.parse().ok()?;
    let st = std::fs::read_to_string(format!("{dir}/status")).ok()?;
    let field = |k: &str| st.lines().find_map(|l| l.strip_prefix(k)).and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0);
    Some(ThreadStat { name, n: 1, cpu_s: cpu_ns as f64 * 1e-9, wakeups: field("voluntary_ctxt_switches:"), preempted: field("nonvoluntary_ctxt_switches:") })
}

struct Track;
impl Drop for Track {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        if let Some(s) = read_task("/proc/thread-self") {
            add(&mut GONE.lock().unwrap(), s);
        }
    }
}
thread_local! { static TRACK: std::cell::OnceCell<Track> = const { std::cell::OnceCell::new() }; }

/// 이 스레드가 끝날 때 몫을 남김(스레드를 띄워 놓고 끝내는 곳: 분할 단계 · 다시 쌓기 등). 여러 번 불러도 한 번
pub fn track_thread() {
    TRACK.with(|t| {
        let _ = t.get_or_init(|| Track);
    });
}

fn add(v: &mut Vec<ThreadStat>, s: ThreadStat) {
    match v.iter_mut().find(|e| e.name == s.name) {
        Some(e) => {
            e.n += s.n;
            e.cpu_s += s.cpu_s;
            e.wakeups += s.wakeups;
            e.preempted += s.preempted;
        }
        None => v.push(s),
    }
}

/// 스레드 이름별 몫(살아 있는 스레드 + 끝난 스레드, 이름 숫자 꼬리는 묶음: `pts-3` → `pts-*`), 살아 있는 스레드 수
pub fn threads() -> (Vec<ThreadStat>, usize) {
    let mut v = GONE.lock().unwrap().clone();
    let mut live = 0;
    #[cfg(target_os = "linux")]
    if let Ok(rd) = std::fs::read_dir("/proc/self/task") {
        for e in rd.flatten() {
            if let Some(mut s) = read_task(&e.path().to_string_lossy()) {
                live += 1;
                let base = s.name.trim_end_matches(|c: char| c.is_ascii_digit());
                if base.len() < s.name.len() && base.ends_with('-') {
                    s.name = format!("{base}*");
                }
                add(&mut v, s);
            }
        }
    }
    v.sort_by(|a, b| b.cpu_s.total_cmp(&a.cpu_s));
    (v, live)
}

/// 값 목록의 p50 · p95 · 최대
pub fn pct(v: &mut [f32]) -> [f32; 3] {
    if v.is_empty() {
        return [0.0; 3];
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
    [q(0.5), q(0.95), q(1.0)]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn counts_and_locks() {
        copy(C::Crop, 10);
        copy(C::Crop, 5);
        assert!(copies().iter().any(|c| c.0 == "crop" && c.1 >= 2 && c.2 >= 15));
        let m = Mutex::new(1);
        *lock(&m, W::LockTaken) += 1;
        assert_eq!(*m.lock().unwrap(), 2);
        assert_eq!(pct(&mut [3.0, 1.0, 2.0]), [2.0, 2.0, 3.0]);
    }
}
