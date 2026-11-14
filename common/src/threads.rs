//! 스레드 배치: 프로세스가 받은 코어, 스레드 하나의 우선순위(nice)·코어 묶기, 이름 붙은 rayon 풀.
//! 리눅스(보드 · WSL)에서만 묶고, 다른 곳에서는 그대로 둔다(실패해도 진행).
use std::sync::OnceLock;

/// 처음 물었을 때 프로세스가 받은 코어(taskset 으로 준 것). 스레드를 묶은 뒤에도 "코어 전부" 는 이 값
static FULL: OnceLock<Vec<usize>> = OnceLock::new();

/// 프로세스가 쓸 수 있는 코어 번호(처음 부를 때 읽어 둔 값)
pub fn allowed_cpus() -> Vec<usize> {
    FULL.get_or_init(read_affinity).clone()
}

fn read_affinity() -> Vec<usize> {
    #[cfg(target_os = "linux")]
    unsafe {
        unsafe extern "C" {
            fn sched_getaffinity(pid: i32, size: usize, mask: *mut u64) -> i32;
        }
        let mut m = [0u64; 16];
        if sched_getaffinity(0, 128, m.as_mut_ptr()) == 0 {
            return (0..1024).filter(|i| m[i / 64] >> (i % 64) & 1 == 1).collect();
        }
    }
    (0..std::thread::available_parallelism().map_or(1, |n| n.get())).collect()
}

/// 코어 목록 → 묶음 비트(앞 64 코어)
pub fn bits(v: &[usize]) -> u64 {
    v.iter().fold(0u64, |m, &c| if c < 64 { m | 1 << c } else { m })
}

/// 묶음 비트(0 = 프로세스 코어 전부)의 코어 수
pub fn count(mask: u64) -> usize {
    (if mask == 0 { bits(&allowed_cpus()) } else { mask }).count_ones().max(1) as usize
}

/// 지금 스레드의 우선순위·코어 묶음(묶음 0 = 프로세스 코어 전부)
pub fn apply(nice: i32, mask: u64) {
    #[cfg(target_os = "linux")]
    unsafe {
        unsafe extern "C" {
            fn setpriority(which: i32, who: u32, prio: i32) -> i32;
            fn gettid() -> i32;
            fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
        }
        let tid = gettid();
        setpriority(0, tid as u32, nice); // PRIO_PROCESS + 스레드 id = 그 스레드만
        let mask = if mask == 0 { bits(&allowed_cpus()) } else { mask };
        if mask != 0 {
            let mut m = [0u64; 16];
            m[0] = mask;
            sched_setaffinity(tid, 128, m.as_ptr());
        }
    }
    let _ = (nice, mask);
}

/// 지금 스레드 id(리눅스, 다른 곳은 0)
pub fn tid() -> i32 {
    #[cfg(target_os = "linux")]
    unsafe {
        unsafe extern "C" {
            fn gettid() -> i32;
        }
        return gettid();
    }
    #[allow(unreachable_code)]
    0
}

/// 이름 붙은 rayon 풀: 스레드마다 시작할 때 우선순위·코어 묶음을 건다. 스레드 수 = 묶음의 코어 수
pub fn pool(name: &'static str, nice: i32, mask: u64) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(count(mask))
        .thread_name(move |i| format!("{name}-{i}"))
        .start_handler(move |_| apply(nice, mask))
        .build()
        .expect("스레드 풀")
}
