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

/// 실시간 배치(`--rt`): 역할 이름마다 SCHED_FIFO 우선순위(1..99)와 코어 묶음(선택).
/// 형식 `camera=50@1,infer=40,reader=30@0+1` (역할=우선순위[@코어 목록, `a-b`·`+` 로 잇기]), `auto` = camera=50, infer=40, reader=30, main=20, `none`/빈칸 = 끔.
/// 권한(CAP_SYS_NICE·rtprio 한도)이 없으면 조용히 보통 스케줄로 남는다. 출력 값에는 영향 없음(순서·차례만)
static RT: OnceLock<Vec<(String, i32, u64)>> = OnceLock::new();

/// 코어 목록 글 `0-2+5` → 묶음 비트
pub fn parse_cores(s: &str) -> u64 {
    let mut cs = vec![];
    for r in s.split('+').filter(|x| !x.is_empty()) {
        let (a, b) = r.split_once('-').unwrap_or((r, r));
        if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
            cs.extend(a..=b);
        }
    }
    bits(&cs)
}

/// `--rt` 해석(처음 한 번). 잘못된 조각은 건너뜀
pub fn set_rt(spec: &str) {
    let spec = if spec == "auto" { "camera=50,infer=40,reader=30,main=20" } else { spec };
    let v = spec
        .split(',')
        .filter(|p| !p.is_empty() && *p != "none")
        .filter_map(|p| {
            let (name, rest) = p.split_once('=')?;
            let (pr, cores) = rest.split_once('@').map_or((rest, ""), |x| x);
            Some((name.to_string(), pr.parse::<i32>().ok()?.clamp(1, 99), parse_cores(cores)))
        })
        .collect();
    let _ = RT.set(v);
}

/// 지금 스레드가 `name` 역할을 맡음: `--rt` 에 있으면 SCHED_FIFO 우선순위·코어 묶음을 건다(없거나 권한이 없으면 그대로). 걸었으면 true
pub fn rt(name: &str) -> bool {
    let Some(v) = RT.get().filter(|v| !v.is_empty()) else { return false };
    let Some(&(_, prio, mask)) = v.iter().find(|e| e.0 == name) else {
        other(); // 같은 스레드가 실시간 역할을 맡았다 내려놓을 때(본 스레드 → 추론 → 본 스레드)
        return false;
    };
    if mask != 0 {
        pin(mask);
    }
    fifo(prio)
}

/// 지금 스레드를 보통 스케줄(SCHED_OTHER)로
pub fn other() {
    #[cfg(target_os = "linux")]
    unsafe {
        unsafe extern "C" {
            fn gettid() -> i32;
            fn sched_setscheduler(pid: i32, policy: i32, param: *const i32) -> i32;
        }
        sched_setscheduler(gettid(), 0, &0);
    }
}

/// 지금 스레드를 묶음 비트의 코어에만(우선순위는 그대로)
pub fn pin(mask: u64) {
    #[cfg(target_os = "linux")]
    unsafe {
        unsafe extern "C" {
            fn gettid() -> i32;
            fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
        }
        let mut m = [0u64; 16];
        m[0] = mask;
        sched_setaffinity(gettid(), 128, m.as_ptr());
    }
    let _ = mask;
}

/// 지금 스레드를 SCHED_FIFO(우선순위 1..99)로. 권한이 없으면 false(보통 스케줄 그대로)
pub fn fifo(prio: i32) -> bool {
    #[cfg(target_os = "linux")]
    unsafe {
        unsafe extern "C" {
            fn gettid() -> i32;
            fn sched_setscheduler(pid: i32, policy: i32, param: *const i32) -> i32;
        }
        const SCHED_FIFO: i32 = 1;
        let p = prio.clamp(1, 99);
        return sched_setscheduler(gettid(), SCHED_FIFO, &p) == 0;
    }
    #[allow(unreachable_code)]
    {
        let _ = prio;
        false
    }
}

/// 프로세스 메모리를 RAM 에 묶음(`--mlock`: 지금 것 + 앞으로 할당할 것, 페이지 내보내기로 생기는 멈춤 없앰). 권한·한도가 모자라면 false
pub fn mlock_all() -> bool {
    #[cfg(target_os = "linux")]
    unsafe {
        unsafe extern "C" {
            fn mlockall(flags: i32) -> i32;
        }
        return mlockall(1 | 2) == 0; // MCL_CURRENT | MCL_FUTURE
    }
    #[allow(unreachable_code)]
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cores_and_rt_spec() {
        assert_eq!(parse_cores("0-2+5"), 0b100111);
        assert_eq!(parse_cores(""), 0);
        set_rt("camera=50@1,infer=200,bad,x=y");
        let v = RT.get().unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], ("camera".to_string(), 50, 2));
        assert_eq!(v[1].1, 99);
        // 없는 역할은 아무것도 하지 않음
        assert!(!rt("nothing"));
    }
}
