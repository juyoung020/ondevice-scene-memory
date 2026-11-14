//! 카메라 장면 버퍼 돌려쓰기: D435 드라이버(rsdriver)가 장면마다 컬러(921,600 B)·정렬 깊이(614,400 B) 버퍼를 새로 잡아 0 으로 채우던 것을,
//! 받는 쪽(memmap 점 작업 스레드)이 다 쓴 버퍼를 돌려주면 다시 쓴다. 다시 쓸 때는 길이가 이미 맞아 0 으로 채우지 않는다
//! (컬러는 YUYV→RGB 가 모든 화소를 쓰고, 정렬 깊이는 `fast_align` 이 처음에 스스로 0 으로 채운 뒤 점을 찍음).
//! 카메라 입력이 켤 때만 받는다(녹화 재생은 돌려받을 곳이 없어 버퍼를 쌓지 않음).
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

/// 종류마다 이만큼까지만 쥠(큐에 떠 있는 장면 수보다 넉넉히)
const KEEP: usize = 4;

static ON: AtomicBool = AtomicBool::new(false);
static RGB: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());
static DEPTH: Mutex<Vec<Vec<u16>>> = Mutex::new(Vec::new());
static GRAY: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// 돌려받기 켜기(카메라 입력을 열 때)
pub fn enable() {
    ON.store(true, Relaxed);
}

pub fn enabled() -> bool {
    ON.load(Relaxed)
}

fn put<T>(pool: &Mutex<Vec<Vec<T>>>, v: Vec<T>) {
    if v.capacity() == 0 || !enabled() {
        return;
    }
    let mut p = pool.lock().unwrap();
    if p.len() < KEEP {
        p.push(v);
    } else if let Some(i) = (0..p.len()).min_by_key(|&i| p[i].capacity()).filter(|&i| p[i].capacity() < v.capacity()) {
        p[i] = v; // 가득 차면 가장 작은 것과 바꿈(크기가 안 맞는 버퍼가 자리를 막지 않게)
    }
}

/// 다 쓴 장면 버퍼 돌려주기(받는 쪽 어디서 버려지든: 장면 구조체의 Drop 이 부름)
pub fn give_gray(v: Vec<u8>) {
    put(&GRAY, v);
}

pub fn take_gray(n: usize) -> Vec<u8> {
    get(&GRAY, n)
}

/// 길이 n 이상 담을 수 있는 버퍼(돌려받은 것이 있으면 그것, 내용은 앞 장면 그대로 — 부르는 쪽이 모두 덮어씀). 없으면 빈 벡터
fn get<T>(pool: &Mutex<Vec<Vec<T>>>, n: usize) -> Vec<T> {
    let mut p = pool.lock().unwrap();
    match p.iter().position(|v| v.capacity() >= n) {
        Some(i) => p.swap_remove(i),
        None => Vec::new(),
    }
}

pub fn give_rgb(v: Vec<u8>) {
    put(&RGB, v);
}

pub fn give_depth(v: Vec<u16>) {
    put(&DEPTH, v);
}

pub fn take_rgb(n: usize) -> Vec<u8> {
    get(&RGB, n)
}

pub fn take_depth(n: usize) -> Vec<u16> {
    get(&DEPTH, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_only_when_on() {
        give_rgb(vec![7u8; 12]);
        assert!(take_rgb(12).is_empty()); // 꺼져 있으면 쥐지 않음
        enable();
        let v = vec![7u8; 12];
        let p = v.as_ptr();
        give_rgb(v);
        assert!(take_rgb(13).is_empty()); // 작으면 안 줌
        let r = take_rgb(12);
        assert_eq!((r.as_ptr(), r.len()), (p, 12)); // 같은 버퍼, 길이 그대로(0 채우기 없음)
        for _ in 0..10 {
            give_depth(vec![1u16; 4]);
        }
        assert_eq!(DEPTH.lock().unwrap().len(), KEEP);
    }
}
