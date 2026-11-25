//! f32 버퍼 되쓰기: 노드 출력을 다 쓴 버퍼에 다시 받아, 큰 버퍼를 매번 새로 받으며 생기는 페이지 부재(xfeat 한 번에 약 3,600 번)를 없앰
use std::any::TypeId;
use std::sync::Mutex;

static FREE: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::new());
const KEEP: usize = 64;

/// 0 으로 채운 길이 n 버퍼(f32 면 되쓴 것)
pub fn zeros<T: Copy + Default + 'static>(n: usize) -> Vec<T> {
    if TypeId::of::<T>() != TypeId::of::<f32>() || n < 1024 {
        return vec![T::default(); n];
    }
    let got = {
        let mut f = FREE.lock().unwrap();
        // 맞는 것 중 가장 작은 것
        let mut best: Option<usize> = None;
        for (i, v) in f.iter().enumerate() {
            if v.capacity() >= n && v.capacity() <= 2 * n && best.is_none_or(|b| f[b].capacity() > v.capacity()) {
                best = Some(i);
            }
        }
        best.map(|i| f.swap_remove(i))
    };
    let mut v = got.unwrap_or_else(|| Vec::with_capacity(n));
    v.clear();
    v.resize(n, 0.0);
    // SAFETY: T 는 f32
    unsafe { std::mem::transmute::<Vec<f32>, Vec<T>>(v) }
}

/// 길이 n 버퍼, 값은 아무것(부르는 쪽이 모두 덮어씀): 되쓴 버퍼면 0 채우기 없이
pub fn take<T: Copy + Default + 'static>(n: usize) -> Vec<T> {
    if TypeId::of::<T>() != TypeId::of::<f32>() || n < 1024 {
        return vec![T::default(); n];
    }
    let got = {
        let mut f = FREE.lock().unwrap();
        // 담을 수 있는 것 중 가장 작은 것
        let mut best: Option<usize> = None;
        for (i, v) in f.iter().enumerate() {
            if v.capacity() >= n && v.capacity() <= 2 * n && best.is_none_or(|b| f[b].capacity() > v.capacity()) {
                best = Some(i);
            }
        }
        best.map(|i| f.swap_remove(i))
    };
    let v = match got {
        Some(mut v) => {
            // 채워진 길이보다 길면 모자란 곳만 0
            if v.len() >= n {
                v.truncate(n);
            } else {
                v.resize(n, 0.0);
            }
            v
        }
        None => vec![0f32; n],
    };
    // SAFETY: T 는 f32
    unsafe { std::mem::transmute::<Vec<f32>, Vec<T>>(v) }
}

/// 다 쓴 버퍼 돌려주기
pub fn give(v: Vec<f32>) {
    if v.capacity() < 1024 {
        return;
    }
    // 받을 수 있는 길이 = 지금 길이(채워진 곳만 되씀, 0 채우기 없음)
    if v.len() < 1024 {
        return;
    }
    let mut f = FREE.lock().unwrap();
    if f.len() >= KEEP {
        // 가장 작은 것을 버림
        let (i, _) = f.iter().enumerate().min_by_key(|(_, v)| v.capacity()).unwrap();
        if f[i].capacity() >= v.capacity() {
            return;
        }
        f.swap_remove(i);
    }
    f.push(v);
}
