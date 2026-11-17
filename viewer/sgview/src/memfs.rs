//! 같은 프로세스 공유(`ondevice --serve`): 기억 폴더에 쓰는 모든 파일(view.json · pose.json · map.pgm · 물체 사진 · 점 파일 ·
//! 사진 벡터 · vocab.json …)을 다 쓴 뒤 같은 바이트를 여기에도 걸어 두면, 서버는 그 파일을 stat·다시 읽지 않고 메모리에서 보낸다.
//! 지우는 파일·폴더는 여기서도 함께 지운다(디스크와 같은 차례). 걸어 둔 적 없는 경로(따로 띄운 서버, 다른 프로그램이 쓴 파일)는
//! 파일에서 읽는다. 파일은 그대로 쓰이므로 기억 폴더(결과물)는 같다.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

static ON: AtomicBool = AtomicBool::new(false);
static SEQ: AtomicU64 = AtomicU64::new(0);
static FILES: Mutex<Option<HashMap<PathBuf, (String, Arc<Vec<u8>>)>>> = Mutex::new(None);

/// 경로 열쇠: 구성 요소로 다시 이음(`a/./b`, `dir.join("x/y")` 와 하나씩 붙인 경로가 같은 열쇠)
fn key(p: &Path) -> PathBuf {
    p.components().collect()
}

/// 켬(`ondevice --serve`): 이후 publish 한 경로는 서버가 메모리에서 읽는다
pub fn enable() {
    FILES.lock().unwrap().get_or_insert_with(HashMap::new);
    ON.store(true, Ordering::Release);
}

pub fn enabled() -> bool {
    ON.load(Ordering::Acquire)
}

/// 파일을 다 쓴 뒤 같은 바이트를 건다(판 = 걸 때마다 늘어나는 번호 + 길이). /stream 을 깨움
pub fn publish(p: &Path, bytes: Arc<Vec<u8>>) {
    let pose = publish_quiet(p, bytes);
    if pose {
        bump_pose();
    } else {
        bump();
    }
}

/// 걸기만 하고 /stream 은 깨우지 않음(여러 파일을 잇달아 걸고 끝에 [`bump`] 한 번). 돌려줌: pose.json 인지
pub fn publish_quiet(p: &Path, bytes: Arc<Vec<u8>>) -> bool {
    let ver = format!("m{}-{}", SEQ.fetch_add(1, Ordering::Relaxed) + 1, bytes.len());
    if let Some(m) = FILES.lock().unwrap().as_mut() {
        m.insert(key(p), (ver, bytes));
    }
    p.file_name().is_some_and(|f| f == "pose.json")
}

/// 파일 하나를 내림
pub fn remove(p: &Path) {
    OTHER.fetch_add(1, Ordering::Relaxed);
    if let Some(m) = FILES.lock().unwrap().as_mut() {
        m.remove(&key(p));
    }
}

/// 폴더 아래 파일을 모두 내림
pub fn remove_dir(p: &Path) {
    OTHER.fetch_add(1, Ordering::Relaxed);
    let k = key(p);
    if let Some(m) = FILES.lock().unwrap().as_mut() {
        m.retain(|q, _| !q.starts_with(&k));
    }
}

/// 걸린 판과 바이트(없으면 None: 파일에서 읽음)
pub fn get(p: &Path) -> Option<(String, Arc<Vec<u8>>)> {
    if !enabled() {
        return None;
    }
    FILES.lock().unwrap().as_ref()?.get(&key(p)).cloned()
}

/// 바이트: 걸린 것이 있으면 그것(복사 없이), 아니면 파일
pub fn read(p: &Path) -> Option<Arc<Vec<u8>>> {
    match get(p) {
        Some((_, b)) => Some(b),
        None => std::fs::read(p).ok().map(Arc::new),
    }
}

/// 판이 바뀔 때마다 늘어나는 번호(파일 걸기 · 지우기 · view 판 걸기): /stream 은 이것을 기다렸다 바뀐 것만 보낸다
static CHANGE: Mutex<u64> = Mutex::new(0);
static CHANGED: Condvar = Condvar::new();
/// pose.json 말고 다른 것이 바뀐 번호
static OTHER: AtomicU64 = AtomicU64::new(0);
/// 깨운 횟수(계측: 파일을 걸 때마다 깨우던 것을 묶은 효과)
static BUMPS: AtomicU64 = AtomicU64::new(0);
pub fn bumps() -> u64 {
    BUMPS.load(Ordering::Relaxed)
}

pub fn other() -> u64 {
    OTHER.load(Ordering::Relaxed)
}

pub fn bump() {
    OTHER.fetch_add(1, Ordering::Relaxed);
    bump_pose();
}

/// pose.json 만 바뀜(장면 사이 위치 200 Hz): /stream 은 위치만 보내고 나머지 판은 보지 않음
pub fn bump_pose() {
    BUMPS.fetch_add(1, Ordering::Relaxed);
    *CHANGE.lock().unwrap() += 1;
    CHANGED.notify_all();
}

/// 번호가 seen 과 달라질 때까지(최대 timeout) 기다려 지금 번호를 돌려줌
pub fn wait_change(seen: u64, timeout: std::time::Duration) -> u64 {
    let g = CHANGE.lock().unwrap();
    let (g, _) = CHANGED.wait_timeout_while(g, timeout, |c| *c == seen).unwrap();
    *g
}

/// 폴더 아래 걸린 파일 경로(기억 폴더를 디스크 대신 메모리에 둘 때 목록 보기)
pub fn list(dir: &Path) -> Vec<PathBuf> {
    let k = key(dir);
    FILES.lock().unwrap().as_ref().map_or_else(Vec::new, |m| m.keys().filter(|q| q.starts_with(&k)).cloned().collect())
}

/// 걸린 파일 수 · 바이트 합(메모리 기억 폴더 크기 재기)
pub fn usage() -> (usize, u64) {
    FILES.lock().unwrap().as_ref().map_or((0, 0), |m| (m.len(), m.values().map(|(_, b)| b.len() as u64).sum()))
}

/// 걸린 파일 전부(디스크에 저장할 때)
pub fn all() -> Vec<(PathBuf, Arc<Vec<u8>>)> {
    FILES.lock().unwrap().as_ref().map_or_else(Vec::new, |m| m.iter().map(|(p, (_, b))| (p.clone(), b.clone())).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn publish_get_remove() {
        enable();
        let d = std::env::temp_dir().join("memfs_test_dir");
        publish(&d.join("scene/5/n0.bin"), Arc::new(vec![1, 2]));
        publish(&d.join("objects").join("O1_rgb.png"), Arc::new(vec![3]));
        assert_eq!(get(&d.join("scene").join("5").join("n0.bin")).unwrap().1.as_slice(), &[1, 2]);
        remove_dir(&d.join("scene"));
        assert!(get(&d.join("scene/5/n0.bin")).is_none());
        assert_eq!(read(&d.join("objects/O1_rgb.png")).unwrap().as_slice(), &[3]);
        remove(&d.join("objects/O1_rgb.png"));
        assert!(get(&d.join("objects/O1_rgb.png")).is_none());
    }
}
