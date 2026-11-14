//! 같은 프로세스 공유(`ondevice --serve`): 기억 폴더에 쓰는 모든 파일(view.json · pose.json · map.pgm · 물체 사진 · 점 파일 ·
//! 사진 벡터 · vocab.json …)을 다 쓴 뒤 같은 바이트를 여기에도 걸어 두면, 서버는 그 파일을 stat·다시 읽지 않고 메모리에서 보낸다.
//! 지우는 파일·폴더는 여기서도 함께 지운다(디스크와 같은 차례). 걸어 둔 적 없는 경로(따로 띄운 서버, 다른 프로그램이 쓴 파일)는
//! 파일에서 읽는다. 파일은 그대로 쓰이므로 기억 폴더(결과물)는 같다.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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

/// 파일을 다 쓴 뒤 같은 바이트를 건다(판 = 걸 때마다 늘어나는 번호 + 길이)
pub fn publish(p: &Path, bytes: Arc<Vec<u8>>) {
    let ver = format!("m{}-{}", SEQ.fetch_add(1, Ordering::Relaxed) + 1, bytes.len());
    if let Some(m) = FILES.lock().unwrap().as_mut() {
        m.insert(key(p), (ver, bytes));
    }
}

/// 파일 하나를 내림
pub fn remove(p: &Path) {
    if let Some(m) = FILES.lock().unwrap().as_mut() {
        m.remove(&key(p));
    }
}

/// 폴더 아래 파일을 모두 내림
pub fn remove_dir(p: &Path) {
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
