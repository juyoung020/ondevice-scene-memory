//! 같은 프로세스 공유(memmap --serve): memmap 이 자주 바꾸는 파일(view.json · pose.json · map.pgm · map.yaml)을 쓸 때
//! 그 바이트를 여기에도 걸어 두면, 서버는 파일을 stat·다시 읽지 않고 메모리에서 바로 보낸다.
//! 걸어 둔 적 없는 경로(물체 사진·점 파일 등)는 지금처럼 파일에서 읽는다. 파일은 그대로 쓰이므로 기록·검증용 출력은 같다.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// 메모리에 걸어 둘 파일 이름(기억 폴더 바로 아래)
pub const HOT: [&str; 4] = ["view.json", "pose.json", "map.pgm", "map.yaml"];

static ON: AtomicBool = AtomicBool::new(false);
static SEQ: AtomicU64 = AtomicU64::new(0);
static FILES: Mutex<Option<HashMap<PathBuf, (String, Arc<Vec<u8>>)>>> = Mutex::new(None);

/// 켬(memmap --serve): 이후 publish 한 경로는 서버가 메모리에서 읽는다
pub fn enable() {
    FILES.lock().unwrap().get_or_insert_with(HashMap::new);
    ON.store(true, Ordering::Release);
}

pub fn enabled() -> bool {
    ON.load(Ordering::Acquire)
}

/// 메모리에 걸 파일인지(켜져 있고 이름이 HOT)
pub fn is_hot(p: &Path) -> bool {
    enabled() && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| HOT.contains(&n))
}

/// 파일을 다 쓴 뒤 같은 바이트를 건다(판 = 걸 때마다 늘어나는 번호 + 길이)
pub fn publish(p: &Path, bytes: Arc<Vec<u8>>) {
    let ver = format!("m{}-{}", SEQ.fetch_add(1, Ordering::Relaxed) + 1, bytes.len());
    if let Some(m) = FILES.lock().unwrap().as_mut() {
        m.insert(p.to_path_buf(), (ver, bytes));
    }
}

/// 걸린 판과 바이트(없으면 None: 파일에서 읽음)
pub fn get(p: &Path) -> Option<(String, Arc<Vec<u8>>)> {
    if !enabled() {
        return None;
    }
    FILES.lock().unwrap().as_ref()?.get(p).cloned()
}
