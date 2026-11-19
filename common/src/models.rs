//! 모델 폴더: $ONDEVICE_MODELS, 없으면 보드(dxrt) ~/models, PC 는 작업 폴더의 models/ · /mnt/c/ondevice/models(WSL) 가운데 있는 곳.
//! 분할 · 사진 벡터 · 루프 클로저 신경망이 모두 이 규칙으로 찾는다
use std::path::PathBuf;

pub fn dir() -> PathBuf {
    if let Some(d) = std::env::var_os("ONDEVICE_MODELS") {
        return d.into();
    }
    if cfg!(feature = "dxrt") {
        if let Some(h) = std::env::var_os("HOME") {
            let d = PathBuf::from(h).join("models");
            if d.is_dir() {
                return d;
            }
        }
    }
    let local = PathBuf::from("models");
    if !local.is_dir() {
        let wsl = PathBuf::from("/mnt/c/ondevice/models");
        if wsl.is_dir() {
            return wsl;
        }
    }
    local
}
