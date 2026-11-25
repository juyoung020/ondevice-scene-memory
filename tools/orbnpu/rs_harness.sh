#!/bin/bash
# Rust 쪽 PC 비교 하네스를 저장소 밖(~/orbnpu_rs)에 만든다: opt/orb3-rust 의 orb3rs::extract(읽기 전용 복사, fast 자리만
# 갈고리로 바꿈) + 이 브랜치의 slam/src/nnrun.rs · orb3npu.rs(그대로 #[path]) + common(ONNX Runtime).
# 다른 브랜치를 병합하지 않으려고 저장소 밖에서만 합친다.
set -e
R=/mnt/c/ondevice-orb-npu
H=~/orbnpu_rs
mkdir -p $H/src/orb3rs
cd $R
for f in types stdsort extract; do git show opt/orb3-rust:slam/src/orb3rs/$f.rs > $H/src/orb3rs/$f.rs; done
git show opt/orb3-rust:slam/src/orb3rs/bit_pattern_31.in > $H/src/orb3rs/bit_pattern_31.in
E=$H/src/orb3rs/extract.rs
# 갈고리: fast → crate::hook::fast(level, ...), 피라미드는 밖에서 넣을 수 있게, levels 공개
sed -i 's/                    fast(img, iw, /                    crate::hook::fast(level, img, iw, /; s/                        fast(img, iw, /                        crate::hook::fast(level, img, iw, /' $E
sed -i 's/        self.compute_pyramid(gray, w, h);/        if !gray.is_empty() { self.compute_pyramid(gray, w, h); }/' $E
sed -i 's/^    levels: Vec<(Vec<u8>, usize, usize)>,/    pub levels: Vec<(Vec<u8>, usize, usize)>,/' $E
grep -c "crate::hook::fast" $E
cp $R/tools/orbnpu/harness_main.rs $H/src/main.rs
cat > $H/Cargo.toml <<EOF
[package]
name = "orbnpu_rs"
version = "0.1.0"
edition = "2024"
[dependencies]
common = { path = "$R/common", features = ["onnx"] }
ort = { version = "=2.0.0-rc.15" }
rayon = "1"
[features]
onnx = []
default = ["onnx"]
[profile.release]
opt-level = 3
EOF
cd $H && CARGO_TARGET_DIR=~/tgt-orbnpu cargo build --release 2>&1 | grep -E "^(error|warning: unused)|Finished" | head -20
