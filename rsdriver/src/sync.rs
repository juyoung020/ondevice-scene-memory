//! 컬러·깊이 짝 맞추기. librealsense v2.59.1 `src/sync.cpp` timestamp_composite_matcher 의 우리 설정(같은 fps 두 스트림) 경우:
//! 두 시각 차가 (1000/fps)/2 ms 보다 작으면 한 묶음, 아니면 더 이른 쪽을 버리고 다음 것을 기다린다.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// 둘을 한 묶음으로
    Pair,
    /// 컬러가 더 이르다: 컬러를 버림
    DropColor,
    /// 깊이가 더 이르다: 깊이를 버림
    DropDepth,
}

/// are_equivalent(a, b, fps): |a − b| < (1000/fps)/2
pub fn are_equivalent(a_ms: f64, b_ms: f64, fps: f64) -> bool {
    let gap = 1000.0 / fps;
    (a_ms - b_ms).abs() < gap / 2.0
}

pub fn decide(color_ms: f64, depth_ms: f64, fps: f64) -> Decision {
    if are_equivalent(color_ms, depth_ms, fps) {
        Decision::Pair
    } else if color_ms < depth_ms {
        Decision::DropColor
    } else {
        Decision::DropDepth
    }
}
