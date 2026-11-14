//! 사원수(x, y, z, w) → 회전 행렬(행 우선 3×3, f64). 위치 파일(TUM)·경로 맞추기·위치 추적이 함께 쓴다.

pub type M3 = [[f64; 3]; 3];

/// 단위 사원수 → 회전 행렬(정규화 없이 그대로)
pub fn quat_m(q: [f64; 4]) -> M3 {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// 사원수를 길이 1 로 나눈 뒤 회전 행렬
pub fn quat_to_m(q: [f64; 4]) -> M3 {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    quat_m([q[0] / n, q[1] / n, q[2] / n, q[3] / n])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalizes_then_same_formula() {
        let q = [0.1, -0.2, 0.3, 0.9];
        let n = (q.iter().map(|v| v * v).sum::<f64>()).sqrt();
        assert_eq!(quat_to_m(q), quat_m(q.map(|v| v / n)));
        assert_eq!(quat_m([0.0, 0.0, 0.0, 1.0]), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }
}
