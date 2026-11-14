//! TUM RGB-D 형식: 목록 파일(`rgb.txt` · `depth.txt` · 녹화 목록 = "시각 경로" 줄), 위치 파일("t x y z qx qy qz qw" 줄),
//! 시퀀스별 카메라 보정값. `#` 줄과 빈 줄은 건너뛴다.

/// 목록 줄마다 (시각 글자, 경로 글자): 앞 두 낱말(낱말이 둘 안 되는 줄은 건너뜀)
pub fn pairs(text: &str) -> impl Iterator<Item = (&str, &str)> {
    text.lines().filter(|l| !l.starts_with('#')).filter_map(|l| {
        let mut it = l.split_whitespace();
        Some((it.next()?, it.next()?))
    })
}

/// 목록(시각, 경로): 시각이 숫자가 아닌 줄은 건너뜀
pub fn list(text: &str) -> Vec<(f64, String)> {
    pairs(text).filter_map(|(t, p)| Some((t.parse().ok()?, p.to_string()))).collect()
}

/// 목록 파일 읽기
pub fn read_list(path: impl AsRef<std::path::Path>) -> std::io::Result<Vec<(f64, String)>> {
    Ok(list(&std::fs::read_to_string(path)?))
}

/// 숫자 줄: `#` 이 아닌 줄마다 숫자로 읽히는 낱말들(위치 파일은 길이 8 = t x y z qx qy qz qw)
pub fn rows(text: &str) -> impl Iterator<Item = Vec<f64>> + '_ {
    text.lines().filter(|l| !l.starts_with('#')).map(|l| l.split_whitespace().filter_map(|x| x.parse().ok()).collect())
}

/// 시퀀스별 카메라 보정값 fx fy cx cy(TUM 문서): 폴더 이름으로 freiburg1 · freiburg2 · 그 밖(freiburg3)
pub fn camera_k(dir: &str) -> [f64; 4] {
    if dir.contains("freiburg1") {
        [517.3, 516.5, 318.6, 255.3]
    } else if dir.contains("freiburg2") {
        [520.9, 521.0, 325.1, 249.7]
    } else {
        [535.4, 539.2, 320.1, 247.6]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn list_skips_comments_blank_and_bad() {
        let t = "# timestamp filename\n1.5 rgb/a.png\n\n   \nx rgb/b.png\n2.0 rgb/c.png extra\r\n3.0\n";
        assert_eq!(list(t), vec![(1.5, "rgb/a.png".to_string()), (2.0, "rgb/c.png".to_string())]);
        let r: Vec<Vec<f64>> = rows("# h\n1 2 3 4 5 6 7 8\n1 a 2\n").collect();
        assert_eq!(r, vec![vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![1., 2.]]);
    }
}
