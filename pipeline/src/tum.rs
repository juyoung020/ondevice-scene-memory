//! TUM RGB-D 시퀀스: rgb.txt · depth.txt · groundtruth.txt. 정답 포즈는 OpenCV 카메라 → 세계,
//! 손에 들고 한 높이로 돌았다고 보고 걸어간 경로 평면을 수평(z 위)으로 돌린다.
use crate::geom::*;
use std::path::{Path, PathBuf};

pub struct Tum {
    pub dir: PathBuf,
    pub rgb: Vec<(f64, String)>,
    pub depth: Vec<(f64, String)>,
    pub ts: Vec<f64>,
    pub ps: Vec<V3>,
    pub qs: Vec<[f64; 4]>,
    /// 수평 맞춤 회전(세계 → 수평 세계)
    pub level: M3,
    pub k: [f64; 4],
}

fn read_list(p: &Path) -> Vec<(f64, String)> {
    std::fs::read_to_string(p)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?.parse().ok()?, it.next()?.to_string()))
        })
        .collect()
}

impl Tum {
    /// 카메라 입력용 빈 시퀀스(내부 보정값만)
    pub fn empty(k: [f64; 4]) -> Tum {
        Tum { dir: PathBuf::new(), rgb: vec![], depth: vec![], ts: vec![], ps: vec![], qs: vec![], level: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], k }
    }

    /// gt: 위치 파일(TUM 형식). 없으면 폴더의 groundtruth.txt(다른 SLAM 이 낸 위치를 넣을 때 씀)
    pub fn open(dir: &Path, gt: Option<&Path>) -> Tum {
        let rgb = read_list(&dir.join("rgb.txt"));
        let depth = read_list(&dir.join("depth.txt"));
        let (mut ts, mut ps, mut qs) = (vec![], vec![], vec![]);
        let gt_path = gt.map_or_else(|| dir.join("groundtruth.txt"), |p| p.to_path_buf());
        for l in std::fs::read_to_string(&gt_path).expect("groundtruth.txt").lines() {
            if l.starts_with('#') {
                continue;
            }
            let v: Vec<f64> = l.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            if v.len() == 8 {
                ts.push(v[0]);
                ps.push([v[1], v[2], v[3]]);
                qs.push([v[4], v[5], v[6], v[7]]);
            }
        }
        // 경로 평면 법선 = 공분산 최소 고유벡터, 부호는 카메라 아래(y)와 반대쪽이 위
        let n = ps.len() as f64;
        let mean = [0, 1, 2].map(|k| ps.iter().map(|p| p[k]).sum::<f64>() / n);
        let mut c = [[0.0; 3]; 3];
        for p in &ps {
            let d = sub(p, &mean);
            for i in 0..3 {
                for j in 0..3 {
                    c[i][j] += d[i] * d[j];
                }
            }
        }
        let (_, v) = eig_sym(&c);
        let mut up = [v[0][0], v[1][0], v[2][0]];
        let mut down = [0.0; 3];
        for q in &qs {
            let r = quat_to_m(*q);
            down = add(&down, &[r[0][1], r[1][1], r[2][1]]);
        }
        if dot(&up, &down) > 0.0 {
            up = up.map(|x| -x);
        }
        let level = rot_between(&up, &[0.0, 0.0, 1.0]);
        // 시퀀스별 카메라 보정(TUM 문서): freiburg1 · freiburg2 · freiburg3
        let name = dir.to_string_lossy();
        let k = if name.contains("freiburg1") { [517.3, 516.5, 318.6, 255.3] } else if name.contains("freiburg2") { [520.9, 521.0, 325.1, 249.7] } else { [535.4, 539.2, 320.1, 247.6] };
        Tum { dir: dir.to_path_buf(), rgb, depth, ts, ps, qs, level, k }
    }

    /// 시각 t 의 카메라 포즈(수평 세계): (위치, OpenCV 카메라 → 세계 회전)
    pub fn pose(&self, t: f64) -> Option<(V3, M3)> {
        if t < self.ts[0] || t > *self.ts.last()? {
            return None;
        }
        let (p, r) = interp_pose(&self.ts, &self.ps, &self.qs, t);
        Some((mv(&self.level, &p), mm(&self.level, &r)))
    }

    /// 시각 t 에 가장 가까운 Kinect 깊이(미터, 0 = 없음)
    pub fn kinect_depth(&self, t: f64) -> Option<(u32, u32, Vec<f32>)> {
        let i = self.depth.partition_point(|d| d.0 < t);
        let j = if i == 0 { 0 } else if i >= self.depth.len() { self.depth.len() - 1 } else if (self.depth[i].0 - t).abs() < (t - self.depth[i - 1].0).abs() { i } else { i - 1 };
        let img = image::open(self.dir.join(&self.depth[j].1)).ok()?.into_luma16();
        let (w, h) = img.dimensions();
        Some((w, h, img.into_raw().into_iter().map(|v| v as f32 / 5000.0).collect()))
    }
}
