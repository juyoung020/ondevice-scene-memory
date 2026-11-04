//! 작은 3D 수학: 3×3 행렬, 쿼터니언 → 회전, 대칭 행렬 고윳값(야코비), 포즈 보간.

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

pub fn mv(m: &M3, v: &V3) -> V3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub fn mm(a: &M3, b: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

pub fn add(a: &V3, b: &V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn sub(a: &V3, b: &V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn dot(a: &V3, b: &V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: &V3, b: &V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn norm(a: &V3) -> f64 {
    dot(a, a).sqrt()
}

/// 쿼터니언(x, y, z, w) → 회전 행렬
pub fn quat_to_m(q: [f64; 4]) -> M3 {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (x, y, z, w) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// 축 a 를 축 b 로 보내는 회전(로드리게스)
pub fn rot_between(a: &V3, b: &V3) -> M3 {
    let v = cross(a, b);
    let s = norm(&v);
    let c = dot(a, b);
    if s < 1e-12 {
        return if c > 0.0 { [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] } else { [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]] };
    }
    let k = [v[0] / s, v[1] / s, v[2] / s];
    let kx = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
    let kk = mm(&kx, &kx);
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = if i == j { 1.0 } else { 0.0 } + s * kx[i][j] + (1.0 - c) * kk[i][j];
        }
    }
    r
}

/// 대칭 3×3 의 고윳값·고유벡터(오름차순), 야코비 회전
pub fn eig_sym(a: &M3) -> ([f64; 3], M3) {
    let mut a = *a;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let (mut p, mut q, mut big) = (0, 1, 0.0);
        for i in 0..3 {
            for j in i + 1..3 {
                if a[i][j].abs() > big {
                    big = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if big < 1e-15 {
            break;
        }
        let th = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let (c, s) = (th.cos(), th.sin());
        for k in 0..3 {
            let (akp, akq) = (a[k][p], a[k][q]);
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
        }
        for k in 0..3 {
            let (apk, aqk) = (a[p][k], a[q][k]);
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
        for k in 0..3 {
            let (vkp, vkq) = (v[k][p], v[k][q]);
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| a[i][i].partial_cmp(&a[j][j]).unwrap());
    let w = [a[idx[0]][idx[0]], a[idx[1]][idx[1]], a[idx[2]][idx[2]]];
    let mut vs = [[0.0; 3]; 3];
    for (c, &i) in idx.iter().enumerate() {
        for r in 0..3 {
            vs[r][c] = v[r][i];
        }
    }
    (w, vs)
}

/// 시각 t 의 포즈: 위치는 선형, 회전은 쿼터니언 nlerp
pub fn interp_pose(ts: &[f64], ps: &[V3], qs: &[[f64; 4]], t: f64) -> (V3, M3) {
    let i = match ts.binary_search_by(|x| x.partial_cmp(&t).unwrap()) {
        Ok(i) => return (ps[i], quat_to_m(qs[i])),
        Err(i) => i.clamp(1, ts.len() - 1),
    };
    let a = ((t - ts[i - 1]) / (ts[i] - ts[i - 1])).clamp(0.0, 1.0);
    let p = [0, 1, 2].map(|k| ps[i - 1][k] * (1.0 - a) + ps[i][k] * a);
    let (q0, mut q1) = (qs[i - 1], qs[i]);
    if q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3] < 0.0 {
        q1 = q1.map(|x| -x);
    }
    let q = [0, 1, 2, 3].map(|k| q0[k] * (1.0 - a) + q1[k] * a);
    (p, quat_to_m(q))
}

/// 빠른 해시(정수 칸 열쇠용, rustc 의 Fx 방식): 기본 SipHash 는 칸 수십만 개를 장면마다 넣을 때 느리다
#[derive(Default, Clone, Copy)]
pub struct Fx(u64);
impl Fx {
    #[inline]
    fn add(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}
impl std::hash::Hasher for Fx {
    #[inline]
    fn finish(&self) -> u64 {
        // 마지막에 비트를 고루 섞음(곱셈만으로는 높은 비트에만 다른 열쇠가 같은 통에 몰린다)
        let mut h = self.0;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h
    }
    fn write(&mut self, b: &[u8]) {
        for &x in b {
            self.add(x as u64);
        }
    }
    #[inline]
    fn write_u64(&mut self, x: u64) {
        self.add(x);
    }
    #[inline]
    fn write_u32(&mut self, x: u32) {
        self.add(x as u64);
    }
    #[inline]
    fn write_i32(&mut self, x: i32) {
        self.add(x as u32 as u64);
    }
    #[inline]
    fn write_i64(&mut self, x: i64) {
        self.add(x as u64);
    }
    #[inline]
    fn write_usize(&mut self, x: usize) {
        self.add(x as u64);
    }
}
pub type FastMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<Fx>>;
pub type FastSet<K> = std::collections::HashSet<K, std::hash::BuildHasherDefault<Fx>>;

/// 회전 행렬 → 사원수 [x, y, z, w]
pub fn quat_of(m: &M3) -> [f64; 4] {
    let tr = m[0][0] + m[1][1] + m[2][2];
    let (x, y, z, w) = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s)
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        (0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s)
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        ((m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s)
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        ((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s)
    };
    [x, y, z, w]
}
