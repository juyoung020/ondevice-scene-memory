//! 장소 인식(루프 · 재위치 후보 찾기): MegaLoc(gmberton/MegaLoc, MIT, DINOv2-B + SALAD 집계)의 신경망 앞뒤를 Rust 로.
//! 신경망 본체(ViT-B/14 12 블록 + SALAD 의 점수 · 군집 · 토큰 MLP)는 NPU(.dxnn) 또는 PC ONNX, 여기는 그 앞뒤만.
//!
//! 1. 전처리 [`prepare`]: RGB 640×480 → 224×168(OpenCV INTER_AREA 와 같은 계산, [`resize_area`]) → 14×14 조각 192 개
//!    ([192, 588], 조각 안 순서 채널 · 세로 · 가로, 값 /255). 평균 · 표준편차 정규화와 조각 합성곱은 모델 첫 행렬곱에 접혀 있다.
//! 2. 신경망 출력: f [192,256](군집 특징) · s [192,64](군집 점수) · t [256](전역 토큰).
//! 3. [`Head::describe`]: 쓰레기 칸을 붙인 최적 수송(Sinkhorn, 로그 공간 3 회) → 군집별 가중합 → 군집마다 정규화
//!    → [t̂, 군집 16,384] 정규화 → 투영(원본 마지막 선형 층 16,640 → 8,448, 또는 그 위에 주성분 투영을 접은 것) → 정규화.
//! 4. [`PlaceDb`]: 키프레임 기술자를 연속 행렬로 두고 내적 상위 k(NEON).

pub const IN_W: usize = 224;
pub const IN_H: usize = 168;
pub const PS: usize = 14;
pub const GW: usize = IN_W / PS;
pub const GH: usize = IN_H / PS;
pub const NTOK: usize = GW * GH; // 192
pub const PATCH: usize = 3 * PS * PS; // 588
pub const NCL: usize = 64;
pub const CDIM: usize = 256;
pub const TDIM: usize = 256;
pub const AGG: usize = TDIM + NCL * CDIM; // 16,640

// ---------- OpenCV INTER_AREA(축소, 정수 배율이 아닐 때) ----------
struct Tab {
    di: usize,
    si: usize,
    alpha: f32,
}

/// OpenCV computeResizeAreaTab 그대로(f64 로 계산해 f32 무게)
fn area_tab(ssize: usize, dsize: usize, cn: usize, scale: f64) -> Vec<Tab> {
    let mut t = Vec::new();
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell = scale.min(ssize as f64 - fsx1);
        let mut sx1 = fsx1.ceil() as i64;
        let mut sx2 = fsx2.floor() as i64;
        sx2 = sx2.min(ssize as i64 - 1);
        sx1 = sx1.min(sx2);
        if sx1 as f64 - fsx1 > 1e-3 {
            t.push(Tab { di: dx * cn, si: (sx1 - 1) as usize * cn, alpha: ((sx1 as f64 - fsx1) / cell) as f32 });
        }
        for sx in sx1..sx2 {
            t.push(Tab { di: dx * cn, si: sx as usize * cn, alpha: (1.0 / cell) as f32 });
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            t.push(Tab { di: dx * cn, si: sx2 as usize * cn, alpha: ((fsx2 - sx2 as f64).min(1.0).min(cell) / cell) as f32 });
        }
    }
    t
}

/// u8 다채널 영상 축소(OpenCV resize INTER_AREA, 배율이 정수가 아닌 경로 ResizeArea_Invoker 의 f32 계산 순서)
pub fn resize_area(src: &[u8], sw: usize, sh: usize, cn: usize, dw: usize, dh: usize, out: &mut Vec<u8>) {
    assert!(dw <= sw && dh <= sh && src.len() == sw * sh * cn);
    let sx = 1.0 / (dw as f64 / sw as f64);
    let sy = 1.0 / (dh as f64 / sh as f64);
    let xtab = area_tab(sw, dw, cn, sx);
    let ytab = area_tab(sh, dh, 1, sy);
    let wd = dw * cn;
    out.resize(wd * dh, 0);
    let mut buf = vec![0f32; wd];
    let mut sum = vec![0f32; wd];
    let mut prev = ytab[0].di;
    let put = |out: &mut [u8], dy: usize, sum: &[f32]| {
        for (o, &v) in out[dy * wd..(dy + 1) * wd].iter_mut().zip(sum) {
            *o = v.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    };
    for y in &ytab {
        let row = &src[y.si * sw * cn..(y.si + 1) * sw * cn];
        buf.fill(0.0);
        for x in &xtab {
            for c in 0..cn {
                buf[x.di + c] += row[x.si + c] as f32 * x.alpha;
            }
        }
        if y.di != prev {
            put(out, prev, &sum);
            for (s, b) in sum.iter_mut().zip(&buf) {
                *s = y.alpha * b;
            }
            prev = y.di;
        } else {
            for (s, b) in sum.iter_mut().zip(&buf) {
                *s += y.alpha * b;
            }
        }
    }
    put(out, prev, &sum);
}

/// 224×168 RGB → 조각 [192, 588] f32(값 /255)
pub fn to_patches(small: &[u8], out: &mut Vec<f32>) {
    assert_eq!(small.len(), IN_W * IN_H * 3);
    out.resize(NTOK * PATCH, 0.0);
    for gy in 0..GH {
        for gx in 0..GW {
            let o = &mut out[(gy * GW + gx) * PATCH..(gy * GW + gx + 1) * PATCH];
            for c in 0..3 {
                for ky in 0..PS {
                    for kx in 0..PS {
                        o[c * PS * PS + ky * PS + kx] = small[((gy * PS + ky) * IN_W + gx * PS + kx) * 3 + c] as f32 / 255.0;
                    }
                }
            }
        }
    }
}

/// RGB(w×h) → 조각 입력. small 은 줄인 사진(재사용 버퍼)
pub fn prepare(rgb: &[u8], w: usize, h: usize, small: &mut Vec<u8>, patches: &mut Vec<f32>) {
    resize_area(rgb, w, h, 3, IN_W, IN_H, small);
    to_patches(small, patches);
}

// ---------- 내적(NEON) ----------
/// a·b (길이 같음). aarch64 는 NEON 4 칸 × 4 누적, 다른 기계는 같은 묶음의 스칼라
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    // SAFETY: i + 16 ≤ n 범위 안에서만 읽음
    let mut s = unsafe {
        use core::arch::aarch64::*;
        let (p, q) = (a.as_ptr(), b.as_ptr());
        let mut acc = [vdupq_n_f32(0.0); 4];
        while i + 16 <= n {
            for j in 0..4 {
                acc[j] = vfmaq_f32(acc[j], vld1q_f32(p.add(i + 4 * j)), vld1q_f32(q.add(i + 4 * j)));
            }
            i += 16;
        }
        vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])))
    };
    #[cfg(not(target_arch = "aarch64"))]
    let mut s = {
        let mut acc = [0f32; 16];
        while i + 16 <= n {
            for j in 0..16 {
                acc[j] += a[i + j] * b[i + j];
            }
            i += 16;
        }
        acc.iter().sum::<f32>()
    };
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

pub fn l2_normalize(v: &mut [f32]) {
    let n = dot(v, v).sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= n;
    }
}

// ---------- 집계 + 투영 ----------
/// 신경망 뒤 CPU 몫: 쓰레기 칸 값과 마지막 투영(dim × 16,640, f32)
pub struct Head {
    pub dust: f32,
    pub dim: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

/// 투영 파일(`megaloc_head.bin`): "MLH1", u32 입력(16,640), u32 출력 dim, f32 dust, 무게 dim×16,640 f32(행 우선), 편향 dim f32. 모두 리틀 엔디언
impl Head {
    pub fn load(path: &str) -> Result<Head, String> {
        let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        if b.len() < 16 || &b[..4] != b"MLH1" {
            return Err(format!("{path}: MLH1 아님"));
        }
        let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
        let (inp, dim) = (u(4), u(8));
        if inp != AGG {
            return Err(format!("{path}: 입력 {inp} ≠ {AGG}"));
        }
        let dust = f32::from_le_bytes(b[12..16].try_into().unwrap());
        let need = 16 + (dim * AGG + dim) * 4;
        if b.len() != need {
            return Err(format!("{path}: 크기 {} ≠ {need}", b.len()));
        }
        let f = |o: usize, n: usize| -> Vec<f32> { b[o..o + n * 4].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
        Ok(Head { dust, dim, w: f(16, dim * AGG), b: f(16 + dim * AGG * 4, dim) })
    }

    /// f [192,256] · s [192,64] · t [256] → 집계 벡터 x [16,640](MegaLoc FeatureAggregator 출력, 정규화까지). 계산은 f64
    pub fn aggregate(&self, f: &[f32], s: &[f32], t: &[f32], x: &mut Vec<f32>) {
        assert!(f.len() == NTOK * CDIM && s.len() == NTOK * NCL && t.len() == TDIM);
        let (n, m) = (NTOK, NCL);
        // S_aug [65, n]
        let mut sa = vec![0f64; (m + 1) * n];
        for k in 0..m {
            for j in 0..n {
                sa[k * n + j] = s[j * m + k] as f64;
            }
        }
        for j in 0..n {
            sa[m * n + j] = self.dust as f64;
        }
        let norm = -((n + m) as f64).ln();
        let mut la = vec![norm; m + 1];
        la[m] += ((n - m) as f64).ln();
        let lb = norm;
        let mut u = vec![0f64; m + 1];
        let mut v = vec![0f64; n];
        let mut tmp = vec![0f64; n.max(m + 1)];
        for _ in 0..3 {
            for k in 0..=m {
                for j in 0..n {
                    tmp[j] = sa[k * n + j] + v[j];
                }
                u[k] = la[k] - lse(&tmp[..n]);
            }
            for j in 0..n {
                for k in 0..=m {
                    tmp[k] = sa[k * n + j] + u[k];
                }
                v[j] = lb - lse(&tmp[..m + 1]);
            }
        }
        // P [64, n] = exp(S + u + v − norm), 군집별 가중합 agg[d][k] = Σ_j f[j][d] P[k][j]
        let mut p = vec![0f64; m * n];
        for k in 0..m {
            for j in 0..n {
                p[k * n + j] = (sa[k * n + j] + u[k] + v[j] - norm).exp();
            }
        }
        let mut agg = vec![0f64; CDIM * m];
        for j in 0..n {
            let fj = &f[j * CDIM..(j + 1) * CDIM];
            for (d, &fv) in fj.iter().enumerate() {
                let fv = fv as f64;
                let row = &mut agg[d * m..(d + 1) * m];
                for k in 0..m {
                    row[k] += fv * p[k * n + j];
                }
            }
        }
        for k in 0..m {
            let mut ss = 0f64;
            for d in 0..CDIM {
                ss += agg[d * m + k] * agg[d * m + k];
            }
            let nn = ss.sqrt().max(1e-12);
            for d in 0..CDIM {
                agg[d * m + k] /= nn;
            }
        }
        let tn = t.iter().map(|&a| (a as f64) * (a as f64)).sum::<f64>().sqrt().max(1e-12);
        let mut all = Vec::with_capacity(AGG);
        all.extend(t.iter().map(|&a| a as f64 / tn));
        all.extend_from_slice(&agg);
        let an = all.iter().map(|a| a * a).sum::<f64>().sqrt().max(1e-12);
        x.clear();
        x.extend(all.iter().map(|a| (a / an) as f32));
    }

    /// 집계 벡터 → 기술자(투영 + 정규화)
    pub fn project(&self, x: &[f32], out: &mut Vec<f32>) {
        out.resize(self.dim, 0.0);
        for (i, o) in out.iter_mut().enumerate() {
            *o = dot(&self.w[i * AGG..(i + 1) * AGG], x) + self.b[i];
        }
        l2_normalize(out);
    }

    pub fn describe(&self, f: &[f32], s: &[f32], t: &[f32], out: &mut Vec<f32>) {
        let mut x = Vec::with_capacity(AGG);
        self.aggregate(f, s, t, &mut x);
        self.project(&x, out);
    }
}

fn lse(a: &[f64]) -> f64 {
    let mx = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    mx + a.iter().map(|&v| (v - mx).exp()).sum::<f64>().ln()
}

// ---------- 키프레임 기술자 표 ----------
/// 키프레임 기술자(단위 길이)를 연속 행렬로: 노드 id 와 같은 순서
#[derive(Default)]
pub struct PlaceDb {
    pub dim: usize,
    pub ids: Vec<i64>,
    pub desc: Vec<f32>,
}
impl PlaceDb {
    pub fn new(dim: usize) -> PlaceDb {
        PlaceDb { dim, ..Default::default() }
    }
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    pub fn add(&mut self, id: i64, d: &[f32]) {
        assert_eq!(d.len(), self.dim);
        self.ids.push(id);
        self.desc.extend_from_slice(d);
    }
    /// 노드 지우기(지도 정리 때)
    pub fn remove(&mut self, id: i64) {
        if let Some(i) = self.ids.iter().position(|&x| x == id) {
            self.ids.swap_remove(i);
            let last = self.ids.len();
            if i != last {
                let (a, b) = self.desc.split_at_mut(last * self.dim);
                a[i * self.dim..(i + 1) * self.dim].copy_from_slice(&b[..self.dim]);
            }
            self.desc.truncate(last * self.dim);
        }
    }
    /// 질의와 내적이 큰 순 상위 k(keep(id) 가 참인 노드만: 예 최근 몇 초 노드 빼기). 같으면 앞 노드
    pub fn topk(&self, q: &[f32], k: usize, keep: impl Fn(i64) -> bool) -> Vec<(i64, f32)> {
        let mut best: Vec<(i64, f32)> = Vec::with_capacity(k + 1);
        for (i, &id) in self.ids.iter().enumerate() {
            if !keep(id) {
                continue;
            }
            let s = dot(&self.desc[i * self.dim..(i + 1) * self.dim], q);
            if best.len() < k || s > best[best.len() - 1].1 {
                let pos = best.iter().position(|b| s > b.1).unwrap_or(best.len());
                best.insert(pos, (id, s));
                best.truncate(k);
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_integer_scale_is_box_mean() {
        // 정수 배율(2×)이면 칸 평균과 같음(OpenCV 도 이 경로는 평균)
        let src: Vec<u8> = (0..8 * 4 * 3).map(|i| (i * 7 % 251) as u8).collect();
        let mut out = Vec::new();
        resize_area(&src, 8, 4, 3, 4, 2, &mut out);
        for dy in 0..2 {
            for dx in 0..4 {
                for c in 0..3 {
                    let mut s = 0u32;
                    for yy in 0..2 {
                        for xx in 0..2 {
                            s += src[((dy * 2 + yy) * 8 + dx * 2 + xx) * 3 + c] as u32;
                        }
                    }
                    let want = (s as f32 / 4.0).round_ties_even() as u8;
                    assert_eq!(out[(dy * 4 + dx) * 3 + c], want);
                }
            }
        }
    }

    #[test]
    fn area_tab_weights_sum_to_one() {
        let s = 1.0 / (224.0 / 640.0);
        let t = area_tab(640, 224, 1, s);
        for dx in 0..224 {
            let w: f32 = t.iter().filter(|e| e.di == dx).map(|e| e.alpha).sum();
            assert!((w - 1.0).abs() < 1e-5, "{dx} {w}");
        }
    }

    #[test]
    fn patches_order() {
        let mut small = vec![0u8; IN_W * IN_H * 3];
        // 조각 (gy 1, gx 2) 의 ky 3, kx 4, 채널 2
        small[((14 + 3) * IN_W + 28 + 4) * 3 + 2] = 255;
        let mut p = Vec::new();
        to_patches(&small, &mut p);
        let n = GW + 2;
        assert_eq!(p[n * PATCH + 2 * 196 + 3 * 14 + 4], 1.0);
        assert_eq!(p.iter().filter(|&&v| v != 0.0).count(), 1);
    }

    #[test]
    fn aggregate_is_unit_and_uniform_scores_spread() {
        let head = Head { dust: 1.0, dim: 0, w: vec![], b: vec![] };
        let f: Vec<f32> = (0..NTOK * CDIM).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        let s = vec![0.0f32; NTOK * NCL];
        let t: Vec<f32> = (0..TDIM).map(|i| i as f32).collect();
        let mut x = Vec::new();
        head.aggregate(&f, &s, &t, &mut x);
        assert_eq!(x.len(), AGG);
        let n = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
        assert!((n - 1.0).abs() < 1e-4, "norm {n} nan {}", x.iter().filter(|v| !v.is_finite()).count());
        // 점수가 모두 같으면 군집마다 같은 가중합 → 같은 정규화 열
        assert!((x[TDIM] - x[TDIM + 1]).abs() < 1e-6);
    }

    #[test]
    fn db_topk() {
        let mut db = PlaceDb::new(3);
        db.add(10, &[1.0, 0.0, 0.0]);
        db.add(11, &[0.0, 1.0, 0.0]);
        db.add(12, &[0.7071, 0.7071, 0.0]);
        let r = db.topk(&[1.0, 0.1, 0.0], 2, |_| true);
        assert_eq!(r.iter().map(|x| x.0).collect::<Vec<_>>(), vec![10, 12]);
        let r = db.topk(&[1.0, 0.1, 0.0], 2, |id| id != 10);
        assert_eq!(r[0].0, 12);
        db.remove(10);
        assert_eq!(db.ids, vec![12, 11]);
        assert_eq!(&db.desc[..3], &[0.7071, 0.7071, 0.0]);
    }
}
