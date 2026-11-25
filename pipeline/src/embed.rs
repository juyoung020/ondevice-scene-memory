//! 물체 사진 → 검색용 벡터. SigLIP2 영상 인코더(PC: vision_model.onnx, 보드 NPU: .dxnn)가 사진 한 장을 768 차원 벡터로 바꾸고,
//! 길이를 1 로 맞춰 `objects/O<id>_emb.bin`(f32 리틀 엔디언 768 개)에 남긴다. 뷰어의 검색이 글 벡터와 내적해 닮은 물체를 찾는다.
//! 지도 쌓기와 따로 도는 스레드 하나가 맡는다: 사진이 바뀐 물체만 받아, 밀리면 물체마다 가장 새 사진만 계산한다.
use std::path::{Path, PathBuf};

/// 인코더 입력 한 변(화소)
pub const SIZE: usize = 256;
/// 벡터 차원
pub const DIM: usize = 768;

/// w×h RGB 를 256×256 으로 늘려(작아질 땐 넓게 평균) NCHW, 값 −1..1 (SigLIP 전처리: (x/255 − 0.5)/0.5)
#[cfg_attr(not(test), allow(dead_code))]
pub fn preprocess(rgb: &[u8], w: u32, h: u32, out: &mut Vec<f32>) {
    preprocess_into(rgb, w, h, false, out)
}

/// 사진(RGB w×h) → 256×256 쌍선형으로 줄여 (값 / 127.5 − 1) 을 out 에 바로 쓴다(중간 사진 없음).
/// patches = false: [3, 256, 256] (채널, 행, 열). true: 32×32 조각 64 개 [64, 3·32·32] (조각 = 행 우선, 안은 채널·행·열) —
/// 조각 나누기 합성곱을 행렬곱으로 바꿔 NPU 에 다 올린 모델(입력 `patches`)이 받는 배치
pub fn preprocess_into(rgb: &[u8], w: u32, h: u32, patches: bool, out: &mut Vec<f32>) {
    let (w, h) = (w as usize, h as usize);
    let n = SIZE * SIZE;
    out.resize(3 * n, 0.0);
    let axis = |dst: usize, src: usize| -> (usize, usize, f32) {
        let f = ((dst as f32 + 0.5) * src as f32 / SIZE as f32 - 0.5).max(0.0);
        let a = (f as usize).min(src - 1);
        (a, (a + 1).min(src - 1), f - a as f32)
    };
    // 가로 위치·가중(열마다 한 번). 한 행의 가로 섞기를 먼저 채널별 배열로 만든 뒤 세로 섞기를 32 칸씩 이어서 쓴다
    // (곱·더하기 순서는 화소마다 하던 식과 같아 값이 비트까지 같다. 세로 섞기 고리는 컴파일러가 NEON 으로 묶음)
    let (mut ix0, mut ix1, mut wx0, mut wx1) = ([0usize; SIZE], [0usize; SIZE], [0f32; SIZE], [0f32; SIZE]);
    for x in 0..SIZE {
        let (a, b, f) = axis(x, w);
        (ix0[x], ix1[x], wx0[x], wx1[x]) = (a * 3, b * 3, 1.0 - f, f);
    }
    const K: f32 = 1.0 / 127.5;
    let (mut top, mut bot) = ([0f32; SIZE], [0f32; SIZE]);
    for y in 0..SIZE {
        let (y0, y1, fy) = axis(y, h);
        let (r0, r1) = (&rgb[y0 * w * 3..(y0 + 1) * w * 3], &rgb[y1 * w * 3..(y1 + 1) * w * 3]);
        let (wy0, wy1) = (1.0 - fy, fy);
        for c in 0..3 {
            for x in 0..SIZE {
                top[x] = r0[ix0[x] + c] as f32 * wx0[x] + r0[ix1[x] + c] as f32 * wx1[x];
                bot[x] = r1[ix0[x] + c] as f32 * wx0[x] + r1[ix1[x] + c] as f32 * wx1[x];
            }
            for px in 0..SIZE / 32 {
                let o = if patches { ((y / 32) * 8 + px) * 3072 + c * 1024 + (y % 32) * 32 } else { c * n + y * SIZE + px * 32 };
                let (d, t, b) = (&mut out[o..o + 32], &top[px * 32..px * 32 + 32], &bot[px * 32..px * 32 + 32]);
                for k in 0..32 {
                    d[k] = (t[k] * wy0 + b[k] * wy1) * K - 1.0;
                }
            }
        }
    }
}

/// 길이 1 로 맞춤(0 벡터는 그대로)
pub fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// 물체 번호 → 지금까지 계산한 사진 벡터의 합(이름 고르기용)
static SUMS: std::sync::Mutex<Option<std::collections::HashMap<u32, Vec<f32>>>> = std::sync::Mutex::new(None);
/// 물체 번호 → 사진 전체 벡터의 합(검색 줄 · 병합)
static VSUMS: std::sync::Mutex<Option<std::collections::HashMap<u32, Vec<f32>>>> = std::sync::Mutex::new(None);

/// 물체 번호 → 그 물체 사진 벡터들의 평균 방향(길이 1) · 바뀐 물체 기록((순번, 물체 번호), 순번은 차례로 늘고 리셋해도 이어 감).
/// 계산 스레드가 채우고 물체 병합(maps::Memory::merge_pass)이 지난번 뒤에 바뀐 것만 가져간다
type Fresh = (std::collections::HashMap<u32, std::sync::Arc<[f32]>>, Vec<(u64, u32)>);
static MEANS: std::sync::Mutex<Option<Fresh>> = std::sync::Mutex::new(None);
/// 마지막 순번(바뀐 것이 없으면 잠그지 않고 끝내려고 따로 둠)
static GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 순번 since 뒤에 평균 벡터가 바뀐 물체와 그 벡터(물체마다 하나, 나눠 쓰는 Arc). 반환: (지금 순번, 목록)
pub fn fresh_since(since: u64) -> (u64, Vec<(u32, std::sync::Arc<[f32]>)>) {
    if GEN.load(std::sync::atomic::Ordering::Acquire) == since {
        return (since, vec![]);
    }
    let g = MEANS.lock().unwrap();
    let now = GEN.load(std::sync::atomic::Ordering::Acquire);
    let Some((means, log)) = g.as_ref() else { return (now, vec![]) };
    let from = log.partition_point(|x| x.0 <= since);
    let mut ids: Vec<u32> = log[from..].iter().map(|x| x.1).collect();
    ids.sort_unstable();
    ids.dedup();
    (now, ids.into_iter().filter_map(|id| Some((id, means.get(&id)?.clone()))).collect())
}

/// SigLIP2 base patch32-256: 로짓 = SCALE · 코사인 (체크포인트의 logit_scale = e^4.717)
const SCALE: f32 = 111.83;
/// 사진으로 붙인 이름을 쓰는 가장 낮은 몫. 사무실 녹화의 물체 33 개에서 0.3 이상은 사진과 맞았고(오리 0.46, 지우개 0.50, 스프레이 통 0.35)
/// 그 아래는 틀린 이름이 섞였다(컵 → bedpan 0.27)
pub const NAME_MIN: f32 = 0.3;
/// 이름 없는 물체(object)에 사진 이름을 쓰는 가장 낮은 몫. ObjectSAM v1.0 으로 돌린 사무실 녹화에서 0.15 아래는 대부분 틀렸다
/// (breechcloth 0.06, ping-pong ball 0.08, vending machine 0.10), 0.2 대는 맞았다(chair 0.27, bottle 0.23, file cabinet 0.20)
pub const NAME_MIN_UNNAMED: f32 = 0.15;
/// 1 등 이름과 2 등 이름의 코사인 차가 이보다 작으면 이름을 붙이지 않는다. fr1_desk 물체 23 개(사람이 이름을 단 것)에서
/// 몫 0.15 만 쓸 때 맞은 이름 비율 0.52, 차 0.003 을 더하면 0.67(이름 붙은 물체 15 개). 마스크 안 토큰 이름(maskpool)으로 바꾼 뒤
/// 강의실 녹화 두 구간(본 횟수 ≥ 12)에서 0.003 → 0.01: 이름 붙은 물체 47 → 30, 마스크 기준 맞음 70 → 80 %(docs/pose_experiments.md 병렬 C 이어서 5)
pub const NAME_MARGIN: f32 = 0.01;

/// 구조물 낱말(가장 닮은 구조물 표 낱말이 이 넷 중 하나면 벽 · 바닥 · 천장 면)
pub const STRUCTURAL: [&str; 4] = ["floor", "wall", "ceiling", "baseboard"];
/// SigLIP2 의 글 · 사진 한 쌍 확률 p = sigmoid(SCALE · cos + LOGIT_BIAS) (체크포인트 logit_bias)
pub const LOGIT_BIAS: f32 = -16.766876;
/// 구조물로 보는 가장 낮은 p(docs/search_direct.md "구조물 판정용 최소 단어": 바닥 · 벽 조각 0.87 맞춤, 물체 59 개 중 0 개를 구조물로)
pub const STRUCT_MIN_P: f32 = 0.003;

/// 이름 목록과 그 글 벡터(`--vocab`): PC 에서 `sgview --encode-vocab` 로 한 번 만들어 둔 파일이라 글 인코더 없이 쓴다
static VOCAB: std::sync::OnceLock<(Vec<String>, Vec<f32>)> = std::sync::OnceLock::new();
/// 구조물 판정 표(`--structural`, 기본 vocab/structural.voc: floor wall ceiling baseboard object furniture window door)
static STRUCT_VOCAB: std::sync::OnceLock<(Vec<String>, Vec<f32>)> = std::sync::OnceLock::new();
/// 물체 번호 → 사진이 가장 닮은 이름과 그 몫
static NAMES: std::sync::Mutex<Option<std::collections::HashMap<u32, (String, f32)>>> = std::sync::Mutex::new(None);
/// 물체 번호 → 구조물 낱말(벽 · 바닥 · 천장 면이면)
static STRUCTS: std::sync::Mutex<Option<std::collections::HashMap<u32, &'static str>>> = std::sync::Mutex::new(None);

/// 글 벡터 파일(첫 줄 `SGVOC 1 <n> <차원>`, 이름 n 줄, 이어서 f32 n·차원 개): (이름, 벡터)
fn read_voc(path: &str, opt: &str) -> (Vec<String>, Vec<f32>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{opt} {path}: {e}"));
    let mut pos = 0;
    let mut line = || {
        let end = pos + b[pos..].iter().position(|c| *c == b'\n').unwrap_or_else(|| panic!("{opt} {path}: 글 벡터 파일이 아님"));
        let s = std::str::from_utf8(&b[pos..end]).unwrap_or_else(|_| panic!("{opt} {path}: 글 벡터 파일이 아님")).to_string();
        pos = end + 1;
        s
    };
    let head = line();
    let h: Vec<&str> = head.split(' ').collect();
    assert!(h.len() == 4 && h[0] == "SGVOC" && h[1] == "1" && h[3].parse() == Ok(DIM), "{opt} {path}: 글 벡터 파일이 아님");
    let n: usize = h[2].parse().unwrap_or_else(|_| panic!("{opt} {path}: 글 벡터 파일이 아님"));
    let names: Vec<String> = (0..n).map(|_| line()).collect();
    assert_eq!(b.len(), pos + n * DIM * 4, "{opt} {path}: 벡터 수가 이름 수와 다름");
    (names, b[pos..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// 지도 이름표용 이름 목록(`--vocab`)을 읽어 둔다. 이름 목록을 돌려준다
pub fn load_vocab(path: &str) -> Vec<String> {
    let (names, v) = read_voc(path, "--vocab");
    let _ = VOCAB.set((names.clone(), v));
    names
}

/// 구조물 판정 표(`--structural`)를 읽어 둔다
pub fn load_structural(path: &str) {
    let _ = STRUCT_VOCAB.set(read_voc(path, "--structural"));
}

/// 구조물 판정: 표에서 가장 닮은 낱말이 STRUCTURAL 중 하나이고 p ≥ STRUCT_MIN_P 이면 그 낱말
fn look_structure(names: &[String], vecs: &[f32], e: &[f32]) -> Option<&'static str> {
    let (i, c) = vecs.chunks(DIM).map(|t| t.iter().zip(e).map(|(a, b)| a * b).sum::<f32>()).enumerate().max_by(|a, b| a.1.total_cmp(&b.1))?;
    let p = 1.0 / (1.0 + (-(SCALE * c + LOGIT_BIAS)).exp());
    STRUCTURAL.iter().find(|s| **s == names[i]).copied().filter(|_| p >= STRUCT_MIN_P)
}

/// 그 물체가 벽 · 바닥 · 천장 면이면 그 낱말(`--structural` 표, 사진 벡터 평균으로 판정). 표가 없거나 벡터 계산 전이면 None
pub fn structure(id: u32) -> Option<&'static str> {
    common::prof::lock(&STRUCTS, common::prof::W::LockEmbed).as_ref()?.get(&id).copied()
}

/// 사진 벡터(길이 1)가 가장 닮은 이름과 그 몫: 이름 전체에 대한 softmax(SCALE · 코사인). 1 등과 2 등의 코사인 차가 NAME_MARGIN 보다
/// 작으면(두 이름 사이에서 갈팡질팡) 몫을 0 으로 돌려 이름을 붙이지 않게 한다
fn look_name(names: &[String], vecs: &[f32], e: &[f32]) -> Option<(String, f32)> {
    let l: Vec<f32> = vecs.chunks(DIM).map(|t| SCALE * t.iter().zip(e).map(|(a, b)| a * b).sum::<f32>()).collect();
    let (i, m) = l.iter().cloned().enumerate().max_by(|a, b| a.1.total_cmp(&b.1))?;
    let second = l.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, x)| *x).fold(f32::NEG_INFINITY, f32::max);
    let z: f32 = l.iter().map(|x| (x - m).exp()).sum();
    let share = if (m - second) / SCALE >= NAME_MARGIN { 1.0 / z } else { 0.0 };
    Some((names[i].clone(), share))
}

/// 그 물체의 사진으로 붙인 이름. 분할 모델이 이름을 준 물체(`sure` 참)는 몫 NAME_MIN 이상일 때만 바꾸고,
/// 이름 없는 물체(object)는 몫 NAME_MIN_UNNAMED 이상이면 가장 닮은 이름을 쓴다. `--vocab` 이 없거나 벡터 계산 전이면 None
pub fn name(id: u32, _sure: bool) -> Option<(String, f32)> {
    // 1 · 2 등이 NAME_MARGIN 안으로 붙은(몫 0) 물체는 이름을 붙이지 않음: 강의실 녹화에서 이런 물체는 번진 · 조각 사진이라
    // 이름이 대부분 틀렸다(sofa · shoe · paper towel 등, docs/pose_experiments.md 병렬 C)
    common::prof::lock(&NAMES, common::prof::W::LockEmbed).as_ref()?.get(&id).filter(|n| n.1 > 0.0).cloned()
}
/// 화면 · 기록에 쓰는 이름: NAME_MIN_OBS 번 넘게 본 물체만(그 아래는 번진 · 조각 사진이라 강의실 녹화 두 구간에서 이름이 대부분 틀렸다,
/// docs/pose_experiments.md 병렬 C 이어서 3). 물체 지우기 판정(maps)은 본 횟수와 상관없이 name 을 본다
pub fn shown_name(id: u32, n_obs: u32) -> Option<(String, f32)> {
    if n_obs < NAME_MIN_OBS {
        return None;
    }
    name(id, true)
}
pub const NAME_MIN_OBS: u32 = 12;
#[cfg_attr(not(any(feature = "onnx", feature = "dxrt")), allow(dead_code))]
/// 벡터 하나를 그 물체에 더함. v = 사진 전체 벡터(검색 줄 · 병합 · 구조물 판정), nv = 마스크 안 토큰 벡터(이름, maskpool).
/// 돌려줌: 지금까지 사진 전체 벡터의 평균 방향(길이 1)
fn put(id: u32, v: Vec<f32>, nv: Vec<f32>) -> Vec<f32> {
    // 이름은 그 물체의 마스크 안 벡터를 모두 더한 평균 방향으로(사진 한 장의 흔들림·어두움에 덜 휘둘림)
    let name_mean = {
        let mut sums = SUMS.lock().unwrap();
        let s = sums.get_or_insert_with(Default::default).entry(id).or_insert_with(|| vec![0.0; nv.len()]);
        s.iter_mut().zip(&nv).for_each(|(a, b)| *a += b);
        *NCOUNT.lock().unwrap().get_or_insert_with(Default::default).entry(id).or_insert(0) += 1;
        let mut m = s.clone();
        normalize(&mut m);
        m
    };
    let mean = {
        let mut sums = VSUMS.lock().unwrap();
        let s = sums.get_or_insert_with(Default::default).entry(id).or_insert_with(|| vec![0.0; v.len()]);
        s.iter_mut().zip(&v).for_each(|(a, b)| *a += b);
        let mut m = s.clone();
        normalize(&mut m);
        m
    };
    if let Some((names, vecs)) = VOCAB.get() {
        if let Some(n) = look_name(names, vecs, &name_mean) {
            common::prof::lock(&NAMES, common::prof::W::LockEmbed).get_or_insert_with(Default::default).insert(id, n);
        }
    }
    if let Some((names, vecs)) = STRUCT_VOCAB.get() {
        let mut g = common::prof::lock(&STRUCTS, common::prof::W::LockEmbed);
        let m = g.get_or_insert_with(Default::default);
        match look_structure(names, vecs, &mean) {
            Some(s) => m.insert(id, s),
            None => m.remove(&id),
        };
    }
    // 병합도 같은 평균 방향을 본다(한 장의 흔들림에 덜 휘둘림)
    let mut g = common::prof::lock(&MEANS, common::prof::W::LockEmbed);
    let (means, log) = g.get_or_insert_with(Default::default);
    means.insert(id, mean.as_slice().into());
    let s = GEN.load(std::sync::atomic::Ordering::Acquire) + 1; // 리셋해도 이어 감(기억이 가진 순번이 거꾸로 가지 않게)
    log.push((s, id));
    GEN.store(s, std::sync::atomic::Ordering::Release);
    drop(g);
    mean
}

/// 리셋: 물체 id 가 처음부터 다시 쓰이므로 id 별 이름·벡터 합을 비운다
pub fn clear() {
    *SUMS.lock().unwrap() = None;
    *VSUMS.lock().unwrap() = None;
    *MEANS.lock().unwrap() = None;
    *NAMES.lock().unwrap() = None;
    *STRUCTS.lock().unwrap() = None;
    *NCOUNT.lock().unwrap() = None;
}

/// 이름 고르기 확인(ONDEVICE_NAME_DEBUG): 그 물체의 평균 벡터에서 상위 4 이름 · 코사인 · 사진 수
pub fn debug_name(id: u32) {
    let Some((names, vecs)) = VOCAB.get() else { return };
    let Some(mut m) = SUMS.lock().unwrap().as_ref().and_then(|s| s.get(&id).cloned()) else { return };
    let n = NCOUNT.lock().unwrap().as_ref().and_then(|c| c.get(&id).cloned()).unwrap_or(0);
    normalize(&mut m);
    let mut l: Vec<(f32, &str)> = vecs.chunks(DIM).zip(names).map(|(t, nm)| (t.iter().zip(&m).map(|(a, b)| a * b).sum::<f32>(), nm.as_str())).collect();
    l.sort_by(|a, b| b.0.total_cmp(&a.0));
    let cur = NAMES.lock().unwrap().as_ref().and_then(|x| x.get(&id).cloned());
    println!("namedbg O{id} n{n} cur {:?} top {}", cur, l.iter().take(10).map(|(c, nm)| format!("{nm}:{c:.4}")).collect::<Vec<_>>().join(" "));
}
static NCOUNT: std::sync::Mutex<Option<std::collections::HashMap<u32, u32>>> = std::sync::Mutex::new(None);

/// 벡터 파일 경로(기억 폴더 기준 상대 경로)
pub fn rel_path(id: u32) -> String {
    format!("objects/O{id}_emb.bin")
}

#[cfg(any(feature = "onnx", feature = "dxrt"))]
mod worker {
    use super::*;
    use std::sync::Mutex;

    struct Job {
        id: u32,
        path: PathBuf,
        w: u32,
        h: u32,
        rgb: std::sync::Arc<Vec<u8>>, // 물체 사진(기억의 사진과 나눠 씀)
        tw: [f32; crate::maskpool::TOKENS], // 토큰마다 마스크가 덮은 몫(이름 벡터용)
        n: u64,                       // 맡긴 차례(결과를 늘 같게 하는 재생은 이 차례대로 반영)
    }
    /// 사진 벡터 줄(작업 풀에서 차례대로 도는 일): 전처리 · NPU 올리기 · 끝난 결과 정리. 스레드 없음(NPU 끝 알림이 이 줄에 일을 넣음)
    static LANE: Mutex<Option<common::rt::SharedLane<St>>> = Mutex::new(None);
    /// 맡겼지만 아직 끝나지 않은 사진 수
    static PENDING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    /// 결과를 늘 같게(재생): 끝난 벡터를 바로 반영하지 않고 쌓아 두었다가, 맡긴 회차에서 DET_LAG 회차 뒤 기록 회차 앞(`sync_round`)에 맡긴 차례대로 반영.
    /// 그 사이 회차에 같은 물체의 더 새 사진이 맡겨졌으면 앞 사진은 건너뜀(반영 규칙이 회차 번호로만 정해져 실행마다 같음).
    /// 루프는 묶음이 DET_LAG 회차 넘게 밀릴 때만 기다린다(사진 벡터 계산이 장면 처리와 겹쳐 돎)
    #[derive(Default)]
    struct Det {
        /// 끝난 벡터: (차례, 물체, 파일, 벡터)
        staged: Vec<(u64, u32, PathBuf, Vec<f32>, Vec<f32>)>,
        /// 맡겼고 아직 반영 · 건너뜀을 정하지 않은 사진: (차례, 회차, 물체)
        log: Vec<(u64, u64, u32)>,
        /// 건너뛸 차례(줄이 아직 시작 전이면 계산하지 않음)
        cancel: std::collections::HashSet<u64>,
    }
    static DETS: Mutex<Option<Det>> = Mutex::new(None);
    static DET_CV: std::sync::Condvar = std::sync::Condvar::new();
    static ROUND: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    /// 반영을 미루는 회차 수
    const DET_LAG: u64 = 4;
    static DET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    static NEXT_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    /// 맡긴 사진이 다 끝났다는 알림(drain 이 잠들어 기다림)
    static DRAINED: (Mutex<()>, std::sync::Condvar) = (Mutex::new(()), std::sync::Condvar::new());
    fn dec_pending() {
        if PENDING.fetch_sub(1, std::sync::atomic::Ordering::SeqCst) == 1 {
            let _g = DRAINED.0.lock().unwrap();
            DRAINED.1.notify_all();
        }
    }

    /// NPU 에 함께 올리는 사진 수(`--embed-jobs`, 기본 3 = DX-M1 NPU 코어 수). 밀린 사진을 이만큼 한꺼번에 올려 올린 순서대로 받는다
    pub static JOBS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(3);

    /// 인코더 실행기: 입력 NCHW f32(−1..1) 256×256 → 768 차원
    enum Net {
        #[cfg(feature = "onnx")]
        Onnx(ort::session::Session),
        /// DX-M1: 입력·출력 모두 f32(조각 배치 입력, NPU 작업 하나)
        #[cfg(feature = "dxrt")]
        Dx(crate::seg::dx::Engine),
    }
    impl Net {
        fn open(model: &str) -> Net {
            #[cfg(feature = "dxrt")]
            if model.ends_with(".dxnn") {
                // 함께 올릴 수만큼 DXRT 입출력 버퍼를 잡는다(모자라면 올리기가 DXRT 안에서 기다림)
                let jobs = JOBS.load(std::sync::atomic::Ordering::Relaxed).max(1) as i32;
                let e = crate::seg::dx::Engine::try_open(model, crate::seg::dx::EMBED_BOUND.load(std::sync::atomic::Ordering::Relaxed),
                    crate::seg::dx::BUFFERS.load(std::sync::atomic::Ordering::Relaxed).max(jobs)).unwrap_or_else(|e| panic!("{e}"));
                assert_eq!(e.out_len, DIM * 4, "embed model output");
                common::info!("embed: NPU 모델 {model}, 출력 {}, 함께 올림 {jobs}", e.tensors[0].0);
                return Net::Dx(e);
            }
            #[cfg(feature = "onnx")]
            {
                // 지도 쌓기·분할과 코어를 다투지 않게 스레드 둘만
                let sess = ort::session::Session::builder(common::ort::env()).unwrap().with_intra_threads(2).unwrap()
                    .commit_from_file(model).unwrap_or_else(|e| panic!("embed model {model}: {e}"));
                Net::Onnx(sess)
            }
            #[cfg(not(feature = "onnx"))]
            panic!("--embed {model}: 이 빌드는 NPU 모델(.dxnn)을 받는다");
        }
        /// 입력이 조각 배치(`patches`)인 NPU 모델인가
        fn patches(&self) -> bool {
            match self {
                #[cfg(feature = "dxrt")]
                Net::Dx(e) => e.input == "patches",
                #[allow(unreachable_patterns)]
                _ => false,
            }
        }
        /// 끝 알림을 단다(NPU 만: `--npu-wake thread` = 기다림 스레드 npu-wait-emb, `callback` = DXRT 콜백)
        #[allow(unused_variables)]
        fn on_done(&mut self, notify: crate::seg::Notify) {
            #[cfg(feature = "dxrt")]
            if let Net::Dx(e) = self {
                e.on_done(crate::seg::dx::wake(), "emb", notify);
            }
        }
        /// 한 장 올리기(표 = 버퍼 자리): NPU 는 결과가 out 에 바로 쓰이고 끝나면 알림. CPU(ONNX)는 여기서 계산해 out 에 둠(None)
        #[allow(unused_variables)]
        fn submit(&mut self, buf: &[f32], out: &mut Vec<f32>, tag: u64) -> Option<i32> {
            match self {
                #[cfg(feature = "dxrt")]
                Net::Dx(e) => {
                    let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4) };
                    Some(e.submit_tagged(bytes, out, tag))
                }
                #[allow(unreachable_patterns)]
                _ => {
                    self.run(buf, out);
                    None
                }
            }
        }
        /// 끝 알림을 받은 작업의 자리를 놓음(콜백 길만, 이미 끝나 바로 돌아옴)
        #[allow(unused_variables)]
        fn release(&mut self, id: Option<i32>) {
            #[cfg(feature = "dxrt")]
            if let (Net::Dx(e), Some(id)) = (self, id) {
                if e.callback_mode() {
                    e.waiter().wait(id);
                }
            }
        }
        fn run(&mut self, buf: &[f32], v: &mut Vec<f32>) {
            match self {
                #[cfg(feature = "onnx")]
                Net::Onnx(sess) => {
                    let inp = ort::value::TensorRef::from_array_view(([1usize, 3, SIZE, SIZE], buf)).unwrap();
                    let out = sess.run(ort::inputs![inp]).expect("embed run");
                    let (_, o) = out["pooler_output"].try_extract_tensor::<f32>().unwrap();
                    assert_eq!(o.len(), DIM, "embed model output");
                    v.clear();
                    v.extend_from_slice(o);
                    // 마스크 안 이름 벡터(maskpool)용 토큰: 뒤에 이어 붙임
                    if let Some(t) = out.get("last_hidden_state") {
                        v.extend_from_slice(t.try_extract_tensor::<f32>().unwrap().1);
                    }
                }
                #[cfg(feature = "dxrt")]
                Net::Dx(e) => {
                    let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4) };
                    e.run_into(bytes, v);
                }
            }
        }
    }

    #[derive(Default)]
    struct Stat {
        n: u32,
        pre: f64,
        net: f64,
        write: f64,
        batches: u32,
        max_batch: usize,
    }
    /// 올린 사진 하나: 일, 작업 번호(NPU), 올린 때, 전처리 시간
    struct Flight {
        job: Job,
        id: Option<i32>,
        t0: std::time::Instant,
        pre: f64,
    }
    /// 사진 벡터 줄의 상태
    struct St {
        net: Option<Net>,
        /// 마스크 안 토큰으로 이름 벡터를 만드는 머리(ONNX 모델만, NPU 모델은 토큰을 내지 않아 사진 전체 벡터로 이름)
        head: Option<crate::maskpool::Head>,
        patches: bool,
        /// 입력 · 출력 버퍼 k 벌(올린 사진마다 한 벌, 받으면 그 자리를 다음 사진이 씀)
        ins: Vec<Vec<f32>>,
        outs: Vec<Vec<f32>>,
        /// 자리마다 올린 사진(None = 빈 자리)
        fly: Vec<Option<Flight>>,
        queue: std::collections::VecDeque<Job>,
        st: Stat,
        /// 열 보호로 늦출 때: 지난 묶음 올린 때, 늦춤을 무시하고 올림(끝 · 회차 맞추기)
        last_pump: std::time::Instant,
        force: bool,
        /// CPU 실행기(올리면 그 자리에서 계산)
        cpu: bool,
    }
    impl St {
        /// 빈 자리만큼 밀린 사진을 전처리(여럿이면 풀에서 함께)해 올린다. CPU(ONNX)는 그 자리에서 계산까지 끝냄
        fn pump(&mut self) {
            // 비상 열 보호 2 단: 묶음 사이를 THROTTLE_GAP 넘게 띄움(밀린 사진은 물체마다 가장 새 것만 남아 기다림)
            let det = DET.load(std::sync::atomic::Ordering::Relaxed);
            if THROTTLE.load(std::sync::atomic::Ordering::Relaxed) && !det && !self.force && self.last_pump.elapsed() < THROTTLE_GAP {
                return;
            }
            self.last_pump = std::time::Instant::now();
            loop {
                if det {
                    // 건너뛰기로 정해진 사진은 계산하지 않음
                    let g = DETS.lock().unwrap();
                    if let Some(d) = g.as_ref() {
                        self.queue.retain(|j| {
                            let skip = d.cancel.contains(&j.n);
                            if skip {
                                dec_pending();
                            }
                            !skip
                        });
                    }
                }
                // CPU 실행기는 한 장씩(그 사이 건너뛰기 표시를 다시 봄)
                let cap = if self.cpu { 1 } else { self.queue.len() };
                let free: Vec<usize> = (0..self.fly.len()).filter(|i| self.fly[*i].is_none()).take(self.queue.len().min(cap)).collect();
                if free.is_empty() {
                    return;
                }
                let jobs: Vec<Job> = self.queue.drain(..free.len()).collect();
                let t = std::time::Instant::now();
                {
                    use rayon::prelude::*;
                    let patches = self.patches;
                    let mut bufs: Vec<(usize, Vec<f32>)> = free.iter().map(|&i| (i, std::mem::take(&mut self.ins[i]))).collect();
                    bufs.par_iter_mut().zip(&jobs).for_each(|((_, b), j)| preprocess_into(&j.rgb, j.w, j.h, patches, b));
                    for (i, b) in bufs {
                        self.ins[i] = b;
                    }
                }
                let pre = t.elapsed().as_secs_f64() * 1e3 / jobs.len() as f64;
                for (slot, j) in free.into_iter().zip(jobs) {
                    common::prof::copy(common::prof::C::EmbedIn, self.ins[slot].len() * 4);
                    let id = self.net.as_mut().expect("embed 모델").submit(&self.ins[slot], &mut self.outs[slot], slot as u64);
                    self.fly[slot] = Some(Flight { job: j, id, t0: t, pre });
                    if id.is_none() {
                        self.done(slot); // CPU: 이미 계산함
                    }
                }
            }
        }
        /// 자리 slot 의 사진이 끝남: 정규화 · 반영(또는 재생이면 쌓아 둠)
        fn done(&mut self, slot: usize) {
            let Some(f) = self.fly[slot].take() else { return };
            if let Some(n) = self.net.as_mut() {
                n.release(f.id);
            }
            let t1 = std::time::Instant::now();
            let raw = &self.outs[slot];
            let mut v = raw[..DIM].to_vec();
            normalize(&mut v);
            let mut nv = match &self.head {
                Some(h) if raw.len() == DIM * (1 + crate::maskpool::TOKENS) && f.job.tw.iter().any(|w| *w > 0.0) => h.pool(&raw[DIM..], &f.job.tw),
                _ => v.clone(),
            };
            normalize(&mut nv);
            if DET.load(std::sync::atomic::Ordering::Relaxed) {
                let mut g = DETS.lock().unwrap();
                let d = g.get_or_insert_with(Default::default);
                if !d.cancel.contains(&f.job.n) {
                    d.staged.push((f.job.n, f.job.id, f.job.path, v, nv));
                }
                DET_CV.notify_all();
            } else {
                apply(f.job.id, &f.job.path, v, nv);
            }
            self.st.n += 1;
            self.st.pre += f.pre;
            self.st.net += (t1 - f.t0).as_secs_f64() * 1e3;
            self.st.write += t1.elapsed().as_secs_f64() * 1e3;
            dec_pending();
        }
    }

    /// 벡터 하나를 물체에 반영: 검색 줄(벡터 파일) = 이 물체를 본 각도 모두의 평균 방향(길이 1)
    fn apply(id: u32, path: &Path, v: Vec<f32>, nv: Vec<f32>) {
        let mean = put(id, v, nv);
        let bytes: Vec<u8> = mean.iter().flat_map(|x| x.to_le_bytes()).collect();
        common::prof::copy(common::prof::C::EmbedBytes, bytes.len());
        if crate::out::memory_only() {
            crate::out::publish(path, bytes); // --serve: 기억 폴더는 메모리에만(끝에 · 저장 버튼으로 디스크에)
        } else {
            // 임시 파일에 쓰고 이름을 바꿔, 읽는 쪽이 반쯤 쓴 파일을 보지 않게
            let tmp = path.with_extension("bin.tmp");
            common::prof::copy(common::prof::C::EmbedDisk, bytes.len());
            if std::fs::write(&tmp, &bytes).is_ok() && std::fs::rename(&tmp, path).is_ok() {
                crate::out::publish(path, bytes);
            }
        }
    }

    /// det = 결과를 늘 같게(최대 속도 재생): 끝난 벡터는 다음 기록 회차 앞에서 맡긴 차례대로 반영
    pub fn start(model: &str, det: bool) {
        DET.store(det, std::sync::atomic::Ordering::Relaxed);
        let mut net = Net::open(model);
        let patches = net.patches();
        let k = JOBS.load(std::sync::atomic::Ordering::Relaxed).max(1);
        let st = St { net: None, head: crate::maskpool::Head::load(model), patches, ins: vec![Vec::new(); k], outs: vec![Vec::new(); k], fly: (0..k).map(|_| None).collect(), queue: Default::default(), st: Stat::default(), last_pump: std::time::Instant::now(), force: false, cpu: !model.ends_with(".dxnn") };
        let lane = common::rt::SharedLane::new("embed", st, 0, None);
        // NPU 끝 알림(기다림 스레드 또는 DXRT 콜백) → 이 줄에 "자리 slot 끝" 일
        let l2 = lane.clone();
        net.on_done(std::sync::Arc::new(move |slot: u64| {
            l2.push(move |s: &mut St| {
                s.done(slot as usize);
                s.pump();
            })
        }));
        lane.push(move |s: &mut St| s.net = Some(net));
        *LANE.lock().unwrap() = Some(lane);
    }

    /// 함께 올릴 사진 수(`--embed-jobs`, start 전에)
    pub fn set_jobs(k: usize) {
        JOBS.store(k.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    pub fn on() -> bool {
        LANE.lock().unwrap().is_some()
    }

    /// 폴더의 사진(png·jpg, 이름 순)을 벡터로 바꿔 `vec_file` 에 쓴다. 파일이 이미 있으면 그 벡터와 코사인을 견준다
    /// (PC 의 ONNX 벡터를 기준으로 보드 NPU 벡터를 확인): `memmap --embed-check <모델> <사진 폴더> <벡터 파일>`
    pub fn check(model: &str, dir: &str, vec_file: &str) {
        let mut net = Net::open(model);
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{dir}: {e}")).filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("png" | "jpg" | "jpeg"))).collect();
        files.sort();
        let (mut buf, mut all, mut ms) = (Vec::new(), Vec::<f32>::new(), 0.0f64);
        for (i, f) in files.iter().enumerate() {
            let img = image::open(f).unwrap_or_else(|e| panic!("{}: {e}", f.display())).to_rgb8();
            preprocess_into(img.as_raw(), img.width(), img.height(), net.patches(), &mut buf);
            let t = std::time::Instant::now();
            let mut v = Vec::new();
            net.run(&buf, &mut v);
            v.truncate(DIM);
            if i > 0 {
                ms += t.elapsed().as_secs_f64() * 1e3; // 첫 장은 모델 올리는 시간이 섞여 뺌
            }
            normalize(&mut v);
            all.extend(v);
        }
        println!("embed check: 사진 {} 장, 추론 {:.1} ms/장", files.len(), ms / (files.len().max(2) - 1) as f64);
        match std::fs::read(vec_file) {
            Ok(b) => {
                let r: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                assert_eq!(r.len(), all.len(), "{vec_file}: 사진 수가 다름");
                let cos: Vec<f32> = all.chunks(DIM).zip(r.chunks(DIM)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p * q).sum()).collect();
                let mean = cos.iter().sum::<f32>() / cos.len().max(1) as f32;
                println!("기준 벡터와의 코사인: 평균 {mean:.4}, 최소 {:.4}", cos.iter().cloned().fold(f32::INFINITY, f32::min));
            }
            Err(_) => {
                let bytes: Vec<u8> = all.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(vec_file, bytes).unwrap_or_else(|e| panic!("{vec_file}: {e}"));
                println!("기준 벡터를 {vec_file} 에 씀");
            }
        }
    }

    /// 한 기록 회차에 바뀐 물체 사진(번호, 폭, 높이, 화소)을 한 묶음으로 사진 벡터 줄에. 밀린 사진은 물체(파일)마다 가장 새 것만
    pub fn submit(dir: &Path, items: Vec<(u32, u32, u32, std::sync::Arc<Vec<u8>>, [f32; crate::maskpool::TOKENS])>) {
        if items.is_empty() {
            return;
        }
        let Some(lane) = LANE.lock().unwrap().clone() else { return };
        PENDING.fetch_add(items.len(), std::sync::atomic::Ordering::SeqCst);
        let jobs: Vec<Job> = items.into_iter().map(|(id, w, h, rgb, tw)| Job { id, path: dir.join(rel_path(id)), w, h, rgb, tw, n: NEXT_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) }).collect();
        if DET.load(std::sync::atomic::Ordering::Relaxed) {
            // 같은 물체의 앞 회차 사진(아직 반영 전)은 건너뜀으로 표시하고 이 묶음을 기록
            let r = ROUND.load(std::sync::atomic::Ordering::Relaxed);
            let mut g = DETS.lock().unwrap();
            let d = g.get_or_insert_with(Default::default);
            for j in &jobs {
                for e in d.log.iter().filter(|e| e.2 == j.id) {
                    d.cancel.insert(e.0);
                }
                d.log.retain(|e| e.2 != j.id);
                d.log.push((j.n, r, j.id));
            }
            d.staged.retain(|x| !d.cancel.contains(&x.0));
        }
        lane.push(move |s: &mut St| {
            s.st.batches += 1;
            s.st.max_batch = s.st.max_batch.max(jobs.len());
            for j in jobs {
                match s.queue.iter_mut().find(|x| x.path == j.path) {
                    Some(x) => {
                        *x = j; // 앞 사진은 버림(셈에서 뺌)
                        dec_pending();
                    }
                    None => s.queue.push_back(j),
                }
            }
            s.pump();
        });
    }

    /// 비상 열 보호: 사진 벡터 묶음을 늦춤(켬) · 되돌림(끔)
    pub fn throttle(on: bool) {
        THROTTLE.store(on, std::sync::atomic::Ordering::Relaxed);
    }
    static THROTTLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    const THROTTLE_GAP: std::time::Duration = std::time::Duration::from_secs(2);

    /// 맡긴 사진이 모두 벡터가 될 때까지 기다린다(쌓아 둔 것도 반영)
    pub fn drain() {
        if let Some(lane) = LANE.lock().unwrap().clone() {
            lane.push(|s: &mut St| {
                s.force = true;
                s.pump();
                s.force = false;
            });
        }
        if rayon::current_thread_index().is_some() {
            common::rt::wait_until(|| PENDING.load(std::sync::atomic::Ordering::SeqCst) == 0);
        } else {
            let mut g = DRAINED.0.lock().unwrap();
            while PENDING.load(std::sync::atomic::Ordering::SeqCst) > 0 {
                g = DRAINED.1.wait(g).unwrap();
            }
        }
        commit(u64::MAX);
    }

    /// 회차 upto 까지 맡긴 사진의 벡터를 맡긴 차례대로 반영(끝나지 않은 것은 기다림)
    fn commit(upto: u64) {
        let mut g = DETS.lock().unwrap();
        let Some(d) = g.as_mut() else { return };
        let need: Vec<u64> = d.log.iter().filter(|e| e.1 <= upto).map(|e| e.0).collect();
        if need.is_empty() {
            return;
        }
        let c = std::time::Instant::now();
        let ready = |d: &Det| need.iter().all(|n| d.staged.iter().any(|x| x.0 == *n));
        if rayon::current_thread_index().is_some() {
            // 풀 일꾼에서 기다림(물체 기억 일): 사진 벡터 줄을 이 일꾼이 대신 돌 수 있게 잠금을 놓고 기다림
            drop(g);
            common::rt::wait_until(|| DETS.lock().unwrap().as_ref().is_some_and(&ready));
            g = DETS.lock().unwrap();
        } else {
            while !ready(g.as_ref().unwrap()) {
                g = DET_CV.wait(g).unwrap();
            }
        }
        common::prof::wait(common::prof::W::EmbedNpu, c);
        let d = g.as_mut().unwrap();
        let mut v: Vec<_> = Vec::new();
        let mut rest = Vec::new();
        for x in d.staged.drain(..) {
            if need.contains(&x.0) { v.push(x) } else { rest.push(x) }
        }
        d.staged = rest;
        d.log.retain(|e| !need.contains(&e.0));
        if d.log.is_empty() {
            d.cancel.clear();
        }
        drop(g);
        v.sort_by_key(|x| x.0);
        for (_, id, path, e, ne) in v {
            apply(id, &path, e, ne);
        }
    }

    /// 기록 회차 앞(결과를 늘 같게 하는 재생): DET_LAG 회차 앞에 맡긴 묶음을 반영(덜 끝났으면 기다림). 실시간은 아무것도 안 함(끝나는 대로 반영됨)
    pub fn sync_round() {
        if DET.load(std::sync::atomic::Ordering::Relaxed) && on() {
            let r = ROUND.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if r > DET_LAG {
                commit(r - DET_LAG);
            }
        }
    }

    pub fn finish() {
        drain();
        if let Some(lane) = LANE.lock().unwrap().take() {
            lane.wait_idle();
            let (jobs, busy) = lane.stats();
            lane.push(move |s: &mut St| {
                let n = s.st.n.max(1) as f64;
                common::info!("embed: {} vectors({} 묶음, 가장 큰 묶음 {}), 전처리 {:.1} ms · 모델(올림 → 받음) {:.1} ms · 정리 {:.2} ms 씩, 줄 일 {jobs} 번 {busy:.2} 초",
                    s.st.n, s.st.batches, s.st.max_batch, s.st.pre / n, s.st.net / n, s.st.write / n);
                s.net = None; // 엔진(과 기다림 스레드)을 여기서 닫음
            });
            lane.wait_idle();
        }
    }
}

/// 인코더를 올리고 계산 스레드를 띄운다(`--embed <vision_model.onnx | .dxnn>`)
#[cfg(any(feature = "onnx", feature = "dxrt"))]
pub use worker::{check, drain, finish, on, set_jobs, start, submit, sync_round, throttle};

#[cfg(not(any(feature = "onnx", feature = "dxrt")))]
pub fn start(_model: &str, _det: bool) {
    panic!("--embed: 이 빌드에는 모델 실행기가 없다(onnx 또는 dxrt 기능으로 빌드)");
}
#[cfg(not(any(feature = "onnx", feature = "dxrt")))]
pub fn check(model: &str, _dir: &str, _vec_file: &str) {
    start(model, false);
}
#[cfg(not(any(feature = "onnx", feature = "dxrt")))]
pub fn sync_round() {}
#[cfg(not(any(feature = "onnx", feature = "dxrt")))]
pub fn throttle(_on: bool) {}
#[cfg(not(any(feature = "onnx", feature = "dxrt")))]
pub fn set_jobs(_k: usize) {}
#[cfg(not(any(feature = "onnx", feature = "dxrt")))]
pub fn on() -> bool {
    false
}
#[cfg(not(any(feature = "onnx", feature = "dxrt")))]
pub fn submit(_dir: &Path, _items: Vec<(u32, u32, u32, std::sync::Arc<Vec<u8>>, [f32; crate::maskpool::TOKENS])>) {}
#[cfg(not(any(feature = "onnx", feature = "dxrt")))]
pub fn finish() {}
#[cfg(not(any(feature = "onnx", feature = "dxrt")))]
pub fn drain() {}

#[cfg(test)]
mod tests {
    use super::*;

    /// 화소마다 하던 원래 식(기준)
    fn preprocess_ref(rgb: &[u8], w: usize, h: usize, patches: bool) -> Vec<f32> {
        let n = SIZE * SIZE;
        let mut out = vec![0.0; 3 * n];
        let axis = |dst: usize, src: usize| -> (usize, usize, f32) {
            let f = ((dst as f32 + 0.5) * src as f32 / SIZE as f32 - 0.5).max(0.0);
            let a = (f as usize).min(src - 1);
            (a, (a + 1).min(src - 1), f - a as f32)
        };
        for y in 0..SIZE {
            let (y0, y1, fy) = axis(y, h);
            for x in 0..SIZE {
                let (x0, x1, fx) = axis(x, w);
                for c in 0..3 {
                    let p = |yy: usize, xx: usize| rgb[(yy * w + xx) * 3 + c] as f32;
                    let top = p(y0, x0) * (1.0 - fx) + p(y0, x1) * fx;
                    let bot = p(y1, x0) * (1.0 - fx) + p(y1, x1) * fx;
                    let i = if patches { ((y / 32) * 8 + x / 32) * 3072 + c * 1024 + (y % 32) * 32 + x % 32 } else { c * n + y * SIZE + x };
                    out[i] = (top * (1.0 - fy) + bot * fy) * (1.0 / 127.5) - 1.0;
                }
            }
        }
        out
    }

    #[test]
    fn preprocess_same_as_reference() {
        let mut s = 7u64;
        for &(w, h) in &[(37usize, 211usize), (640, 480), (256, 256), (3, 900)] {
            let rgb: Vec<u8> = (0..w * h * 3).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (s >> 56) as u8 }).collect();
            for patches in [false, true] {
                let mut out = vec![];
                preprocess_into(&rgb, w as u32, h as u32, patches, &mut out);
                let want = preprocess_ref(&rgb, w, h, patches);
                assert!(out.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{w}x{h} {patches}");
            }
        }
    }

    #[test]
    fn preprocess_range_and_layout() {
        // 왼쪽 절반 빨강, 오른쪽 절반 파랑인 4×2 사진
        let mut rgb = vec![];
        for _ in 0..2 {
            for x in 0..4 {
                rgb.extend_from_slice(if x < 2 { &[255, 0, 0] } else { &[0, 0, 255] });
            }
        }
        let mut out = vec![];
        preprocess(&rgb, 4, 2, &mut out);
        let n = SIZE * SIZE;
        assert_eq!(out.len(), 3 * n);
        let at = |c: usize, x: usize, y: usize| out[c * n + y * SIZE + x];
        assert!((at(0, 10, 100) - 1.0).abs() < 1e-4 && (at(2, 10, 100) + 1.0).abs() < 1e-4);
        assert!((at(0, 245, 100) + 1.0).abs() < 1e-4 && (at(2, 245, 100) - 1.0).abs() < 1e-4);
        assert!(out.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    /// `cargo test -p ondevice --release -- --ignored --nocapture name_time`: 물체 하나 이름 붙이기(실내 낱말 104 개와 견줌) 시간
    #[test]
    #[ignore]
    fn name_time() {
        let mut s = 3u64;
        let mut rnd = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 40) as f32) / (1u64 << 24) as f32 - 0.5 };
        let names: Vec<String> = (0..104).map(|i| format!("n{i}")).collect();
        let vecs: Vec<f32> = (0..104 * DIM).map(|_| rnd()).collect();
        let es: Vec<Vec<f32>> = (0..1000).map(|_| { let mut e: Vec<f32> = (0..DIM).map(|_| rnd()).collect(); normalize(&mut e); e }).collect();
        let c = std::time::Instant::now();
        for e in &es {
            std::hint::black_box(look_name(&names, &vecs, e));
        }
        println!("이름 붙이기(낱말 104 개): {:.1} µs/물체", c.elapsed().as_secs_f64() * 1e6 / es.len() as f64);
    }

    #[test]
    fn structure_needs_closest_structural_word_and_p() {
        // 낱말 셋이 축 0 · 1 · 2 를 가리킴: floor, wall, furniture
        let names: Vec<String> = ["floor", "wall", "furniture"].iter().map(|s| s.to_string()).collect();
        let mut vecs = vec![0.0; 3 * DIM];
        for i in 0..3 {
            vecs[i * DIM + i] = 1.0;
        }
        let pic = |ax: usize, c: f32| { let mut e = vec![0.0; DIM]; e[ax] = c; e[9] = (1.0 - c * c).sqrt(); e };
        assert_eq!(look_structure(&names, &vecs, &pic(1, 0.2)), Some("wall")); // p = 0.99
        assert_eq!(look_structure(&names, &vecs, &pic(2, 0.2)), None); // 가장 닮은 낱말이 가구
        assert_eq!(look_structure(&names, &vecs, &pic(0, 0.05)), None); // floor 이지만 p < 0.003
    }

    #[test]
    fn look_name_picks_the_closest_word() {
        // 이름 셋이 축 0·1·2 를 가리키고, 사진은 축 1 쪽으로 조금 기움(글·사진 코사인은 실제 모델에서도 작다)
        let names: Vec<String> = ["chair", "monitor", "cup"].iter().map(|s| s.to_string()).collect();
        let mut vecs = vec![0.0; 3 * DIM];
        for i in 0..3 {
            vecs[i * DIM + i] = 1.0;
        }
        let mut e = vec![0.0; DIM];
        (e[1], e[9]) = (0.1, 0.995);
        let (n, p) = look_name(&names, &vecs, &e).unwrap();
        assert_eq!(n, "monitor");
        assert!(p > 0.99, "{p}");
        // 어느 이름도 닮지 않은 사진: 1 등과 2 등의 차가 NAME_MARGIN 아래라 몫 0(이름 안 붙음)
        e[1] = 0.0;
        assert_eq!(look_name(&names, &vecs, &e).unwrap().1, 0.0);
    }

    #[test]
    fn normalize_unit_length() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let mut z = vec![0.0; 4];
        normalize(&mut z);
        assert!(z.iter().all(|x| *x == 0.0));
    }
}
