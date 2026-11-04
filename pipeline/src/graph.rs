//! 씬그래프 층: 방(점유 지도의 빈칸을 구역으로) + 엣지(contains 방→물체, on 위·아래, near 가까운 2 개).
//! 방 나누기: 빈칸의 작은 구멍 메우기(닫힘) → 빈칸 거리변환(체임퍼 3-4) → 통로보다 넓은 곳(폭 ≥ min_width)을 씨앗 → 빈칸 안에서 씨앗을 동시에 키움(BFS)
//! → 맞닿은 경계가 1.2 m 넘는 방끼리 합침(문은 그보다 좁다).
use crate::maps::GridImg;
use serde_json::{json, Value};
use std::collections::VecDeque;

pub struct Rooms {
    pub label: Vec<i32>, // 칸마다 방 번호(-1 = 없음)
    pub n: usize,
}

pub fn segment(g: &GridImg, res: f64, min_width: f64, min_area: f64) -> Rooms {
    let (w, h) = (g.w, g.h);
    // 빈칸 사이의 작은 미관측 구멍 메우기(닫힘: 빈칸을 r 칸 넓혔다가 다시 좁힘, 막힌 칸은 그대로): 손카메라는 바닥을 듬성듬성 봐서
    // 빈칸이 얼룩덜룩하면 거리변환이 작아져 한 방이 여러 씨앗으로 갈라진다
    let occ: Vec<bool> = g.px.iter().map(|&p| p == 0).collect();
    let mut free: Vec<bool> = g.px.iter().map(|&p| p == 254).collect();
    let r = 2i64;
    let grow = |m: &[bool], want: bool| -> Vec<bool> {
        let mut o = m.to_vec();
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let i = (y * w as i64 + x) as usize;
                if m[i] == want {
                    continue;
                }
                'n: for dy in -r..=r {
                    for dx in -r..=r {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 && m[(ny * w as i64 + nx) as usize] == want {
                            o[i] = want;
                            break 'n;
                        }
                    }
                }
            }
        }
        o
    };
    let closed = grow(&grow(&free, true), false);
    for i in 0..w * h {
        free[i] = (free[i] || closed[i]) && !occ[i];
    }
    // 체임퍼 거리(칸 단위 ×3)
    let big = i32::MAX / 4;
    let mut d: Vec<i32> = free.iter().map(|&f| if f { big } else { 0 }).collect();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if d[i] == 0 {
                continue;
            }
            let mut m = d[i];
            if x > 0 { m = m.min(d[i - 1] + 3); }
            if y > 0 { m = m.min(d[i - w] + 3); }
            if x > 0 && y > 0 { m = m.min(d[i - w - 1] + 4); }
            if x + 1 < w && y > 0 { m = m.min(d[i - w + 1] + 4); }
            d[i] = m;
        }
    }
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            if d[i] == 0 {
                continue;
            }
            let mut m = d[i];
            if x + 1 < w { m = m.min(d[i + 1] + 3); }
            if y + 1 < h { m = m.min(d[i + w] + 3); }
            if x + 1 < w && y + 1 < h { m = m.min(d[i + w + 1] + 4); }
            if x > 0 && y + 1 < h { m = m.min(d[i + w - 1] + 4); }
            d[i] = m;
        }
    }
    let thr = ((min_width / 2.0 / res) * 3.0) as i32;
    let mut lab = vec![-1i32; w * h];
    let mut q = VecDeque::new();
    let comp = |seed: &dyn Fn(usize) -> bool, lab: &mut Vec<i32>, n: &mut i32| {
        for s in 0..w * h {
            if !seed(s) || lab[s] >= 0 {
                continue;
            }
            let mut st = vec![s];
            lab[s] = *n;
            while let Some(i) = st.pop() {
                let (x, y) = (i % w, i / w);
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if seed(j) && lab[j] < 0 {
                        lab[j] = *n;
                        st.push(j);
                    }
                }
            }
            *n += 1;
        }
    };
    let mut ni = 0i32;
    comp(&|i| d[i] >= thr, &mut lab, &mut ni);
    if ni == 0 {
        comp(&|i| free[i], &mut lab, &mut ni);
    }
    let n = ni as usize;
    for i in 0..w * h {
        if lab[i] >= 0 {
            q.push_back(i);
        }
    }
    while let Some(i) = q.pop_front() {
        let (x, y) = (i % w, i / w);
        for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let (nx, ny) = (x as i32 + dx, y as i32 + dy);
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let j = ny as usize * w + nx as usize;
            if free[j] && lab[j] < 0 {
                lab[j] = lab[i];
                q.push_back(j);
            }
        }
    }
    // 맞닿은 경계가 문보다 넓으면(> 1.2 m) 같은 방: 가구가 빈칸을 좁혀 한 방이 여러 씨앗으로 갈라지는 것을 되붙인다
    let mut contact: std::collections::HashMap<(i32, i32), usize> = std::collections::HashMap::new();
    for y in 0..h {
        for x in 0..w {
            let a = lab[y * w + x];
            if a < 0 {
                continue;
            }
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx >= w || ny >= h {
                    continue;
                }
                let b = lab[ny * w + nx];
                if b >= 0 && b != a {
                    *contact.entry((a.min(b), a.max(b))).or_insert(0) += 1;
                }
            }
        }
    }
    let mut parent: Vec<i32> = (0..n as i32).collect();
    fn root(p: &mut [i32], mut i: i32) -> i32 {
        while p[i as usize] != i {
            p[i as usize] = p[p[i as usize] as usize];
            i = p[i as usize];
        }
        i
    }
    for (&(a, b), &c) in &contact {
        if c as f64 * res > 1.2 {
            let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
            if ra != rb {
                parent[ra.max(rb) as usize] = ra.min(rb);
            }
        }
    }
    for l in lab.iter_mut() {
        if *l >= 0 {
            *l = root(&mut parent, *l);
        }
    }
    // 작은 방 버리고 번호 다시 매김
    let mut area = vec![0usize; n];
    for &l in &lab {
        if l >= 0 {
            area[l as usize] += 1;
        }
    }
    let mut remap = vec![-1i32; n];
    let mut k = 0;
    for r in 0..n {
        if area[r] as f64 * res * res >= min_area {
            remap[r] = k;
            k += 1;
        }
    }
    for l in lab.iter_mut() {
        if *l >= 0 {
            *l = remap[*l as usize];
        }
    }
    Rooms { label: lab, n: k as usize }
}

const PALETTE: [[u8; 3]; 7] = [[66, 133, 244], [52, 168, 83], [251, 188, 5], [234, 67, 53], [171, 71, 188], [0, 172, 193], [255, 112, 67]];

/// 방 종류: 방 안 물체 이름(COCO)으로 점수 — 종류마다 단서 물체와 가중. 가장 높은 종류가 2 점 넘으면 그 이름, 확신 = 그 점수 / 전체.
pub fn room_type(names: &[&str]) -> (&'static str, f32) {
    const KINDS: [(&str, &[(&str, f32)]); 6] = [
        ("office", &[("tv", 1.0), ("laptop", 1.5), ("keyboard", 1.5), ("mouse", 1.5), ("book", 0.5), ("chair", 0.5), ("cell phone", 0.5)]),
        ("kitchen", &[("refrigerator", 2.0), ("microwave", 2.0), ("oven", 2.0), ("sink", 1.5), ("toaster", 2.0), ("bowl", 0.5), ("cup", 0.3), ("bottle", 0.3), ("knife", 0.5), ("spoon", 0.5), ("fork", 0.5)]),
        ("living room", &[("couch", 2.0), ("tv", 0.8), ("remote", 1.0), ("potted plant", 0.5), ("vase", 0.5)]),
        ("bedroom", &[("bed", 3.0), ("teddy bear", 0.5), ("clock", 0.3)]),
        ("dining room", &[("dining table", 1.5), ("chair", 0.5), ("wine glass", 1.0), ("cup", 0.3), ("bowl", 0.3)]),
        ("bathroom", &[("toilet", 3.0), ("sink", 1.0), ("toothbrush", 2.0), ("hair drier", 2.0)]),
    ];
    let sc: Vec<f32> = KINDS.iter().map(|(_, cues)| names.iter().map(|n| cues.iter().find(|c| c.0 == *n).map_or(0.0, |c| c.1)).sum()).collect();
    let tot: f32 = sc.iter().sum();
    let (bi, &bs) = sc.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
    if bs < 2.0 {
        return ("room", 0.0);
    }
    (KINDS[bi].0, bs / tot.max(1e-6))
}

/// objs: (id, pos, extent, gone, 이름). 반환: (rooms JSON, edges JSON)
pub fn build(g: &GridImg, res: f64, objs: &[(u32, [f64; 3], [f64; 3], bool, &str)]) -> (Value, Value) {
    let rooms = segment(g, res, 1.2, 3.0);
    let (w, h) = (g.w, g.h);
    let cell = |x: f64, y: f64| -> (i64, i64) { ((x / res).floor() as i64 - g.x0 as i64, h as i64 - 1 - ((y / res).floor() as i64 - g.y0 as i64)) };
    let room_at = |x: f64, y: f64| -> Option<usize> {
        let (cx, cy) = cell(x, y);
        let r = (0.5 / res) as i64;
        let mut best: Option<(i64, usize)> = None;
        for yy in (cy - r).max(0)..=(cy + r).min(h as i64 - 1) {
            for xx in (cx - r).max(0)..=(cx + r).min(w as i64 - 1) {
                let l = rooms.label[yy as usize * w + xx as usize];
                if l >= 0 {
                    let dd = (xx - cx).pow(2) + (yy - cy).pow(2);
                    if best.is_none_or(|b| dd < b.0) {
                        best = Some((dd, l as usize));
                    }
                }
            }
        }
        best.map(|b| b.1)
    };
    let mut sum = vec![(0.0f64, 0.0f64, 0usize); rooms.n];
    for (i, &l) in rooms.label.iter().enumerate() {
        if l >= 0 {
            let s = &mut sum[l as usize];
            s.0 += (i % w) as f64;
            s.1 += (i / w) as f64;
            s.2 += 1;
        }
    }
    let mut members: Vec<Vec<u32>> = vec![vec![]; rooms.n];
    let mut edges: Vec<Value> = vec![];
    let live: Vec<_> = objs.iter().filter(|o| !o.3).collect();
    for o in &live {
        if let Some(r) = room_at(o.1[0], o.1[1]) {
            members[r].push(o.0);
            edges.push(json!([format!("R{r}"), format!("O{}", o.0), "contains"]));
        }
    }
    // on / near(물체마다 가까운 2 개)
    let mut near = vec![];
    for i in 0..live.len() {
        for j in i + 1..live.len() {
            let (a, b) = (live[i], live[j]);
            let dxy = ((a.1[0] - b.1[0]).powi(2) + (a.1[1] - b.1[1]).powi(2)).sqrt();
            let over = (0..2).all(|k| (a.1[k] - b.1[k]).abs() < (a.2[k] + b.2[k]) / 2.0 * 0.8);
            if over && ((a.1[2] - a.2[2] / 2.0) - (b.1[2] + b.2[2] / 2.0)).abs() < 0.12 {
                edges.push(json!([format!("O{}", a.0), format!("O{}", b.0), "on"]));
            } else if over && ((b.1[2] - b.2[2] / 2.0) - (a.1[2] + a.2[2] / 2.0)).abs() < 0.12 {
                edges.push(json!([format!("O{}", b.0), format!("O{}", a.0), "on"]));
            } else if dxy < 0.8 && (a.1[2] - b.1[2]).abs() < 0.5 {
                near.push((dxy, a.0, b.0));
            }
        }
    }
    near.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let mut deg = std::collections::HashMap::new();
    for (_, a, b) in near {
        if *deg.get(&a).unwrap_or(&0) >= 2 || *deg.get(&b).unwrap_or(&0) >= 2 {
            continue;
        }
        *deg.entry(a).or_insert(0) += 1;
        *deg.entry(b).or_insert(0) += 1;
        edges.push(json!([format!("O{a}"), format!("O{b}"), "near"]));
    }
    let rj: Vec<Value> = (0..rooms.n)
        .map(|r| {
            let s = sum[r];
            let cx = (g.x0 as f64 + s.0 / s.2 as f64 + 0.5) * res;
            let cy = (g.y0 as f64 + (h as f64 - 1.0 - s.1 / s.2 as f64) + 0.5) * res;
            let nm: Vec<&str> = live.iter().filter(|o| members[r].contains(&o.0)).map(|o| o.4).collect();
            let (kind, conf) = room_type(&nm);
            let name = if kind == "room" { format!("room {r}") } else { kind.to_string() };
            json!({"id": r, "name": name, "conf": conf, "centroid": [cx, cy], "area_m2": s.2 as f64 * res * res,
                   "color": PALETTE[r % PALETTE.len()], "objects": members[r]})
        })
        .collect();
    (Value::Array(rj), Value::Array(edges))
}
