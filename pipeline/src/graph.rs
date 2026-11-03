//! 씬그래프 층: 방(점유 지도의 빈칸을 구역으로) + 엣지(contains 방→물체, on 위·아래, near 가까운 2 개).
//! 방 나누기: 빈칸 거리변환(체임퍼 3-4) → 통로보다 넓은 곳(폭 ≥ min_width)을 씨앗 → 빈칸 안에서 씨앗을 동시에 키움(BFS).
use crate::maps::GridImg;
use serde_json::{json, Value};
use std::collections::VecDeque;

pub struct Rooms {
    pub label: Vec<i32>, // 칸마다 방 번호(-1 = 없음)
    pub n: usize,
}

pub fn segment(g: &GridImg, res: f64, min_width: f64, min_area: f64) -> Rooms {
    let (w, h) = (g.w, g.h);
    let free: Vec<bool> = g.px.iter().map(|&p| p == 254).collect();
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

/// objs: (id, pos, extent, gone). 반환: (rooms JSON, edges JSON)
pub fn build(g: &GridImg, res: f64, objs: &[(u32, [f64; 3], [f64; 3], bool)], names: &dyn Fn(usize) -> (String, f32)) -> (Value, Value) {
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
            let (name, conf) = names(r);
            json!({"id": r, "name": name, "conf": conf, "centroid": [cx, cy], "area_m2": s.2 as f64 * res * res,
                   "color": PALETTE[r % PALETTE.len()], "objects": members[r]})
        })
        .collect();
    (Value::Array(rj), Value::Array(edges))
}
