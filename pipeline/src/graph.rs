//! 씬그래프 층: 방(rooms.rs 가 점유 지도의 빈칸을 나눈 것) + 엣지(contains 방→물체, door 방↔방, on 위·아래, near 가까운 2 개).
use crate::maps::GridImg;
use crate::rooms::{self, Tracker};
use serde_json::{json, Value};

const PALETTE: [[u8; 3]; 7] = [[66, 133, 244], [52, 168, 83], [251, 188, 5], [234, 67, 53], [171, 71, 188], [0, 172, 193], [255, 112, 67]];

/// objs: (id, pos, extent, gone, 이름). 반환: (rooms JSON, doors JSON, edges JSON)
pub fn build(tr: &mut Tracker, g: &GridImg, res: f64, objs: &[(u32, [f64; 3], [f64; 3], bool, &str)]) -> (Value, Value, Value) {
    let params = tr.params.clone();
    let seg = tr.update(g, res);
    let live: Vec<_> = objs.iter().filter(|o| !o.3).collect();
    let ro: Vec<_> = live.iter().map(|o| (o.0, o.4, o.1, o.2)).collect();
    let (labels, obj_room) = rooms::name_rooms(seg, &ro, &params);
    let mut edges: Vec<Value> = vec![];
    for (o, &r) in live.iter().zip(&obj_room) {
        if r != 0 {
            edges.push(json!([format!("R{r}"), format!("O{}", o.0), "contains"]));
        }
    }
    for d in &seg.doors {
        edges.push(json!([format!("R{}", d.a), format!("R{}", d.b), "door"]));
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
    let rj: Vec<Value> = seg.rooms.iter().zip(&labels)
        .map(|(r, l)| {
            let ev: Vec<Value> = l.evidence.iter().map(|e| json!({"obj": e.0, "name": e.1, "type": e.2, "w": e.3})).collect();
            json!({"id": r.id, "name": l.name, "type": l.kind, "conf": l.conf, "probs": l.probs, "centroid": r.centroid, "area_m2": r.area_m2,
                   "max_clear": r.max_clear, "color": PALETTE[r.id as usize % PALETTE.len()], "objects": l.objects, "evidence": ev})
        })
        .collect();
    let dj: Vec<Value> = seg.doors.iter().map(|d| json!({"a": d.a, "b": d.b, "pos": d.pos, "width": d.width, "seam": d.seam})).collect();
    (Value::Array(rj), Value::Array(dj), Value::Array(edges))
}
