//! view.json 한 판: 같은 프로세스의 물체 기억(`ondevice --serve`)이 넘긴 구조(물체 · 격자 · 방 · 엣지 JSON 값 + 지나온 길 로그)이거나,
//! 따로 띄운 뷰어가 읽은 파일 바이트. JSON 바이트는 HTTP 로 누가 물을 때 판마다 한 번만 짠다(디스크에 쓰는 view.json 과 같은 바이트).
//! 검색 · 가구 자리는 값을 그대로 읽는다(다시 해석하지 않음).
use serde_json::Value;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

/// 지나온 길(시각, 위치): 덧붙이기만 하는 로그. 판은 길이만 들고 있어 판마다 길 전체를 복사하지 않는다
/// (다시 쌓기로 길이 새로 만들어지면 로그도 새로)
#[derive(Default)]
pub struct TrajLog(RwLock<Vec<(f64, [f64; 3])>>);

impl TrajLog {
    pub fn new() -> Arc<TrajLog> {
        Arc::new(TrajLog::default())
    }
    pub fn extend(&self, s: &[(f64, [f64; 3])]) {
        self.0.write().unwrap().extend_from_slice(s);
    }
    /// view.json 바이트(길은 앞 n 개)
    pub fn view_bytes(&self, head: &Value, n: usize) -> Vec<u8> {
        live_bytes(head, &self.0.read().unwrap()[..n])
    }
    pub fn len(&self) -> usize {
        self.0.read().unwrap().len()
    }
}

enum Src {
    /// head: view.json 에서 graph.nodes(지나온 길)만 뺀 값, 길은 로그 앞 n 개
    Live { traj: Arc<TrajLog>, n: usize },
    File(Arc<Vec<u8>>),
}

pub struct View {
    pub ver: String,
    src: Src,
    doc: OnceLock<Option<Value>>,
    bytes: OnceLock<Arc<Vec<u8>>>,
    body: Mutex<Option<(String, Arc<Vec<u8>>)>>,
}

impl View {
    pub fn live(ver: u64, head: Value, traj: Arc<TrajLog>, n: usize) -> View {
        let doc = OnceLock::new();
        let _ = doc.set(Some(head));
        View { ver: format!("s{ver}"), src: Src::Live { traj, n }, doc, bytes: OnceLock::new(), body: Mutex::new(None) }
    }
    pub fn file(ver: String, bytes: Arc<Vec<u8>>) -> View {
        let b = OnceLock::new();
        let _ = b.set(bytes.clone());
        View { ver, src: Src::File(bytes), doc: OnceLock::new(), bytes: b, body: Mutex::new(None) }
    }
    /// 값(물체 · 방 …): 같은 프로세스 판은 그대로, 파일 판은 처음 물을 때 한 번 해석
    pub fn doc(&self) -> Option<&Value> {
        self.doc
            .get_or_init(|| match &self.src {
                Src::File(b) => serde_json::from_slice(b).ok(),
                Src::Live { .. } => None,
            })
            .as_ref()
    }
    /// view.json 바이트(디스크에 쓰는 것과 같은 바이트)
    pub fn bytes(&self) -> Arc<Vec<u8>> {
        self.bytes
            .get_or_init(|| match &self.src {
                Src::File(b) => b.clone(),
                Src::Live { traj, n } => Arc::new(live_bytes(self.doc().unwrap_or(&Value::Null), &traj.0.read().unwrap()[..*n])),
            })
            .clone()
    }
    /// /api/view 몸: {"v":판,"map_v":지도 판,"view":…} — 지도 판이 같으면 짠 것을 그대로
    pub fn api_body(&self, map_v: &str) -> Arc<Vec<u8>> {
        let mut c = self.body.lock().unwrap();
        if let Some((m, b)) = c.as_ref() {
            if m == map_v {
                return b.clone();
            }
        }
        let v = self.bytes();
        let mut b = Vec::with_capacity(v.len() + 64);
        b.extend_from_slice(format!("{{\"v\":\"{}\",\"map_v\":\"{}\",\"view\":", self.ver, map_v).as_bytes());
        b.extend_from_slice(&v);
        b.push(b'}');
        let b = Arc::new(b);
        *c = Some((map_v.to_string(), b.clone()));
        b
    }
}

/// head(키 차례 = serde_json 맵 차례) + 지나온 길 노드 → `serde_json::to_vec(view)` 와 같은 바이트
pub fn live_bytes(head: &Value, traj: &[(f64, [f64; 3])]) -> Vec<u8> {
    let mut b = Vec::with_capacity(4096 + traj.len() * 80);
    let Some(m) = head.as_object() else {
        serde_json::to_writer(&mut b, head).unwrap();
        return b;
    };
    b.push(b'{');
    for (i, (k, v)) in m.iter().enumerate() {
        if i > 0 {
            b.push(b',');
        }
        serde_json::to_writer(&mut b, k).unwrap();
        b.push(b':');
        if k == "graph" {
            // {"edges": …, "nodes": [{"id":"A<i>","kind":"agent","pos":[…],"t":…}, …]}
            b.extend_from_slice(b"{\"edges\":");
            serde_json::to_writer(&mut b, &v["edges"]).unwrap();
            b.extend_from_slice(b",\"nodes\":[");
            for (i, (t, p)) in traj.iter().enumerate() {
                if i > 0 {
                    b.push(b',');
                }
                b.extend_from_slice(format!("{{\"id\":\"A{i}\",\"kind\":\"agent\",\"pos\":").as_bytes());
                serde_json::to_writer(&mut b, &serde_json::json!(p)).unwrap();
                b.extend_from_slice(b",\"t\":");
                serde_json::to_writer(&mut b, &serde_json::json!(t)).unwrap();
                b.push(b'}');
            }
            b.extend_from_slice(b"]}");
        } else {
            serde_json::to_writer(&mut b, v).unwrap();
        }
    }
    b.push(b'}');
    b
}

static CUR: Mutex<Option<Arc<View>>> = Mutex::new(None);

/// 같은 프로세스의 물체 기억이 새 판을 건다(뷰어는 파일 대신 이것을 씀)
pub fn publish(v: View) {
    *CUR.lock().unwrap() = Some(Arc::new(v));
    crate::memfs::bump();
}

pub fn current() -> Option<Arc<View>> {
    CUR.lock().unwrap().clone()
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    #[test]
    fn same_bytes_as_value() {
        let traj = vec![(1.5, [0.1, -0.0, 2.0]), (2.25, [1e-9, 3.0, f64::NAN])];
        let edges = json!([["R1", "O2", "contains"]]);
        let nodes: Vec<_> = traj.iter().enumerate().map(|(i, (t, p))| json!({"id": format!("A{i}"), "kind": "agent", "pos": p, "t": t})).collect();
        let full = json!({"stamp": 3.0, "pose": [1, 2], "grid": null, "objects": [{"a": 1}], "rooms": [], "doors": [], "graph": {"nodes": nodes, "edges": edges}, "robot": "x", "cam": {"pos": [1.0]}, "scene_points": {"run": 1}});
        let mut head = full.clone();
        head["graph"] = json!({"edges": edges});
        assert_eq!(super::live_bytes(&head, &traj), serde_json::to_vec(&full).unwrap());
    }
}
