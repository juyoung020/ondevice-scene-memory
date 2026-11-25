//! ONNX 모델을 Rust 로 직접 돌리는 실행기(ONNX Runtime 대신). 모델 파일의 그래프를 읽어 상수 접기 · 묶기(Conv+Relu/SiLU,
//! MatMul+편향) · 무게 미리 펴기를 한 번 하고, 노드를 차례로 실행한다. f32 만(정수는 모양 · 번호 계산용 i64).
//! 무거운 연산(Conv · MatMul · Gemm · ConvTranspose)은 [`gemm`] 의 판 행렬곱, 나머지는 rayon 으로 나눈 단순 반복.
//! 지원하는 연산은 이 저장소 모델(MegaLoc · XFeat · SigLIP2 그림 · ObjectSAM)이 쓰는 것.
pub mod buf;
pub mod gemm;
mod ops;
pub mod pool;
pub mod proto;

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum Data {
    F(Vec<f32>),
    I(Vec<i64>),
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Data,
}

impl Tensor {
    pub fn f(shape: Vec<usize>, v: Vec<f32>) -> Tensor {
        Tensor { shape, data: Data::F(v) }
    }
    pub fn i(shape: Vec<usize>, v: Vec<i64>) -> Tensor {
        Tensor { shape, data: Data::I(v) }
    }
    pub fn len(&self) -> usize {
        match &self.data {
            Data::F(v) => v.len(),
            Data::I(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// f32 값(정수 텐서면 오류)
    pub fn fv(&self) -> Result<&[f32], String> {
        match &self.data {
            Data::F(v) => Ok(v),
            Data::I(_) => Err("f32 텐서가 필요".into()),
        }
    }
    /// 정수로(모양 · 번호 입력)
    pub fn ints(&self) -> Vec<i64> {
        match &self.data {
            Data::F(v) => v.iter().map(|&x| x as i64).collect(),
            Data::I(v) => v.clone(),
        }
    }
    pub fn into_f(self) -> Vec<f32> {
        match self.data {
            Data::F(v) => v,
            Data::I(v) => v.into_iter().map(|x| x as f32).collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Attr {
    F(f32),
    I(i64),
    S(String),
    T(Tensor),
    Fs(Vec<f32>),
    Is(Vec<i64>),
}

/// 행렬곱 · 합성곱 끝에 바로 붙이는 활성
#[derive(Clone, Debug)]
pub enum Act {
    None,
    Relu,
    /// x · sigmoid(x)
    Silu,
    /// 원소별 연산 줄(뒤따르던 노드들을 묶은 것)
    Prog(std::sync::Arc<Prog>),
}

impl Act {
    #[inline]
    pub fn apply(&self, v: &mut [f32]) {
        match self {
            Act::None => {}
            Act::Relu => v.iter_mut().for_each(|x| *x = x.max(0.0)),
            Act::Silu => v.iter_mut().for_each(|x| *x *= sigmoid(*x)),
            Act::Prog(p) => {
                for c in v.chunks_mut(64) {
                    p.run(c);
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Src {
    Reg(usize),
    K(f32),
}

#[derive(Clone, Copy, Debug)]
pub enum EOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Relu,
    Sigmoid,
    Tanh,
    Erf,
    Sqrt,
    Neg,
    Exp,
}

/// 원소별 계산 순서: 칸 0 = 입력, 명령마다 새 칸 하나, 결과 = 마지막 칸
#[derive(Debug)]
pub struct Prog {
    pub ins: Vec<(EOp, Src, Src)>,
}

impl Prog {
    fn run(&self, v: &mut [f32]) {
        let n = v.len();
        let mut regs = vec![[0f32; 64]; self.ins.len() + 1];
        regs[0][..n].copy_from_slice(v);
        for (i, &(op, a, b)) in self.ins.iter().enumerate() {
            let (lo, hi) = regs.split_at_mut(i + 1);
            let o = &mut hi[0][..n];
            let get = |s: Src, j: usize| match s {
                Src::Reg(r) => lo[r][j],
                Src::K(k) => k,
            };
            match op {
                EOp::Add => (0..n).for_each(|j| o[j] = get(a, j) + get(b, j)),
                EOp::Sub => (0..n).for_each(|j| o[j] = get(a, j) - get(b, j)),
                EOp::Mul => (0..n).for_each(|j| o[j] = get(a, j) * get(b, j)),
                EOp::Div => (0..n).for_each(|j| o[j] = get(a, j) / get(b, j)),
                EOp::Pow => (0..n).for_each(|j| {
                    let (x, y) = (get(a, j), get(b, j));
                    o[j] = if y == 2.0 { x * x } else { x.powf(y) }
                }),
                EOp::Relu => (0..n).for_each(|j| o[j] = get(a, j).max(0.0)),
                EOp::Sigmoid => (0..n).for_each(|j| o[j] = sigmoid(get(a, j))),
                EOp::Tanh => (0..n).for_each(|j| o[j] = get(a, j).tanh()),
                EOp::Erf => (0..n).for_each(|j| o[j] = erf(get(a, j))),
                EOp::Sqrt => (0..n).for_each(|j| o[j] = get(a, j).sqrt()),
                EOp::Neg => (0..n).for_each(|j| o[j] = -get(a, j)),
                EOp::Exp => (0..n).for_each(|j| o[j] = exp(get(a, j))),
            }
        }
        v.copy_from_slice(&regs[self.ins.len()][..n]);
    }
}

/// erf(x): erfc 의 Chebyshev 근사(Numerical Recipes erfcc, 상대 오차 < 1.2e-7) 를 f32 로, |x| < 0.5 는 테일러 급수
#[inline(always)]
pub fn erf(x: f32) -> f32 {
    let z = x.abs();
    // 작은 |x|: 테일러 급수, 그 밖: erfc 근사. 둘 다 계산해 고름(분기 없이 벡터화)
    let z2 = z * z;
    let small = z * (1.0 - z2 * (1.0 / 3.0 - z2 * (0.1 - z2 * (1.0 / 42.0 - z2 * (1.0 / 216.0 - z2 / 1320.0))))) * std::f32::consts::FRAC_2_SQRT_PI;
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * exp(-z2 - 1.265_512_2
        + t * (1.000_023_7 + t * (0.374_091_96 + t * (0.096_784_18 + t * (-0.186_288_06 + t * (0.278_868_07 + t * (-1.135_204 + t * (1.488_515_9 + t * (-0.822_152_2 + t * 0.170_872_77)))))))));
    let e = if z < 0.5 { small } else { 1.0 - r };
    e.copysign(x)
}

/// e^z: Cephes 단정밀 다항식(상대 오차 ~1e-7), 평범한 산술이라 반복이 벡터화됨
#[inline(always)]
pub fn exp(z: f32) -> f32 {
    let z = z.clamp(-87.0, 88.0);
    let t = z * std::f32::consts::LOG2_E + 12_582_912.0;
    let n = t - 12_582_912.0;
    let r = z - n * 0.693_359_4 + n * 2.121_944_4e-4;
    let mut p = 1.987_569_1e-4f32;
    p = p * r + 1.398_199_9e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 1.666_666_5e-1;
    p = p * r + 5.000_000_1e-1;
    let y = p * r * r + r + 1.0;
    y * f32::from_bits(((t.to_bits() as i32 - 0x4B40_0000 + 127) as u32) << 23)
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

pub(crate) enum Pre {
    None,
    /// MatMul/Gemm 의 상수 B: 판들, k, n, 열 편향
    MatB { panels: Vec<f32>, k: usize, n: usize, bias: Option<Vec<f32>> },
    /// Conv(group 1) 무게를 줄 묶음으로
    ConvA(Vec<f32>),
    /// ConvTranspose: [Co·kh·kw, C] 를 줄 묶음으로
    ConvTA(Vec<f32>),
}

pub(crate) struct Node {
    pub op: String,
    pub ins: Vec<Option<usize>>,
    pub outs: Vec<usize>,
    pub attrs: HashMap<String, Attr>,
    pub act: Act,
    pub pre: Pre,
}

impl Node {
    pub fn ai(&self, n: &str, d: i64) -> i64 {
        match self.attrs.get(n) {
            Some(Attr::I(v)) => *v,
            _ => d,
        }
    }
    pub fn af(&self, n: &str, d: f32) -> f32 {
        match self.attrs.get(n) {
            Some(Attr::F(v)) => *v,
            _ => d,
        }
    }
    pub fn astr(&self, n: &str, d: &str) -> String {
        match self.attrs.get(n) {
            Some(Attr::S(v)) => v.clone(),
            _ => d.to_string(),
        }
    }
    pub fn ais(&self, n: &str) -> Option<Vec<i64>> {
        match self.attrs.get(n) {
            Some(Attr::Is(v)) => Some(v.clone()),
            _ => None,
        }
    }
}

pub struct Model {
    nodes: Vec<Node>,
    consts: Vec<Option<Tensor>>,
    /// 이름, 칸, 모양(-1 = 미정), u8 입력인가
    inputs: Vec<(String, usize, Vec<i64>, bool)>,
    outputs: Vec<(String, usize)>,
    free_after: Vec<Vec<usize>>,
}

impl Model {
    pub fn load(path: &str) -> Result<Model, String> {
        let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        Model::from_bytes(&b).map_err(|e| format!("{path}: {e}"))
    }

    pub fn from_bytes(b: &[u8]) -> Result<Model, String> {
        let g = proto::model(b)?;
        let mut slot: HashMap<String, usize> = HashMap::new();
        let id = |n: &str, slot: &mut HashMap<String, usize>| -> usize {
            let l = slot.len();
            *slot.entry(n.to_string()).or_insert(l)
        };
        let mut consts: Vec<Option<Tensor>> = vec![];
        let set = |consts: &mut Vec<Option<Tensor>>, s: usize, t: Tensor| {
            if consts.len() <= s {
                consts.resize(s + 1, None);
            }
            consts[s] = Some(t);
        };
        for (n, t) in g.inits {
            let s = id(&n, &mut slot);
            set(&mut consts, s, t);
        }
        let mut inputs: Vec<(String, usize, Vec<i64>, bool)> = vec![];
        for (n, s, et) in &g.inputs {
            if slot.contains_key(n) {
                continue; // 초기값이 있는 입력은 상수
            }
            inputs.push((n.clone(), id(n, &mut slot), s.clone(), *et == 2));
        }
        let outputs: Vec<(String, usize)> = g.outputs.iter().map(|n| (n.clone(), id(n, &mut slot))).collect();
        let mut nodes = vec![];
        for rn in g.nodes {
            let ins: Vec<Option<usize>> = rn.ins.iter().map(|n| if n.is_empty() { None } else { Some(id(n, &mut slot)) }).collect();
            let outs: Vec<usize> = rn.outs.iter().map(|n| id(n, &mut slot)).collect();
            let node = Node { op: rn.op, ins, outs, attrs: rn.attrs.into_iter().collect(), act: Act::None, pre: Pre::None };
            consts.resize(slot.len(), None);
            if node.op == "Constant" {
                let t = match node.attrs.iter().next().map(|a| a.1.clone()) {
                    Some(Attr::T(t)) => t,
                    Some(Attr::F(f)) => Tensor::f(vec![], vec![f]),
                    Some(Attr::Fs(f)) => Tensor::f(vec![f.len()], f),
                    Some(Attr::I(i)) => Tensor::i(vec![], vec![i]),
                    Some(Attr::Is(i)) => Tensor::i(vec![i.len()], i),
                    _ => return Err("Constant 값 없음".into()),
                };
                consts[node.outs[0]] = Some(t);
                continue;
            }
            // 입력이 모두 상수면 지금 계산(상수 접기)
            if node.ins.iter().flatten().all(|&s| consts[s].is_some()) && !node.ins.is_empty() {
                let ins: Vec<Option<&Tensor>> = node.ins.iter().map(|s| s.map(|s| consts[s].as_ref().unwrap())).collect();
                let outs = ops::exec(&node, &ins).map_err(|e| format!("{} 접기: {e}", node.op))?;
                for (s, t) in node.outs.iter().zip(outs) {
                    consts[*s] = Some(t);
                }
                continue;
            }
            nodes.push(node);
        }
        consts.resize(slot.len(), None);
        let mut m = Model { nodes, consts, inputs, outputs, free_after: vec![] };
        m.fuse();
        m.fuse_ln();
        m.prepack()?;
        m.fuse_progs();
        m.plan();
        Ok(m)
    }

    fn uses(&self) -> HashMap<usize, Vec<usize>> {
        let mut u: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, n) in self.nodes.iter().enumerate() {
            for s in n.ins.iter().flatten() {
                u.entry(*s).or_default().push(i);
            }
        }
        u
    }

    fn is_out(&self, s: usize) -> bool {
        self.outputs.iter().any(|o| o.1 == s)
    }

    fn fuse(&mut self) {
        let mut dead = vec![false; self.nodes.len()];
        let u = self.uses();
        for i in 0..self.nodes.len() {
            if dead[i] {
                continue;
            }
            let op = self.nodes[i].op.clone();
            let x = self.nodes[i].outs[0];
            if self.is_out(x) {
                continue;
            }
            let cons = u.get(&x).cloned().unwrap_or_default();
            if op == "Conv" || op == "ConvTranspose" {
                if cons.len() == 1 && self.nodes[cons[0]].op == "Relu" {
                    let r = cons[0];
                    self.nodes[i].act = Act::Relu;
                    self.nodes[i].outs[0] = self.nodes[r].outs[0];
                    dead[r] = true;
                } else if cons.len() == 2 && op == "Conv" {
                    let (a, b) = (cons[0], cons[1]);
                    let (sg, mu) = if self.nodes[a].op == "Sigmoid" { (a, b) } else { (b, a) };
                    if self.nodes[sg].op == "Sigmoid" && self.nodes[mu].op == "Mul" {
                        let s = self.nodes[sg].outs[0];
                        let mi = &self.nodes[mu].ins;
                        let su = u.get(&s).map(|v| v.len()).unwrap_or(0);
                        if !self.is_out(s) && su == 1 && (mi[..] == [Some(x), Some(s)] || mi[..] == [Some(s), Some(x)]) {
                            self.nodes[i].act = Act::Silu;
                            self.nodes[i].outs[0] = self.nodes[mu].outs[0];
                            dead[sg] = true;
                            dead[mu] = true;
                        }
                    }
                }
            }
        }
        let mut k = 0;
        self.nodes.retain(|_| {
            k += 1;
            !dead[k - 1]
        });
    }

    fn kscalar(&self, s: Option<usize>) -> Option<f32> {
        let t = self.consts[s?].as_ref()?;
        if t.len() == 1 { Some(t.clone().into_f()[0]) } else { None }
    }

    fn drop_dead(&mut self, dead: &[bool]) {
        let mut k = 0;
        self.nodes.retain(|_| {
            k += 1;
            !dead[k - 1]
        });
    }

    /// ReduceMean · Sub · Pow 2 · ReduceMean · Add eps · Sqrt · Div (· Mul γ · Add β) → LayerNorm 하나
    fn fuse_ln(&mut self) {
        let u = self.uses();
        let mut dead = vec![false; self.nodes.len()];
        let one = |s: usize, dead: &Vec<bool>| -> Option<usize> {
            let c = u.get(&s)?;
            if c.len() == 1 && !self.is_out(s) && !dead[c[0]] { Some(c[0]) } else { None }
        };
        let mut reps = vec![];
        for i in 0..self.nodes.len() {
            let n = &self.nodes[i];
            if n.op != "ReduceMean" || dead[i] || n.ai("keepdims", 1) == 0 || n.ais("axes") != Some(vec![-1]) {
                continue;
            }
            let x = n.ins[0];
            let Some(sub) = one(n.outs[0], &dead) else { continue };
            let sn = &self.nodes[sub];
            if sn.op != "Sub" || sn.ins[0] != x {
                continue;
            }
            let d = sn.outs[0];
            let Some(c) = u.get(&d) else { continue };
            if c.len() != 2 || self.is_out(d) {
                continue;
            }
            let (pw, dv) = if self.nodes[c[0]].op == "Pow" { (c[0], c[1]) } else { (c[1], c[0]) };
            if self.nodes[pw].op != "Pow" || self.kscalar(self.nodes[pw].ins[1]) != Some(2.0) || self.nodes[dv].op != "Div" || self.nodes[dv].ins[0] != Some(d) {
                continue;
            }
            let Some(rm) = one(self.nodes[pw].outs[0], &dead) else { continue };
            if self.nodes[rm].op != "ReduceMean" || self.nodes[rm].ais("axes") != Some(vec![-1]) {
                continue;
            }
            let Some(ad) = one(self.nodes[rm].outs[0], &dead) else { continue };
            if self.nodes[ad].op != "Add" {
                continue;
            }
            let Some(eps) = self.kscalar(self.nodes[ad].ins[1]).or(self.kscalar(self.nodes[ad].ins[0])) else { continue };
            let Some(sq) = one(self.nodes[ad].outs[0], &dead) else { continue };
            if self.nodes[sq].op != "Sqrt" || one(self.nodes[sq].outs[0], &dead) != Some(dv) {
                continue;
            }
            let mut used = vec![sub, pw, rm, ad, sq, dv];
            let mut out = self.nodes[dv].outs[0];
            let (mut g, mut b) = (None, None);
            let vec_const = |s: Option<usize>| s.filter(|&s| self.consts[s].as_ref().is_some_and(|t| t.shape.len() == 1));
            if let Some(m) = one(out, &dead) {
                let mn = &self.nodes[m];
                if mn.op == "Mul" {
                    if let Some(gs) = vec_const(if mn.ins[0] == Some(out) { mn.ins[1] } else { mn.ins[0] }) {
                        g = Some(gs);
                        used.push(m);
                        out = mn.outs[0];
                        if let Some(a) = one(out, &dead) {
                            let an = &self.nodes[a];
                            if an.op == "Add" {
                                if let Some(bs) = vec_const(if an.ins[0] == Some(out) { an.ins[1] } else { an.ins[0] }) {
                                    b = Some(bs);
                                    used.push(a);
                                    out = an.outs[0];
                                }
                            }
                        }
                    }
                }
            }
            for &k in &used {
                dead[k] = true;
            }
            reps.push((i, x, g, b, out, eps));
        }
        for (i, x, g, b, out, eps) in reps {
            let n = &mut self.nodes[i];
            n.op = "LayerNorm".into();
            n.ins = vec![x, g, b];
            n.outs = vec![out];
            n.attrs.insert("epsilon".into(), Attr::F(eps));
        }
        self.drop_dead(&dead);
    }

    /// 행렬곱 · 합성곱 출력에 바로 이어지는 원소별 노드들(다른 입력은 상수 스칼라)을 출력 쓰기 때 한 번에 계산
    fn fuse_progs(&mut self) {
        let u = self.uses();
        let mut dead = vec![false; self.nodes.len()];
        for i in 0..self.nodes.len() {
            let n = &self.nodes[i];
            let heavy = matches!(n.pre, Pre::MatB { .. } | Pre::ConvA(_) | Pre::ConvTA(_)) || (n.op == "Conv" && n.ai("group", 1) > 1);
            if !heavy || dead[i] || !matches!(n.act, Act::None) {
                continue;
            }
            let x = n.outs[0];
            if self.is_out(x) {
                continue;
            }
            // 키우기: 상수 아닌 입력이 모두 이미 든 값인 원소별 노드
            let mut reg: HashMap<usize, usize> = HashMap::from([(x, 0)]);
            let mut ins: Vec<(EOp, Src, Src)> = vec![];
            let mut set: Vec<usize> = vec![];
            let mut grew = true;
            while grew {
                grew = false;
                let mut cand: Vec<usize> = reg.keys().flat_map(|v| u.get(v).cloned().unwrap_or_default()).collect();
                cand.sort();
                cand.dedup();
                for j in cand {
                    if set.contains(&j) || dead[j] {
                        continue;
                    }
                    let m = &self.nodes[j];
                    let op = match m.op.as_str() {
                        "Add" => EOp::Add,
                        "Sub" => EOp::Sub,
                        "Mul" => EOp::Mul,
                        "Div" => EOp::Div,
                        "Pow" => EOp::Pow,
                        "Relu" => EOp::Relu,
                        "Sigmoid" => EOp::Sigmoid,
                        "Tanh" => EOp::Tanh,
                        "Erf" => EOp::Erf,
                        "Sqrt" => EOp::Sqrt,
                        "Neg" => EOp::Neg,
                        "Exp" => EOp::Exp,
                        _ => continue,
                    };
                    let mut srcs = vec![];
                    let mut ok = true;
                    for s in &m.ins {
                        match s {
                            Some(s) if reg.contains_key(s) => srcs.push(Src::Reg(reg[s])),
                            s => match self.kscalar(*s) {
                                Some(k) => srcs.push(Src::K(k)),
                                None => ok = false,
                            },
                        }
                    }
                    if !ok || srcs.is_empty() || !srcs.iter().any(|s| matches!(s, Src::Reg(_))) {
                        continue;
                    }
                    let b = srcs.get(1).copied().unwrap_or(Src::K(0.0));
                    ins.push((op, srcs[0], b));
                    reg.insert(m.outs[0], ins.len());
                    set.push(j);
                    grew = true;
                }
            }
            if set.is_empty() {
                continue;
            }
            // 밖에서 쓰이는 값이 마지막 하나뿐이어야 함
            let outside = |v: usize| self.is_out(v) || u.get(&v).is_some_and(|c| c.iter().any(|j| !set.contains(j)));
            let used: Vec<usize> = reg.keys().copied().filter(|&v| outside(v)).collect();
            if used.len() != 1 || reg[&used[0]] != ins.len() {
                if std::env::var_os("NN_DEBUG").is_some() {
                    eprintln!("묶기 실패 {}: {:?} used {:?}", n.op, set.iter().map(|&j| self.nodes[j].op.as_str()).collect::<Vec<_>>(), used.iter().map(|v| reg[v]).collect::<Vec<_>>());
                }
                continue;
            }
            let last = used[0];
            for &j in &set {
                dead[j] = true;
            }
            self.nodes[i].act = Act::Prog(std::sync::Arc::new(Prog { ins }));
            self.nodes[i].outs[0] = last;
        }
        self.drop_dead(&dead);
    }

    fn prepack(&mut self) -> Result<(), String> {
        let u = self.uses();
        let mut dead = vec![false; self.nodes.len()];
        for i in 0..self.nodes.len() {
            let n = &self.nodes[i];
            let c = |s: Option<usize>| s.and_then(|s| self.consts[s].as_ref());
            match n.op.as_str() {
                "MatMul" => {
                    let Some(b) = c(n.ins[1]) else { continue };
                    if b.shape.len() != 2 {
                        continue;
                    }
                    let (k, nn) = (b.shape[0], b.shape[1]);
                    let panels = gemm::pack_b(b.fv()?, k, nn, nn);
                    // 하나뿐인 다음 노드가 상수 [n] 더하기면 편향으로 묶음
                    let x = n.outs[0];
                    let mut bias = None;
                    let cons = u.get(&x).cloned().unwrap_or_default();
                    if !self.is_out(x) && cons.len() == 1 && self.nodes[cons[0]].op == "Add" {
                        let a = &self.nodes[cons[0]];
                        let other = if a.ins[0] == Some(x) { a.ins[1] } else { a.ins[0] };
                        if let Some(t) = c(other) {
                            if t.len() == nn && t.shape.last() == Some(&nn) && t.shape.len() <= 2 {
                                bias = Some(t.fv()?.to_vec());
                                dead[cons[0]] = true;
                                let out = a.outs[0];
                                self.nodes[i].outs[0] = out;
                            }
                        }
                    }
                    self.nodes[i].pre = Pre::MatB { panels, k, n: nn, bias };
                }
                "Gemm" => {
                    let Some(b) = c(n.ins[1]) else { continue };
                    if n.ai("transA", 0) != 0 || n.af("alpha", 1.0) != 1.0 || n.af("beta", 1.0) != 1.0 {
                        continue;
                    }
                    let (panels, k, nn) = if n.ai("transB", 0) != 0 {
                        (gemm::pack_bt(b.fv()?, b.shape[1], b.shape[0]), b.shape[1], b.shape[0])
                    } else {
                        (gemm::pack_b(b.fv()?, b.shape[0], b.shape[1], b.shape[1]), b.shape[0], b.shape[1])
                    };
                    let bias = match n.ins.get(2).copied().flatten().and_then(|s| self.consts[s].as_ref()) {
                        Some(t) if t.len() == nn => Some(t.fv()?.to_vec()),
                        Some(_) => continue,
                        None => None,
                    };
                    self.nodes[i].pre = Pre::MatB { panels, k, n: nn, bias };
                }
                "Conv" => {
                    let Some(w) = c(n.ins[1]) else { continue };
                    if n.ai("group", 1) == 1 {
                        let co = w.shape[0];
                        let kk = w.len() / co;
                        self.nodes[i].pre = Pre::ConvA(gemm::pack_a(w.fv()?, co, kk, kk));
                    }
                }
                "ConvTranspose" => {
                    let Some(w) = c(n.ins[1]) else { continue };
                    let (ci, co, kh, kw) = (w.shape[0], w.shape[1], w.shape[2], w.shape[3]);
                    let wv = w.fv()?;
                    let mut a = vec![0f32; co * kh * kw * ci];
                    for c_ in 0..ci {
                        for o in 0..co * kh * kw {
                            a[o * ci + c_] = wv[c_ * co * kh * kw + o];
                        }
                    }
                    self.nodes[i].pre = Pre::ConvTA(gemm::pack_a(&a, co * kh * kw, ci, ci));
                }
                _ => {}
            }
        }
        let mut k = 0;
        self.nodes.retain(|_| {
            k += 1;
            !dead[k - 1]
        });
        Ok(())
    }

    fn plan(&mut self) {
        let mut last: HashMap<usize, usize> = HashMap::new();
        for (i, n) in self.nodes.iter().enumerate() {
            for s in n.ins.iter().flatten() {
                last.insert(*s, i);
            }
        }
        self.free_after = vec![vec![]; self.nodes.len()];
        for (s, i) in last {
            if self.consts[s].is_none() && !self.is_out(s) {
                self.free_after[i].push(s);
            }
        }
    }

    /// 입력 이름 · 모양(미정 = -1) · u8 인가
    pub fn inputs(&self) -> impl Iterator<Item = (&str, &[i64], bool)> {
        self.inputs.iter().map(|i| (i.0.as_str(), &i.2[..], i.3))
    }
    pub fn output_names(&self) -> impl Iterator<Item = &str> {
        self.outputs.iter().map(|o| o.0.as_str())
    }

    /// 입력(그래프 입력 차례) → 출력(그래프 출력 차례)
    pub fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, String> {
        let prof = std::env::var_os("NN_PROF").is_some();
        let mut tm: HashMap<String, f64> = HashMap::new();
        let mut vals: Vec<Option<Tensor>> = vec![None; self.consts.len()];
        if inputs.len() != self.inputs.len() {
            return Err(format!("입력 {} 개, 모델은 {}", inputs.len(), self.inputs.len()));
        }
        for (t, i) in inputs.into_iter().zip(&self.inputs) {
            vals[i.1] = Some(t);
        }
        for (ni, n) in self.nodes.iter().enumerate() {
            let t0 = std::time::Instant::now();
            let outs = {
                let ins: Vec<Option<&Tensor>> = n
                    .ins
                    .iter()
                    .map(|s| s.and_then(|s| self.consts[s].as_ref().or(vals[s].as_ref())))
                    .collect();
                if let Some(p) = n.ins.iter().zip(&ins).position(|(s, t)| s.is_some() && t.is_none()) {
                    return Err(format!("{}: 입력 {p} 이 아직 없음", n.op));
                }
                ops::exec(n, &ins).map_err(|e| format!("{}: {e}", n.op))?
            };
            for (s, t) in n.outs.iter().zip(outs) {
                vals[*s] = Some(t);
            }
            for &s in &self.free_after[ni] {
                if let Some(Tensor { data: Data::F(v), .. }) = vals[s].take() {
                    buf::give(v);
                }
            }
            if prof {
                if std::env::var("NN_PROF").as_deref() == Ok("2") {
                    eprintln!("  {} {:?} {:.3} ms", n.op, vals[n.outs[0]].as_ref().map(|t| t.shape.clone()), t0.elapsed().as_secs_f64() * 1e3);
                }
                *tm.entry(n.op.clone()).or_default() += t0.elapsed().as_secs_f64();
            }
        }
        if prof {
            let mut v: Vec<_> = tm.into_iter().collect();
            v.sort_by(|a, b| b.1.total_cmp(&a.1));
            eprintln!("nn prof ms: {}", v.iter().map(|(k, t)| format!("{k} {:.2}", t * 1e3)).collect::<Vec<_>>().join(", "));
        }
        self.outputs
            .iter()
            .map(|o| self.consts[o.1].clone().or_else(|| vals[o.1].take()).ok_or(format!("출력 {} 없음", o.0)))
            .collect()
    }
}
