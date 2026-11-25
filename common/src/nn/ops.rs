//! 노드 하나 실행
use super::gemm::{self, Panels, RowMajor};
use super::{Act, Data, Node, Pre, Tensor};
use super::{buf, pool, simd, vmath};

fn copy(v: &[f32]) -> Vec<f32> {
    let mut o = buf::take::<f32>(v.len());
    o.copy_from_slice(v);
    o
}

type R = Result<Vec<Tensor>, String>;

const PAR: usize = 1 << 15;

pub(crate) fn need<'a>(ins: &[Option<&'a Tensor>], i: usize) -> Result<&'a Tensor, String> {
    ins.get(i).copied().flatten().ok_or(format!("입력 {i} 없음"))
}

fn axis(a: i64, r: usize) -> usize {
    if a < 0 { (a + r as i64) as usize } else { a as usize }
}

fn strides(s: &[usize]) -> Vec<usize> {
    let mut st = vec![1; s.len()];
    for i in (0..s.len().saturating_sub(1)).rev() {
        st[i] = st[i + 1] * s[i + 1];
    }
    st
}

pub(crate) fn exec(n: &Node, ins: &[Option<&Tensor>]) -> R {
    let x = || need(ins, 0);
    match n.op.as_str() {
        "Add" | "Sub" | "Mul" | "Div" if need(ins, 0)?.blk > 0 || need(ins, 1)?.blk > 0 => super::nchwc::binary(&n.op, need(ins, 0)?, need(ins, 1)?),
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Mod" => binary(&n.op, need(ins, 0)?, need(ins, 1)?, n.ai("fmod", 0) != 0),
        "Relu" => unary_s(x()?, vmath::relu),
        "Sigmoid" => unary_s(x()?, vmath::sigmoid_s),
        "Tanh" => unary_s(x()?, vmath::tanh_s),
        "Sqrt" => unary_s(x()?, vmath::sqrt_s),
        "Erf" => unary_s(x()?, vmath::erf_s),
        "Neg" => unary_s(x()?, vmath::neg_s),
        "Exp" => unary_s(x()?, vmath::exp_s),
        "Identity" => Ok(vec![x()?.clone()]),
        "Cast" => {
            let t = x()?;
            Ok(vec![match n.ai("to", 1) {
                1 | 10 | 11 => Tensor::f(t.shape.clone(), t.clone().into_f()),
                9 => Tensor::i(
                    t.shape.clone(),
                    match &t.data {
                        Data::F(v) => v.iter().map(|&a| (a != 0.0) as i64).collect(),
                        Data::I(v) => v.iter().map(|&a| (a != 0) as i64).collect(),
                    },
                ),
                _ => Tensor::i(t.shape.clone(), t.ints()),
            }])
        }
        "Shape" => {
            let t = x()?;
            Ok(vec![Tensor::i(vec![t.shape.len()], t.shape.iter().map(|&d| d as i64).collect())])
        }
        "Reshape" => {
            let t = x()?;
            let s = need(ins, 1)?.ints();
            let total = t.len();
            let mut out: Vec<usize> = s.iter().enumerate().map(|(i, &d)| if d == 0 && n.ai("allowzero", 0) == 0 { t.shape[i] } else { d.max(0) as usize }).collect();
            if let Some(p) = s.iter().position(|&d| d == -1) {
                let known: usize = out.iter().enumerate().filter(|(i, _)| *i != p).map(|(_, d)| *d).product();
                out[p] = total / known.max(1);
            }
            Ok(vec![Tensor { shape: out, data: t.data.clone(), blk: 0 }])
        }
        "Flatten" => {
            let t = x()?;
            let a = axis(n.ai("axis", 1), t.shape.len());
            let o: usize = t.shape[..a].iter().product();
            Ok(vec![Tensor { shape: vec![o, t.len() / o.max(1)], data: t.data.clone(), blk: 0 }])
        }
        "Unsqueeze" => {
            let t = x()?;
            let ax = ins.get(1).copied().flatten().map(|a| a.ints()).or(n.ais("axes")).ok_or("axes 없음")?;
            let r = t.shape.len() + ax.len();
            let mut ax: Vec<usize> = ax.iter().map(|&a| axis(a, r)).collect();
            ax.sort();
            let mut s = t.shape.clone();
            for a in ax {
                s.insert(a, 1);
            }
            Ok(vec![Tensor { shape: s, data: t.data.clone(), blk: 0 }])
        }
        "Squeeze" => {
            let t = x()?;
            let ax = ins.get(1).copied().flatten().map(|a| a.ints()).or(n.ais("axes"));
            let r = t.shape.len();
            let s: Vec<usize> = match ax {
                Some(ax) => {
                    let ax: Vec<usize> = ax.iter().map(|&a| axis(a, r)).collect();
                    t.shape.iter().enumerate().filter(|(i, _)| !ax.contains(i)).map(|(_, d)| *d).collect()
                }
                None => t.shape.iter().copied().filter(|&d| d != 1).collect(),
            };
            Ok(vec![Tensor { shape: s, data: t.data.clone(), blk: 0 }])
        }
        "Transpose" => {
            let t = x()?;
            let perm: Vec<usize> = n.ais("perm").map(|p| p.iter().map(|&a| a as usize).collect()).unwrap_or_else(|| (0..t.shape.len()).rev().collect());
            Ok(vec![transpose(t, &perm)])
        }
        "Concat" => {
            let ts: Vec<&Tensor> = ins.iter().flatten().copied().collect();
            if ts[0].blk > 0 {
                return super::nchwc::concat(&ts);
            }
            let r = ts[0].shape.len();
            let a = axis(n.ai("axis", 0), r);
            concat(&ts, a)
        }
        "Split" => {
            let t = x()?;
            let a = axis(n.ai("axis", 0), t.shape.len());
            let k = n.outs.len();
            let sizes: Vec<usize> = match ins.get(1).copied().flatten().map(|s| s.ints()).or(n.ais("split")) {
                Some(s) => s.iter().map(|&v| v as usize).collect(),
                None => {
                    let d = t.shape[a];
                    let c = d.div_ceil(k);
                    (0..k).map(|i| c.min(d - (c * i).min(d))).collect()
                }
            };
            let keep = n.ais("keep");
            if t.blk > 0 {
                return super::nchwc::split(t, &sizes, keep.as_deref());
            }
            split(t, a, &sizes, keep.as_deref())
        }
        "Slice" => {
            let t = x()?;
            let st = need(ins, 1)?.ints();
            let en = need(ins, 2)?.ints();
            let r = t.shape.len();
            let ax: Vec<usize> = ins.get(3).copied().flatten().map(|a| a.ints().iter().map(|&a| axis(a, r)).collect()).unwrap_or_else(|| (0..st.len()).collect());
            let sp = ins.get(4).copied().flatten().map(|a| a.ints()).unwrap_or_else(|| vec![1; st.len()]);
            slice(t, &st, &en, &ax, &sp)
        }
        "Gather" => {
            let t = x()?;
            let idx = need(ins, 1)?;
            let a = axis(n.ai("axis", 0), t.shape.len());
            gather(t, idx, a)
        }
        "GatherElements" => {
            let t = x()?;
            let idx = need(ins, 1)?;
            let a = axis(n.ai("axis", 0), t.shape.len());
            gather_elements(t, idx, a)
        }
        "Expand" => {
            let t = x()?;
            let s: Vec<usize> = need(ins, 1)?.ints().iter().map(|&d| d as usize).collect();
            let shape = bshape(&t.shape, &s)?;
            Ok(vec![broadcast_to(t, &shape)])
        }
        "Tile" => {
            let t = x()?;
            let rep: Vec<usize> = need(ins, 1)?.ints().iter().map(|&d| d as usize).collect();
            let mut cur = t.clone();
            for (a, &k) in rep.iter().enumerate() {
                if k != 1 {
                    let v = vec![&cur; k];
                    cur = concat(&v, a)?.pop().unwrap();
                }
            }
            Ok(vec![cur])
        }
        "ReduceMean" | "ReduceMax" | "ReduceSum" => {
            let t = x()?;
            let r = t.shape.len();
            let ax = ins.get(1).copied().flatten().map(|a| a.ints()).or(n.ais("axes")).unwrap_or_else(|| (0..r as i64).collect());
            let keep = n.ai("keepdims", 1) != 0;
            reduce(&n.op, t, &ax, keep)
        }
        "Softmax" => {
            let t = x()?;
            let a = axis(n.ai("axis", -1), t.shape.len());
            if a != t.shape.len() - 1 {
                return Err("Softmax 는 끝 축만".into());
            }
            let d = t.shape[a];
            let mut v = copy(t.fv()?);
            pool::chunks_mut(&mut v, d * (PAR / d.max(1)).max(1), |_, rows| rows.chunks_mut(d).for_each(vmath::softmax_row));
            Ok(vec![Tensor::f(t.shape.clone(), v)])
        }
        "TopK" => {
            let t = x()?;
            let k = need(ins, 1)?.ints()[0] as usize;
            let a = axis(n.ai("axis", -1), t.shape.len());
            topk(t, k, a, n.ai("largest", 1) != 0)
        }
        "LayerNorm" => {
            let t = x()?;
            let d = *t.shape.last().unwrap();
            let eps = n.af("epsilon", 1e-5);
            let g = ins.get(1).copied().flatten().map(|t| t.fv()).transpose()?;
            let b = ins.get(2).copied().flatten().map(|t| t.fv()).transpose()?;
            let mut v = copy(t.fv()?);
            pool::chunks_mut(&mut v, d * (PAR / d.max(1)).max(1), |_, rows| {
                for row in rows.chunks_mut(d) {
                    let m = row.iter().sum::<f32>() / d as f32;
                    let var = row.iter().map(|a| (a - m) * (a - m)).sum::<f32>() / d as f32;
                    let s = (var + eps).sqrt();
                    for (j, x) in row.iter_mut().enumerate() {
                        let mut y = (*x - m) / s;
                        if let Some(g) = g {
                            y *= g[j];
                        }
                        if let Some(b) = b {
                            y += b[j];
                        }
                        *x = y;
                    }
                }
            });
            Ok(vec![Tensor::f(t.shape.clone(), v)])
        }
        "MatMul" => matmul(n, need(ins, 0)?, need(ins, 1)?),
        "Gemm" => gemm_op(n, ins),
        "Conv" => super::nchwc::conv(n, ins),
        "ConvTranspose" => super::nchwc::conv_t(n, ins),
        "MaxPool" | "AveragePool" => pool(n, x()?),
        "Resize" => resize(n, ins),
        o => Err(format!("연산 {o} 은 아직 없음")),
    }
}

/// 첫 입력을 넘겨 받아 제자리로 계산할 수 있는 노드인가
pub(crate) fn in_place(n: &Node) -> bool {
    matches!(
        n.op.as_str(),
        "Reshape" | "Flatten" | "Squeeze" | "Unsqueeze" | "Identity" | "Relu" | "Sigmoid" | "Tanh" | "Sqrt" | "Erf" | "Neg" | "Exp" | "Add" | "Sub" | "Mul" | "Div" | "Softmax" | "LayerNorm"
    )
}

/// [`in_place`] 노드: x = 첫 입력(넘겨 받음), rest = 나머지 입력
pub(crate) fn exec_own(n: &Node, mut x: Tensor, rest: &[Option<&Tensor>]) -> R {
    match n.op.as_str() {
        "Reshape" | "Flatten" | "Squeeze" | "Unsqueeze" | "Identity" => {
            // 모양만 계산(값은 그대로 옮김)
            let mut ins: Vec<Option<&Tensor>> = vec![None];
            ins.extend_from_slice(rest);
            let probe = Tensor { shape: x.shape.clone(), data: Data::F(vec![]), blk: 0 };
            ins[0] = Some(&probe);
            let shape = shape_only(n, &ins)?;
            x.shape = shape;
            Ok(vec![x])
        }
        "Relu" => f32_un_s(vmath::relu, &mut x).map(|_| vec![x]),
        "Sigmoid" => f32_un_s(vmath::sigmoid_s, &mut x).map(|_| vec![x]),
        "Tanh" => f32_un_s(vmath::tanh_s, &mut x).map(|_| vec![x]),
        "Sqrt" => f32_un_s(vmath::sqrt_s, &mut x).map(|_| vec![x]),
        "Erf" => f32_un_s(vmath::erf_s, &mut x).map(|_| vec![x]),
        "Neg" => f32_un_s(vmath::neg_s, &mut x).map(|_| vec![x]),
        "Exp" => f32_un_s(vmath::exp_s, &mut x).map(|_| vec![x]),
        "Add" | "Sub" | "Mul" | "Div" => {
            let b = need(rest, 0)?;
            let same_out = bshape(&x.shape, &b.shape).map(|s| s == x.shape).unwrap_or(false);
            match (&mut x.data, &b.data) {
                (Data::F(v), Data::F(w)) if same_out => {
                    let shape = x.shape.clone();
                    match n.op.as_str() {
                        "Add" => bin_in_place(v, &shape, w, &b.shape, |p, q| p + q),
                        "Sub" => bin_in_place(v, &shape, w, &b.shape, |p, q| p - q),
                        "Mul" => bin_in_place(v, &shape, w, &b.shape, |p, q| p * q),
                        _ => bin_in_place(v, &shape, w, &b.shape, |p, q| p / q),
                    }
                    Ok(vec![x])
                }
                _ => binary(&n.op, &x, b, false),
            }
        }
        _ => {
            let mut ins: Vec<Option<&Tensor>> = vec![Some(&x)];
            ins.extend_from_slice(rest);
            exec(n, &ins)
        }
    }
}

/// 슬라이스 함수로 제자리(크면 나눠서)
fn f32_un_s(f: fn(&mut [f32]), x: &mut Tensor) -> Result<(), String> {
    let Data::F(v) = &mut x.data else {
        return Err("f32 텐서가 필요".into());
    };
    if v.len() > PAR {
        pool::chunks_mut(v, PAR, |_, c| f(c));
    } else {
        f(v);
    }
    Ok(())
}

fn unary_s(t: &Tensor, f: fn(&mut [f32])) -> R {
    let mut x = Tensor { shape: t.shape.clone(), data: Data::F(copy(t.fv()?)), blk: t.blk };
    f32_un_s(f, &mut x)?;
    Ok(vec![x])
}

/// v(모양 shape) ⊙= w(넓혀 읽음, 결과 모양은 shape)
fn bin_in_place(v: &mut [f32], shape: &[usize], w: &[f32], wsh: &[usize], op: impl Fn(f32, f32) -> f32 + Sync) {
    if w.len() == v.len() {
        if v.len() > PAR {
            pool::chunks_mut(v, PAR, |c, ch| ch.iter_mut().zip(&w[c * PAR..]).for_each(|(a, &q)| *a = op(*a, q)));
        } else {
            v.iter_mut().zip(w.iter()).for_each(|(a, &q)| *a = op(*a, q));
        }
        return;
    }
    if w.len() == 1 {
        let q = w[0];
        if v.len() > PAR {
            pool::chunks_mut(v, PAR, |_, ch| ch.iter_mut().for_each(|a| *a = op(*a, q)));
        } else {
            v.iter_mut().for_each(|a| *a = op(*a, q));
        }
        return;
    }
    let sb = bstrides(wsh, shape);
    let r = shape.len();
    let last = shape[r - 1];
    let lb = sb[r - 1];
    let per = (PAR / last.max(1)).max(1);
    let body = |c: usize, ch: &mut [f32]| {
        for (k, row) in ch.chunks_mut(last).enumerate() {
            let mut q = c * per + k;
            let mut ib = 0;
            for d in (0..r - 1).rev() {
                ib += (q % shape[d]) * sb[d];
                q /= shape[d];
            }
            if lb == 0 {
                let s = w[ib];
                row.iter_mut().for_each(|a| *a = op(*a, s));
            } else {
                row.iter_mut().zip(&w[ib..]).for_each(|(a, &s)| *a = op(*a, s));
            }
        }
    };
    if v.len() > PAR {
        pool::chunks_mut(v, per * last, body);
    } else {
        body(0, v);
    }
}

/// 모양 바꾸기 노드의 출력 모양
fn shape_only(n: &Node, ins: &[Option<&Tensor>]) -> Result<Vec<usize>, String> {
    let t = need(ins, 0)?;
    let total: usize = t.shape.iter().product();
    Ok(match n.op.as_str() {
        "Identity" => t.shape.clone(),
        "Reshape" => {
            let s = need(ins, 1)?.ints();
            let mut out: Vec<usize> = s.iter().enumerate().map(|(i, &d)| if d == 0 && n.ai("allowzero", 0) == 0 { t.shape[i] } else { d.max(0) as usize }).collect();
            if let Some(p) = s.iter().position(|&d| d == -1) {
                let known: usize = out.iter().enumerate().filter(|(i, _)| *i != p).map(|(_, d)| *d).product();
                out[p] = total / known.max(1);
            }
            out
        }
        "Flatten" => {
            let a = axis(n.ai("axis", 1), t.shape.len());
            let o: usize = t.shape[..a].iter().product();
            vec![o, total / o.max(1)]
        }
        "Unsqueeze" => {
            let ax = ins.get(1).copied().flatten().map(|a| a.ints()).or(n.ais("axes")).ok_or("axes 없음")?;
            let r = t.shape.len() + ax.len();
            let mut ax: Vec<usize> = ax.iter().map(|&a| axis(a, r)).collect();
            ax.sort();
            let mut s = t.shape.clone();
            for a in ax {
                s.insert(a, 1);
            }
            s
        }
        _ => {
            let ax = ins.get(1).copied().flatten().map(|a| a.ints()).or(n.ais("axes"));
            let r = t.shape.len();
            match ax {
                Some(ax) => {
                    let ax: Vec<usize> = ax.iter().map(|&a| axis(a, r)).collect();
                    t.shape.iter().enumerate().filter(|(i, _)| !ax.contains(i)).map(|(_, d)| *d).collect()
                }
                None => t.shape.iter().copied().filter(|&d| d != 1).collect(),
            }
        }
    })
}

fn bshape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let r = a.len().max(b.len());
    let mut s = vec![0; r];
    for i in 0..r {
        let da = if i + a.len() >= r { a[i + a.len() - r] } else { 1 };
        let db = if i + b.len() >= r { b[i + b.len() - r] } else { 1 };
        s[i] = if da == db || db == 1 {
            da
        } else if da == 1 {
            db
        } else {
            return Err(format!("모양 안 맞음 {a:?} {b:?}"));
        };
    }
    Ok(s)
}

/// 출력 모양에서 본 입력 보폭(넓혀진 축은 0)
fn bstrides(s: &[usize], out: &[usize]) -> Vec<usize> {
    let st = strides(s);
    let r = out.len();
    (0..r)
        .map(|i| {
            if i + s.len() < r {
                0
            } else {
                let j = i + s.len() - r;
                if s[j] == 1 { 0 } else { st[j] }
            }
        })
        .collect()
}

/// 출력 끝 축 한 줄마다 (a 시작, b 시작) 를 주며 f 호출
fn rows2(out: &[usize], sa: &[usize], sb: &[usize], len: usize, f: impl Fn(usize, usize, usize) + Sync) {
    let r = out.len();
    let last = if r == 0 { 1 } else { out[r - 1] };
    let nrows = len / last.max(1);
    let body = |row: usize| {
        let (mut ia, mut ib, mut q) = (0, 0, row);
        for d in (0..r.saturating_sub(1)).rev() {
            let c = q % out[d];
            q /= out[d];
            ia += c * sa[d];
            ib += c * sb[d];
        }
        f(row, ia, ib);
    };
    if len > PAR {
        let per = (PAR / last.max(1)).max(1);
        pool::for_each(nrows.div_ceil(per), &|c| (c * per..((c + 1) * per).min(nrows)).for_each(&body));
    } else {
        (0..nrows).for_each(body);
    }
}

struct Ptr<T>(*mut T);
unsafe impl<T> Sync for Ptr<T> {}
unsafe impl<T> Send for Ptr<T> {}
impl<T> Ptr<T> {
    #[inline(always)]
    fn at(&self, i: usize) -> *mut T {
        unsafe { self.0.add(i) }
    }
}

fn bin_loop<T: Copy + Send + Sync + Default + 'static>(a: &[T], ash: &[usize], b: &[T], bsh: &[usize], f: impl Fn(T, T) -> T + Sync) -> Result<(Vec<usize>, Vec<T>), String> {
    let out = bshape(ash, bsh)?;
    let len: usize = out.iter().product();
    let mut v = buf::take::<T>(len);
    if a.len() == len && b.len() == len {
        if len > PAR {
            pool::chunks_mut(&mut v, 4096, |c, o| {
                for (i, x) in o.iter_mut().enumerate() {
                    *x = f(a[c * 4096 + i], b[c * 4096 + i]);
                }
            });
        } else {
            for i in 0..len {
                v[i] = f(a[i], b[i]);
            }
        }
        return Ok((out, v));
    }
    if b.len() == 1 {
        let bb = b[0];
        if len > PAR {
            pool::chunks_mut(&mut v, 4096, |c, o| o.iter_mut().enumerate().for_each(|(i, x)| *x = f(a[c * 4096 + i], bb)));
        } else {
            v.iter_mut().zip(a).for_each(|(x, &aa)| *x = f(aa, bb));
        }
        return Ok((out, v));
    }
    let sa = bstrides(ash, &out);
    let sb = bstrides(bsh, &out);
    let r = out.len();
    let last = out[r - 1];
    let (la, lb) = (sa[r - 1], sb[r - 1]);
    let p = Ptr(v.as_mut_ptr());
    rows2(&out, &sa, &sb, len, |row, ia, ib| {
        for j in 0..last {
            unsafe { *p.at(row * last + j) = f(a[ia + j * la], b[ib + j * lb]) };
        }
    });
    Ok((out, v))
}

fn binary(op: &str, a: &Tensor, b: &Tensor, fmod: bool) -> R {
    match (&a.data, &b.data) {
        (Data::F(x), Data::F(y)) => {
            let (s, v) = match op {
                "Add" => bin_loop(x, &a.shape, y, &b.shape, |p, q| p + q)?,
                "Sub" => bin_loop(x, &a.shape, y, &b.shape, |p, q| p - q)?,
                "Mul" => bin_loop(x, &a.shape, y, &b.shape, |p, q| p * q)?,
                "Div" => bin_loop(x, &a.shape, y, &b.shape, |p, q| p / q)?,
                "Pow" => {
                    if y.len() == 1 && y[0] == 2.0 {
                        bin_loop(x, &a.shape, y, &b.shape, |p, _| p * p)?
                    } else {
                        bin_loop(x, &a.shape, y, &b.shape, f32::powf)?
                    }
                }
                _ => {
                    if fmod {
                        bin_loop(x, &a.shape, y, &b.shape, |p, q| p % q)?
                    } else {
                        bin_loop(x, &a.shape, y, &b.shape, |p, q| {
                            let r = p % q;
                            if r != 0.0 && ((r < 0.0) != (q < 0.0)) { r + q } else { r }
                        })?
                    }
                }
            };
            Ok(vec![Tensor::f(s, v)])
        }
        _ => {
            let (x, y) = (a.ints(), b.ints());
            let (s, v) = match op {
                "Add" => bin_loop(&x, &a.shape, &y, &b.shape, |p, q| p + q)?,
                "Sub" => bin_loop(&x, &a.shape, &y, &b.shape, |p, q| p - q)?,
                "Mul" => bin_loop(&x, &a.shape, &y, &b.shape, |p, q| p * q)?,
                "Div" => bin_loop(&x, &a.shape, &y, &b.shape, |p, q| if q == 0 { 0 } else { p / q })?,
                "Pow" => bin_loop(&x, &a.shape, &y, &b.shape, |p, q| p.pow(q.max(0) as u32))?,
                _ => bin_loop(&x, &a.shape, &y, &b.shape, move |p, q| {
                    if q == 0 {
                        return 0;
                    }
                    let r = p % q;
                    if !fmod && r != 0 && ((r < 0) != (q < 0)) { r + q } else { r }
                })?,
            };
            Ok(vec![Tensor::i(s, v)])
        }
    }
}

fn map_data(t: &Tensor, shape: Vec<usize>, idx: impl Fn(usize) -> usize + Sync) -> Tensor {
    let len: usize = shape.iter().product();
    match &t.data {
        Data::F(v) => {
            let o: Vec<f32> = if len > PAR {
                {
                    let mut o = buf::take::<f32>(len);
                    pool::chunks_mut(&mut o, PAR, |c, ch| ch.iter_mut().enumerate().for_each(|(i, x)| *x = v[idx(c * PAR + i)]));
                    o
                }
            } else {
                (0..len).map(|i| v[idx(i)]).collect()
            };
            Tensor::f(shape, o)
        }
        Data::I(v) => Tensor::i(shape, (0..len).map(|i| v[idx(i)]).collect()),
    }
}

/// 출력 번호 → 입력 번호(출력 축마다 입력 보폭)

/// 보폭 복사: 출력[i…] = 입력[base + Σ i_d · ist_d]. 끝 축 한 줄씩(보폭 1 이면 통째 복사), 여러 줄을 한 일로
fn strided<T: Copy + Send + Sync + Default + 'static>(src: &[T], out: &[usize], ist: &[usize], base: usize) -> Vec<T> {
    let len: usize = out.iter().product();
    let mut o = buf::take::<T>(len);
    if len == 0 {
        return o;
    }
    let r = out.len();
    if r == 0 {
        o[0] = src[base];
        return o;
    }
    let last = out[r - 1];
    let ls = ist[r - 1];
    let per = (PAR / last.max(1)).max(1);
    let body = |c: usize, ch: &mut [T]| {
        for (k, row) in ch.chunks_mut(last).enumerate() {
            let mut q = c * per + k;
            let mut s = base;
            for d in (0..r - 1).rev() {
                s += (q % out[d]) * ist[d];
                q /= out[d];
            }
            if ls == 1 {
                row.copy_from_slice(&src[s..s + last]);
            } else {
                for (j, x) in row.iter_mut().enumerate() {
                    *x = src[s + j * ls];
                }
            }
        }
    };
    if len > PAR {
        pool::chunks_mut(&mut o, per * last, body);
    } else {
        body(0, &mut o);
    }
    o
}

fn strided_t(t: &Tensor, out: Vec<usize>, ist: &[usize], base: usize) -> Tensor {
    match &t.data {
        Data::F(v) => Tensor::f(out.clone(), strided(v, &out, ist, base)),
        Data::I(v) => Tensor::i(out.clone(), strided(v, &out, ist, base)),
    }
}

fn transpose(t: &Tensor, perm: &[usize]) -> Tensor {
    let st = strides(&t.shape);
    let out: Vec<usize> = perm.iter().map(|&p| t.shape[p]).collect();
    let ist: Vec<usize> = perm.iter().map(|&p| st[p]).collect();
    strided_t(t, out, &ist, 0)
}

fn concat(ts: &[&Tensor], a: usize) -> R {
    let mut shape = ts[0].shape.clone();
    shape[a] = ts.iter().map(|t| t.shape[a]).sum();
    let outer: usize = shape[..a].iter().product();
    let inner: usize = shape[a + 1..].iter().product();
    let len: usize = shape.iter().product();
    // 복사 목록(출력 자리, 입력 번호, 입력 자리, 길이), 큰 것은 PAR 씩 쪼갬
    let mut cp: Vec<(usize, usize, usize, usize)> = vec![];
    let mut d = 0;
    for i in 0..outer {
        for (k, t) in ts.iter().enumerate() {
            let c = t.shape[a] * inner;
            let mut s = 0;
            while s < c {
                let l = PAR.min(c - s);
                cp.push((d + s, k, i * c + s, l));
                s += l;
            }
            d += c;
        }
    }
    fn go<T: Copy + Send + Sync + Default + 'static>(srcs: &[&[T]], cp: &[(usize, usize, usize, usize)], len: usize) -> Vec<T> {
        let mut o = buf::take::<T>(len);
        let p = Ptr(o.as_mut_ptr());
        let body = |i: usize| {
            let (d, k, s, l) = cp[i];
            // SAFETY: 복사끼리 출력 자리가 겹치지 않음
            unsafe { std::ptr::copy_nonoverlapping(srcs[k].as_ptr().add(s), p.at(d), l) };
        };
        if len > PAR {
            pool::for_each(cp.len(), &body);
        } else {
            (0..cp.len()).for_each(body);
        }
        o
    }
    if ts.iter().all(|t| matches!(t.data, Data::F(_))) {
        let srcs: Vec<&[f32]> = ts.iter().map(|t| t.fv().unwrap()).collect();
        Ok(vec![Tensor::f(shape, go(&srcs, &cp, len))])
    } else {
        let iv: Vec<Vec<i64>> = ts.iter().map(|t| t.ints()).collect();
        let srcs: Vec<&[i64]> = iv.iter().map(|v| &v[..]).collect();
        Ok(vec![Tensor::i(shape, go(&srcs, &cp, len))])
    }
}

/// keep[i] == 0 인 출력은 아무도 안 씀: 빈 텐서
fn split(t: &Tensor, a: usize, sizes: &[usize], keep: Option<&[i64]>) -> R {
    let mut start = 0;
    let mut outs = vec![];
    for (i, &s) in sizes.iter().enumerate() {
        if keep.is_some_and(|k| k[i] == 0) {
            outs.push(Tensor::f(vec![0], vec![]));
        } else {
            outs.push(slice(t, &[start as i64], &[(start + s) as i64], &[a], &[1])?.pop().unwrap());
        }
        start += s;
    }
    Ok(outs)
}

fn slice(t: &Tensor, st: &[i64], en: &[i64], ax: &[usize], sp: &[i64]) -> R {
    let r = t.shape.len();
    let mut start = vec![0i64; r];
    let mut step = vec![1i64; r];
    let mut out = t.shape.clone();
    for (i, &a) in ax.iter().enumerate() {
        let d = t.shape[a] as i64;
        let s = sp[i];
        let (mut b, mut e) = (st[i], en[i]);
        if b < 0 {
            b += d;
        }
        if e < 0 {
            e += d;
        }
        if s > 0 {
            b = b.clamp(0, d);
            e = e.clamp(0, d);
            out[a] = if e > b { ((e - b + s - 1) / s) as usize } else { 0 };
        } else {
            b = b.clamp(-1, d - 1);
            e = e.clamp(-1, d - 1);
            out[a] = if b > e { ((b - e - s - 1) / -s) as usize } else { 0 };
        }
        start[a] = b;
        step[a] = s;
    }
    let ist0 = strides(&t.shape);
    let base: i64 = (0..r).map(|d| start[d] * ist0[d] as i64).sum();
    // 음수 보폭은 없다고 보고(이 모델들) 양수만 처리
    if step.iter().any(|&s| s < 0) {
        return Err("음수 보폭 Slice 없음".into());
    }
    let ist: Vec<usize> = (0..r).map(|d| ist0[d] * step[d] as usize).collect();
    Ok(vec![strided_t(t, out, &ist, base as usize)])
}

fn gather(t: &Tensor, idx: &Tensor, a: usize) -> R {
    let iv: Vec<usize> = idx.ints().iter().map(|&i| if i < 0 { (i + t.shape[a] as i64) as usize } else { i as usize }).collect();
    let inner: usize = t.shape[a + 1..].iter().product();
    let d = t.shape[a];
    let mut shape: Vec<usize> = t.shape[..a].to_vec();
    shape.extend(&idx.shape);
    shape.extend(&t.shape[a + 1..]);
    let ni = iv.len();
    // 줄(바깥 o, 번호 k) 마다 inner 개 연속 복사 = 보폭 복사 [바깥, 번호, inner] 를 번호별로
    fn go<T: Copy + Send + Sync + Default + 'static>(v: &[T], iv: &[usize], ni: usize, d: usize, inner: usize, len: usize) -> Vec<T> {
        let mut o = buf::take::<T>(len);
        let per = (PAR / inner.max(1)).max(1);
        let body = |c: usize, ch: &mut [T]| {
            for (k, row) in ch.chunks_mut(inner.max(1)).enumerate() {
                let q = c * per + k;
                let s = ((q / ni) * d + iv[q % ni]) * inner;
                row.copy_from_slice(&v[s..s + inner]);
            }
        };
        if inner == 0 || len == 0 {
            return o;
        }
        if len > PAR {
            pool::chunks_mut(&mut o, per * inner, body);
        } else {
            body(0, &mut o);
        }
        o
    }
    let len: usize = shape.iter().product();
    Ok(vec![match &t.data {
        Data::F(v) => Tensor::f(shape, go(v, &iv, ni, d, inner, len)),
        Data::I(v) => Tensor::i(shape, go(v, &iv, ni, d, inner, len)),
    }])
}

fn gather_elements(t: &Tensor, idx: &Tensor, a: usize) -> R {
    let iv = idx.ints();
    let out = idx.shape.clone();
    let ist = strides(&t.shape);
    let d = t.shape[a] as i64;
    let f = move |mut i: usize| {
        let mut o = 0;
        let k = i;
        for dd in (0..out.len()).rev() {
            let c = i % out[dd];
            i /= out[dd];
            let c = if dd == a {
                let v = iv[k];
                (if v < 0 { v + d } else { v }) as usize
            } else {
                c
            };
            o += c * ist[dd];
        }
        o
    };
    Ok(vec![map_data(t, idx.shape.clone(), f)])
}

fn broadcast_to(t: &Tensor, shape: &[usize]) -> Tensor {
    let ist = bstrides(&t.shape, shape);
    strided_t(t, shape.to_vec(), &ist, 0)
}

fn reduce(op: &str, t: &Tensor, ax: &[i64], keep: bool) -> R {
    let r = t.shape.len();
    let mut ax: Vec<usize> = ax.iter().map(|&a| axis(a, r)).collect();
    ax.sort();
    ax.dedup();
    // 이어진 축 묶음만(이 모델들은 끝 축 하나)
    if ax.windows(2).any(|w| w[1] != w[0] + 1) {
        return Err("떨어진 축 줄이기 없음".into());
    }
    let (a0, a1) = (ax[0], *ax.last().unwrap() + 1);
    let outer: usize = t.shape[..a0].iter().product();
    let mid: usize = t.shape[a0..a1].iter().product();
    let inner: usize = t.shape[a1..].iter().product();
    let v = t.fv()?;
    let mut o = buf::take::<f32>(outer * inner);
    let p = Ptr(o.as_mut_ptr());
    let body = |i: usize| {
        for k in 0..inner {
            let it = (0..mid).map(|m| v[(i * mid + m) * inner + k]);
            let x = match op {
                "ReduceMax" => it.fold(f32::NEG_INFINITY, f32::max),
                "ReduceSum" => it.sum(),
                _ => it.sum::<f32>() / mid as f32,
            };
            unsafe { *p.at(i * inner + k) = x };
        }
    };
    if outer * mid * inner > PAR {
        pool::for_each(outer, &body);
    } else {
        (0..outer).for_each(body);
    }
    let mut shape = vec![];
    for (d, &s) in t.shape.iter().enumerate() {
        if d < a0 || d >= a1 {
            shape.push(s);
        } else if keep {
            shape.push(1);
        }
    }
    Ok(vec![Tensor::f(shape, o)])
}

fn topk(t: &Tensor, k: usize, a: usize, largest: bool) -> R {
    let outer: usize = t.shape[..a].iter().product();
    let d = t.shape[a];
    let inner: usize = t.shape[a + 1..].iter().product();
    let v = t.fv()?;
    let mut vals = buf::take::<f32>(outer * k * inner);
    let mut idx = vec![0i64; outer * k * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut ord: Vec<usize> = (0..d).collect();
            let at = |j: usize| v[(o * d + j) * inner + i];
            // 값 · 번호 차례(전순서): 앞 k 개만 골라 정렬해도 전체 정렬의 앞 k 개와 같음
            let cmp = |x: &usize, y: &usize| {
                let c = if largest { at(*y).total_cmp(&at(*x)) } else { at(*x).total_cmp(&at(*y)) };
                c.then(x.cmp(y))
            };
            if k < d {
                ord.select_nth_unstable_by(k, cmp);
            }
            ord[..k].sort_by(cmp);
            for (j, &s) in ord[..k].iter().enumerate() {
                vals[(o * k + j) * inner + i] = at(s);
                idx[(o * k + j) * inner + i] = s as i64;
            }
        }
    }
    let mut shape = t.shape.clone();
    shape[a] = k;
    Ok(vec![Tensor::f(shape.clone(), vals), Tensor::i(shape, idx)])
}

fn matmul(n: &Node, a: &Tensor, b: &Tensor) -> R {
    let av = a.fv()?;
    let ar = a.shape.len();
    let (m, k) = (a.shape[ar - 2], a.shape[ar - 1]);
    if let Pre::MatB { panels, k: kb, n: nn, bias } = &n.pre {
        if *kb != k {
            return Err(format!("MatMul k {k} ≠ {kb}"));
        }
        let rows = a.len() / k;
        let mut out = buf::take::<f32>(rows * nn);
        let ap = gemm::pack_a(av, rows, k, k);
        let sink = RowMajor { out: out.as_mut_ptr(), ldc: *nn, col_bias: bias.as_deref(), row_bias: None, act: n.act.clone() };
        gemm::gemm(&ap, rows, k, *nn, Panels::Packed(panels), &sink);
        buf::give(ap);
        let mut shape = a.shape.clone();
        shape[ar - 1] = *nn;
        return Ok(vec![Tensor::f(shape, out)]);
    }
    let bv = b.fv()?;
    let br = b.shape.len();
    let nn = b.shape[br - 1];
    if b.shape[br - 2] != k {
        return Err(format!("MatMul 모양 {:?} {:?}", a.shape, b.shape));
    }
    let batch = bshape(&a.shape[..ar - 2], &b.shape[..br - 2])?;
    let nb: usize = batch.iter().product();
    let sa = bstrides(&a.shape[..ar - 2], &batch);
    let sb = bstrides(&b.shape[..br - 2], &batch);
    let mut out = buf::take::<f32>(nb * m * nn);
    let op = Ptr(out.as_mut_ptr());
    pool::for_each(nb, &|bi| {
        let (mut ia, mut ib, mut q) = (0, 0, bi);
        for d in (0..batch.len()).rev() {
            let c = q % batch[d];
            q /= batch[d];
            ia += c * sa[d];
            ib += c * sb[d];
        }
        let am = &av[ia * m * k..(ia + 1) * m * k];
        let bm = &bv[ib * k * nn..(ib + 1) * k * nn];
        let ap = gemm::pack_a(am, m, k, k);
        let rows: Vec<usize> = (0..k).map(|j| bm[j * nn..].as_ptr() as usize).collect();
        let sink = RowMajor { out: op.at(bi * m * nn), ldc: nn, col_bias: None, row_bias: None, act: Act::None };
        gemm::gemm(&ap, m, k, nn, Panels::Gather { rows: &rows, avail: nn }, &sink);
        buf::give(ap);
    });
    let mut shape = batch;
    shape.push(m);
    shape.push(nn);
    Ok(vec![Tensor::f(shape, out)])
}

fn gemm_op(n: &Node, ins: &[Option<&Tensor>]) -> R {
    let a = need(ins, 0)?;
    if let Pre::MatB { .. } = n.pre {
        return matmul(n, a, a);
    }
    // 상수 아닌 Gemm(접기 때): 단순 계산
    let b = need(ins, 1)?;
    let (ta, tb) = (n.ai("transA", 0) != 0, n.ai("transB", 0) != 0);
    let (alpha, beta) = (n.af("alpha", 1.0), n.af("beta", 1.0));
    let (m, k) = if ta { (a.shape[1], a.shape[0]) } else { (a.shape[0], a.shape[1]) };
    let nn = if tb { b.shape[0] } else { b.shape[1] };
    let (av, bv) = (a.fv()?, b.fv()?);
    let c = ins.get(2).copied().flatten();
    let mut o = buf::take::<f32>(m * nn);
    for i in 0..m {
        for j in 0..nn {
            let mut s = 0.0;
            for l in 0..k {
                let x = if ta { av[l * m + i] } else { av[i * k + l] };
                let y = if tb { bv[j * k + l] } else { bv[l * nn + j] };
                s += x * y;
            }
            let cv = c.map(|c| {
                let v = c.fv().unwrap();
                if v.len() == 1 {
                    v[0]
                } else if v.len() == nn {
                    v[j]
                } else {
                    v[i * nn + j]
                }
            });
            o[i * nn + j] = alpha * s + beta * cv.unwrap_or(0.0);
        }
    }
    Ok(vec![Tensor::f(vec![m, nn], o)])
}

pub(crate) struct ConvGeo {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub dh: usize,
    pub dw: usize,
    pub pt: usize,
    pub pl: usize,
    pub oh: usize,
    pub ow: usize,
}

pub(crate) fn geo(n: &Node, xs: &[usize], kh: usize, kw: usize) -> ConvGeo {
    let (c, h, w) = (xs[1], xs[2], xs[3]);
    let s = n.ais("strides").unwrap_or(vec![1, 1]);
    let d = n.ais("dilations").unwrap_or(vec![1, 1]);
    let mut p = n.ais("pads").unwrap_or(vec![0, 0, 0, 0]);
    let ap = n.astr("auto_pad", "NOTSET");
    let (sh, sw, dh, dw) = (s[0] as usize, s[1] as usize, d[0] as usize, d[1] as usize);
    if ap == "SAME_UPPER" || ap == "SAME_LOWER" {
        let oh = h.div_ceil(sh);
        let ow = w.div_ceil(sw);
        let ph = ((oh - 1) * sh + dh * (kh - 1) + 1).saturating_sub(h);
        let pw = ((ow - 1) * sw + dw * (kw - 1) + 1).saturating_sub(w);
        p = if ap == "SAME_UPPER" {
            vec![(ph / 2) as i64, (pw / 2) as i64, (ph - ph / 2) as i64, (pw - pw / 2) as i64]
        } else {
            vec![(ph - ph / 2) as i64, (pw - pw / 2) as i64, (ph / 2) as i64, (pw / 2) as i64]
        };
    } else if ap == "VALID" {
        p = vec![0, 0, 0, 0];
    }
    let (pt, pl, pb, pr) = (p[0] as usize, p[1] as usize, p[2] as usize, p[3] as usize);
    let oh = (h + pt + pb - dh * (kh - 1) - 1) / sh + 1;
    let ow = (w + pl + pr - dw * (kw - 1) - 1) / sw + 1;
    ConvGeo { c, h, w, kh, kw, sh, sw, dh, dw, pt, pl, oh, ow }
}

fn pool(n: &Node, x: &Tensor) -> R {
    let k = n.ais("kernel_shape").ok_or("kernel_shape 없음")?;
    let (kh, kw) = (k[0] as usize, k[1] as usize);
    let g = geo(n, &x.shape, kh, kw);
    if n.ai("ceil_mode", 0) != 0 {
        return Err("ceil_mode 없음".into());
    }
    let max = n.op == "MaxPool";
    let incl = n.ai("count_include_pad", 0) != 0;
    if x.blk > 0 {
        return super::nchwc::pool_blk(x, &g, max, incl);
    }
    let nb = x.shape[0];
    let xv = x.fv()?;
    let mut out = buf::take::<f32>(nb * g.c * g.oh * g.ow);
    let sep = max && g.sh == 1 && g.sw == 1 && g.dh == 1 && g.dw == 1;
    if !sep && nb * g.c < 2 * pool::threads() {
        // 평면이 적으면(XFeat 1 채널 평균) 출력 줄 묶음으로 나눔
        let nblk = (2 * pool::threads()).div_ceil(nb * g.c).min(g.oh).max(1);
        let blk = g.oh.div_ceil(nblk);
        let op = Ptr(out.as_mut_ptr());
        pool::for_each(nb * g.c * nblk, &|t| {
            let (bc, bi) = (t / nblk, t % nblk);
            let plane = &xv[bc * g.h * g.w..(bc + 1) * g.h * g.w];
            for oy in bi * blk..((bi + 1) * blk).min(g.oh) {
                // SAFETY: 출력 줄은 일마다 겹치지 않음
                let o = unsafe { std::slice::from_raw_parts_mut(op.at((bc * g.oh + oy) * g.ow), g.ow) };
                pool_row(&g, plane, oy, kh, kw, max, incl, o);
            }
        });
        return Ok(vec![Tensor::f(vec![nb, g.c, g.oh, g.ow], out)]);
    }
    pool::chunks_mut(&mut out, g.oh * g.ow, |bc, o| {
        let plane = &xv[bc * g.h * g.w..(bc + 1) * g.h * g.w];
        if sep {
            // 보폭 1 최대: 가로 창 최대(입력 줄마다) → 세로 창 최대. 최대는 차례와 상관없이 같은 값
            return simd::run(|| {
                let mut t = vec![f32::NEG_INFINITY; g.h * g.ow];
                for y in 0..g.h {
                    let (row, tr) = (&plane[y * g.w..(y + 1) * g.w], &mut t[y * g.ow..(y + 1) * g.ow]);
                    for kx in 0..kw {
                        let off = kx as isize - g.pl as isize;
                        let lo = (-off).clamp(0, g.ow as isize) as usize;
                        let hi = (g.w as isize - off).clamp(lo as isize, g.ow as isize) as usize;
                        for ox in lo..hi {
                            tr[ox] = tr[ox].max(row[(ox as isize + off) as usize]);
                        }
                    }
                }
                for oy in 0..g.oh {
                    let orow = &mut o[oy * g.ow..(oy + 1) * g.ow];
                    orow.fill(f32::NEG_INFINITY);
                    for ky in 0..kh {
                        let iy = (oy + ky) as isize - g.pt as isize;
                        if iy < 0 || iy >= g.h as isize {
                            continue;
                        }
                        let tr = &t[iy as usize * g.ow..(iy as usize + 1) * g.ow];
                        for (a, &b) in orow.iter_mut().zip(tr) {
                            *a = a.max(b);
                        }
                    }
                }
            });
        }
        for oy in 0..g.oh {
            pool_row(&g, plane, oy, kh, kw, max, incl, &mut o[oy * g.ow..(oy + 1) * g.ow]);
        }
    });
    Ok(vec![Tensor::f(vec![nb, g.c, g.oh, g.ow], out)])
}

/// 풀링 출력 한 줄(oy)
#[allow(clippy::too_many_arguments)]
fn pool_row(g: &ConvGeo, plane: &[f32], oy: usize, kh: usize, kw: usize, max: bool, incl: bool, o: &mut [f32]) {
    for (ox, ov) in o.iter_mut().enumerate() {
        let (mut acc, mut cnt) = (if max { f32::NEG_INFINITY } else { 0.0 }, 0usize);
        for ky in 0..kh {
            let iy = (oy * g.sh + ky * g.dh) as isize - g.pt as isize;
            for kx in 0..kw {
                let ix = (ox * g.sw + kx * g.dw) as isize - g.pl as isize;
                if iy >= 0 && iy < g.h as isize && ix >= 0 && ix < g.w as isize {
                    let v = plane[iy as usize * g.w + ix as usize];
                    if max {
                        acc = acc.max(v);
                    } else {
                        acc += v;
                    }
                    cnt += 1;
                }
            }
        }
        *ov = if max { acc } else { acc / (if incl { kh * kw } else { cnt.max(1) }) as f32 };
    }
}

fn resize(n: &Node, ins: &[Option<&Tensor>]) -> R {
    let x = need(ins, 0)?;
    let r = x.shape.len();
    let scales = ins.get(2).copied().flatten().map(|s| s.clone().into_f()).filter(|s| !s.is_empty());
    let sizes = ins.get(3).copied().flatten().map(|s| s.ints()).filter(|s| !s.is_empty());
    let (out, sc): (Vec<usize>, Vec<f32>) = if let Some(sz) = sizes {
        let o: Vec<usize> = sz.iter().map(|&v| v as usize).collect();
        let s = (0..r).map(|d| o[d] as f32 / x.shape[d] as f32).collect();
        (o, s)
    } else if let Some(s) = scales {
        ((0..r).map(|d| (x.shape[d] as f32 * s[d]).floor() as usize).collect(), s)
    } else {
        return Err("Resize 크기 없음".into());
    };
    if r != 4 || out[0] != x.shape[0] || out[1] != x.shape[1] {
        return Err("Resize 는 NCHW 의 H · W 만".into());
    }
    let mode = n.astr("mode", "nearest");
    let ctm = n.astr("coordinate_transformation_mode", "half_pixel");
    let nm = n.astr("nearest_mode", "round_prefer_floor");
    let (h, w, oh, ow) = (x.shape[2], x.shape[3], out[2], out[3]);
    let src = |o: usize, s: f32, len_in: usize, len_out: usize| -> f32 {
        let o = o as f32;
        match ctm.as_str() {
            "asymmetric" => o / s,
            "align_corners" => {
                if len_out == 1 {
                    0.0
                } else {
                    o * (len_in as f32 - 1.0) / (len_out as f32 - 1.0)
                }
            }
            "pytorch_half_pixel" => {
                if len_out > 1 {
                    (o + 0.5) / s - 0.5
                } else {
                    0.0
                }
            }
            _ => (o + 0.5) / s - 0.5,
        }
    };
    let xv = x.fv()?;
    let mut o = buf::take::<f32>(x.shape[0] * x.shape[1] * oh * ow);
    if mode == "nearest" {
        let pick = |v: f32, len: usize| -> usize {
            let i = match nm.as_str() {
                "floor" => v.floor(),
                "ceil" => v.ceil(),
                "round_prefer_ceil" => (v + 0.5).floor(),
                _ => {
                    if v.fract() == 0.5 {
                        v.floor()
                    } else {
                        v.round()
                    }
                }
            };
            (i.max(0.0) as usize).min(len - 1)
        };
        let ys: Vec<usize> = (0..oh).map(|y| pick(src(y, sc[2], h, oh), h)).collect();
        let xs: Vec<usize> = (0..ow).map(|x| pick(src(x, sc[3], w, ow), w)).collect();
        if x.blk > 0 {
            let yt: Vec<(usize, usize, f32)> = ys.iter().map(|&y| (y, y, 0.0)).collect();
            let xt: Vec<(usize, usize, f32)> = xs.iter().map(|&x| (x, x, 0.0)).collect();
            return super::nchwc::resize(x, oh, ow, &yt, &xt, false);
        }
        pool::chunks_mut(&mut o, oh * ow, |c, op| {
            let plane = &xv[c * h * w..(c + 1) * h * w];
            for y in 0..oh {
                for xx in 0..ow {
                    op[y * ow + xx] = plane[ys[y] * w + xs[xx]];
                }
            }
        });
    } else if mode == "linear" {
        let lin = |o: usize, s: f32, len: usize, lo: usize| -> (usize, usize, f32) {
            let v = src(o, s, len, lo).clamp(0.0, (len - 1) as f32);
            let i0 = v.floor() as usize;
            let i1 = (i0 + 1).min(len - 1);
            (i0, i1, v - i0 as f32)
        };
        let ys: Vec<_> = (0..oh).map(|y| lin(y, sc[2], h, oh)).collect();
        let xs: Vec<_> = (0..ow).map(|x| lin(x, sc[3], w, ow)).collect();
        if x.blk > 0 {
            return super::nchwc::resize(x, oh, ow, &ys, &xs, true);
        }
        pool::chunks_mut(&mut o, oh * ow, |c, op| {
            let plane = &xv[c * h * w..(c + 1) * h * w];
            for (y, &(y0, y1, fy)) in ys.iter().enumerate() {
                for (xx, &(x0, x1, fx)) in xs.iter().enumerate() {
                    let a = plane[y0 * w + x0] * (1.0 - fx) + plane[y0 * w + x1] * fx;
                    let b = plane[y1 * w + x0] * (1.0 - fx) + plane[y1 * w + x1] * fx;
                    op[y * ow + xx] = a * (1.0 - fy) + b * fy;
                }
            }
        });
    } else {
        return Err(format!("Resize {mode} 없음"));
    }
    Ok(vec![Tensor::f(out, o)])
}
