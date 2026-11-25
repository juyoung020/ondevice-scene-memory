//! ONNX 파일(protobuf) 읽기: 필요한 칸만(그래프 · 노드 · 속성 · 텐서 · 입출력 모양)
use super::{Attr, Tensor};

pub struct Rd<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Rd<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Rd { b, i: 0 }
    }
    fn varint(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        let mut s = 0;
        loop {
            let c = *self.b.get(self.i).ok_or("protobuf: 끝")?;
            self.i += 1;
            v |= ((c & 0x7f) as u64) << s;
            if c & 0x80 == 0 {
                return Ok(v);
            }
            s += 7;
            if s > 63 {
                return Err("protobuf: varint".into());
            }
        }
    }
    /// 다음 칸: (번호, 값). 값은 varint/고정 폭이면 Num, 길이 붙은 칸이면 Bytes
    pub fn next(&mut self) -> Result<Option<(u32, Val<'a>)>, String> {
        if self.i >= self.b.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let (f, wt) = ((key >> 3) as u32, key & 7);
        let v = match wt {
            0 => Val::Num(self.varint()?),
            1 => {
                let s = self.b.get(self.i..self.i + 8).ok_or("protobuf: 끝")?;
                self.i += 8;
                Val::Num(u64::from_le_bytes(s.try_into().unwrap()))
            }
            5 => {
                let s = self.b.get(self.i..self.i + 4).ok_or("protobuf: 끝")?;
                self.i += 4;
                Val::Num(u32::from_le_bytes(s.try_into().unwrap()) as u64)
            }
            2 => {
                let n = self.varint()? as usize;
                let s = self.b.get(self.i..self.i + n).ok_or("protobuf: 끝")?;
                self.i += n;
                Val::Bytes(s)
            }
            _ => return Err(format!("protobuf: 형식 {wt}")),
        };
        Ok(Some((f, v)))
    }
}

pub enum Val<'a> {
    Num(u64),
    Bytes(&'a [u8]),
}

impl<'a> Val<'a> {
    fn num(&self) -> u64 {
        match self {
            Val::Num(n) => *n,
            _ => 0,
        }
    }
    fn bytes(&self) -> &'a [u8] {
        match self {
            Val::Bytes(b) => b,
            _ => &[],
        }
    }
    fn str(&self) -> String {
        String::from_utf8_lossy(self.bytes()).into_owned()
    }
    /// 반복 정수: 묶음(packed) 또는 하나
    fn ints(&self, out: &mut Vec<i64>) -> Result<(), String> {
        match self {
            Val::Num(n) => out.push(*n as i64),
            Val::Bytes(b) => {
                let mut r = Rd::new(b);
                while r.i < b.len() {
                    out.push(r.varint()? as i64);
                }
            }
        }
        Ok(())
    }
    fn floats(&self, out: &mut Vec<f32>) {
        match self {
            Val::Num(n) => out.push(f32::from_bits(*n as u32)),
            Val::Bytes(b) => out.extend(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()))),
        }
    }
}

pub fn f16(h: u16) -> f32 {
    let s = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            // 정규화 안 된 수
            let mut e = 113u32;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// TensorProto → (이름, 텐서)
pub fn tensor(b: &[u8]) -> Result<(String, Tensor), String> {
    let mut r = Rd::new(b);
    let (mut dims, mut dt, mut name) = (vec![], 0u64, String::new());
    let (mut fl, mut i32s, mut i64s, mut raw, mut f64s): (Vec<f32>, Vec<i64>, Vec<i64>, &[u8], Vec<f64>) = (vec![], vec![], vec![], &[], vec![]);
    while let Some((f, v)) = r.next()? {
        match f {
            1 => v.ints(&mut dims)?,
            2 => dt = v.num(),
            4 => v.floats(&mut fl),
            5 => v.ints(&mut i32s)?,
            7 => v.ints(&mut i64s)?,
            8 => name = v.str(),
            9 => raw = v.bytes(),
            10 => match v {
                Val::Num(n) => f64s.push(f64::from_bits(n)),
                Val::Bytes(b) => f64s.extend(b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()))),
            },
            13 | 14 => return Err(format!("{name}: 바깥 파일 데이터는 못 읽음")),
            _ => {}
        }
    }
    let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
    let n: usize = shape.iter().product();
    let t = match dt {
        1 => Tensor::f(shape, if raw.is_empty() { fl } else { raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }),
        10 => Tensor::f(
            shape,
            if raw.is_empty() { i32s.iter().map(|&v| f16(v as u16)).collect() } else { raw.chunks_exact(2).map(|c| f16(u16::from_le_bytes([c[0], c[1]]))).collect() },
        ),
        11 => Tensor::f(shape, if raw.is_empty() { f64s.iter().map(|&v| v as f32).collect() } else { raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect() }),
        7 => Tensor::i(shape, if raw.is_empty() { i64s } else { raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect() }),
        6 | 5 | 3 => Tensor::i(
            shape,
            if raw.is_empty() {
                i32s.iter().map(|&v| v as i32 as i64).collect()
            } else {
                match dt {
                    6 => raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
                    5 => raw.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as i64).collect(),
                    _ => raw.iter().map(|&c| c as i8 as i64).collect(),
                }
            },
        ),
        2 | 9 | 4 => Tensor::i(shape, if raw.is_empty() { i32s } else { raw.iter().map(|&c| c as i64).collect() }),
        _ => return Err(format!("{name}: 자료형 {dt} 못 읽음")),
    };
    let have = t.len();
    if have != n {
        return Err(format!("{name}: 값 {have} 개, 모양은 {n}"));
    }
    Ok((name, t))
}

fn attr(b: &[u8]) -> Result<(String, Attr), String> {
    let mut r = Rd::new(b);
    let mut name = String::new();
    let (mut f, mut i, mut s, mut t, mut fs, mut is, mut ty) = (None, None, None, None, vec![], vec![], 0u64);
    while let Some((k, v)) = r.next()? {
        match k {
            1 => name = v.str(),
            2 => f = Some(f32::from_bits(v.num() as u32)),
            3 => i = Some(v.num() as i64),
            4 => s = Some(v.str()),
            5 => t = Some(tensor(v.bytes())?.1),
            7 => v.floats(&mut fs),
            8 => v.ints(&mut is)?,
            20 => ty = v.num(),
            _ => {}
        }
    }
    let a = match ty {
        1 => Attr::F(f.unwrap_or(0.0)),
        2 => Attr::I(i.unwrap_or(0)),
        3 => Attr::S(s.unwrap_or_default()),
        4 => Attr::T(t.ok_or("속성 텐서 없음")?),
        6 => Attr::Fs(fs),
        7 => Attr::Is(is),
        _ => {
            if let Some(t) = t {
                Attr::T(t)
            } else if let Some(i) = i {
                Attr::I(i)
            } else if let Some(f) = f {
                Attr::F(f)
            } else if let Some(s) = s {
                Attr::S(s)
            } else if !is.is_empty() {
                Attr::Is(is)
            } else {
                Attr::Fs(fs)
            }
        }
    };
    Ok((name, a))
}

pub struct RawNode {
    pub op: String,
    pub ins: Vec<String>,
    pub outs: Vec<String>,
    pub attrs: Vec<(String, Attr)>,
}

pub struct RawGraph {
    pub nodes: Vec<RawNode>,
    pub inits: Vec<(String, Tensor)>,
    /// 이름, 모양(-1 = 정해지지 않음), 원소형
    pub inputs: Vec<(String, Vec<i64>, u64)>,
    pub outputs: Vec<String>,
}

fn value_info(b: &[u8]) -> Result<(String, Vec<i64>, u64), String> {
    let mut r = Rd::new(b);
    let (mut name, mut shape, mut et) = (String::new(), vec![], 0);
    while let Some((f, v)) = r.next()? {
        match f {
            1 => name = v.str(),
            2 => {
                let mut tr = Rd::new(v.bytes());
                while let Some((f, v)) = tr.next()? {
                    if f != 1 {
                        continue;
                    }
                    let mut tt = Rd::new(v.bytes());
                    while let Some((f, v)) = tt.next()? {
                        match f {
                            1 => et = v.num(),
                            2 => {
                                let mut sr = Rd::new(v.bytes());
                                while let Some((f, v)) = sr.next()? {
                                    if f != 1 {
                                        continue;
                                    }
                                    let mut dr = Rd::new(v.bytes());
                                    let mut d = -1i64;
                                    while let Some((f, v)) = dr.next()? {
                                        if f == 1 {
                                            d = v.num() as i64;
                                        }
                                    }
                                    shape.push(d);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok((name, shape, et))
}

pub fn model(b: &[u8]) -> Result<RawGraph, String> {
    let mut r = Rd::new(b);
    let mut graph = None;
    while let Some((f, v)) = r.next()? {
        if f == 7 {
            graph = Some(v.bytes());
        }
    }
    let mut r = Rd::new(graph.ok_or("그래프 없음")?);
    let mut g = RawGraph { nodes: vec![], inits: vec![], inputs: vec![], outputs: vec![] };
    while let Some((f, v)) = r.next()? {
        match f {
            1 => {
                let mut nr = Rd::new(v.bytes());
                let mut n = RawNode { op: String::new(), ins: vec![], outs: vec![], attrs: vec![] };
                while let Some((f, v)) = nr.next()? {
                    match f {
                        1 => n.ins.push(v.str()),
                        2 => n.outs.push(v.str()),
                        4 => n.op = v.str(),
                        5 => n.attrs.push(attr(v.bytes())?),
                        _ => {}
                    }
                }
                g.nodes.push(n);
            }
            5 => g.inits.push(tensor(v.bytes())?),
            11 => g.inputs.push(value_info(v.bytes())?),
            12 => g.outputs.push(value_info(v.bytes())?.0),
            _ => {}
        }
    }
    Ok(g)
}
