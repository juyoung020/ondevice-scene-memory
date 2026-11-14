//! 아주 작은 protobuf 읽기·쓰기(ONNX 모델 손질용): 메시지를 (필드 번호, 값) 목록으로 풀고 다시 짠다.
//! 모르는 필드는 바이트 그대로 지나가므로, 고친 곳 말고는 원래 파일과 같은 내용이 된다.

#[derive(Clone, Debug)]
pub enum Val {
    Varint(u64),
    I64([u8; 8]),
    Bytes(Vec<u8>),
    I32([u8; 4]),
}

#[derive(Clone, Debug, Default)]
pub struct Msg(pub Vec<(u32, Val)>);

fn varint(b: &[u8], i: &mut usize) -> Result<u64, String> {
    let (mut v, mut s) = (0u64, 0);
    loop {
        let x = *b.get(*i).ok_or("protobuf: varint 가 잘림")?;
        *i += 1;
        v |= ((x & 0x7f) as u64) << s;
        if x < 0x80 {
            return Ok(v);
        }
        s += 7;
        if s > 63 {
            return Err("protobuf: varint 가 너무 김".into());
        }
    }
}

fn put_varint(o: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        o.push((v as u8) | 0x80);
        v >>= 7;
    }
    o.push(v as u8);
}

impl Msg {
    pub fn parse(b: &[u8]) -> Result<Msg, String> {
        let (mut i, mut f) = (0, vec![]);
        while i < b.len() {
            let key = varint(b, &mut i)?;
            let (no, wt) = ((key >> 3) as u32, key & 7);
            let v = match wt {
                0 => Val::Varint(varint(b, &mut i)?),
                1 => {
                    let v = b.get(i..i + 8).ok_or("protobuf: 잘림")?.try_into().unwrap();
                    i += 8;
                    Val::I64(v)
                }
                2 => {
                    let n = varint(b, &mut i)? as usize;
                    let v = b.get(i..i + n).ok_or("protobuf: 길이 필드가 잘림")?.to_vec();
                    i += n;
                    Val::Bytes(v)
                }
                5 => {
                    let v = b.get(i..i + 4).ok_or("protobuf: 잘림")?.try_into().unwrap();
                    i += 4;
                    Val::I32(v)
                }
                _ => return Err(format!("protobuf: 모르는 형식 {wt}")),
            };
            f.push((no, v));
        }
        Ok(Msg(f))
    }

    pub fn bytes(&self) -> Vec<u8> {
        let mut o = vec![];
        for (no, v) in &self.0 {
            let wt = match v {
                Val::Varint(_) => 0,
                Val::I64(_) => 1,
                Val::Bytes(_) => 2,
                Val::I32(_) => 5,
            };
            put_varint(&mut o, ((*no as u64) << 3) | wt);
            match v {
                Val::Varint(x) => put_varint(&mut o, *x),
                Val::I64(x) => o.extend_from_slice(x),
                Val::Bytes(x) => {
                    put_varint(&mut o, x.len() as u64);
                    o.extend_from_slice(x);
                }
                Val::I32(x) => o.extend_from_slice(x),
            }
        }
        o
    }

    /// 필드 no 의 첫 문자열
    pub fn str(&self, no: u32) -> Option<String> {
        self.0.iter().find_map(|(n, v)| match v {
            Val::Bytes(b) if *n == no => Some(String::from_utf8_lossy(b).into_owned()),
            _ => None,
        })
    }
    /// 필드 no 의 문자열 전부(반복 필드)
    pub fn strs(&self, no: u32) -> Vec<String> {
        self.0.iter().filter_map(|(n, v)| match v {
            Val::Bytes(b) if *n == no => Some(String::from_utf8_lossy(b).into_owned()),
            _ => None,
        }).collect()
    }
    /// 필드 no 의 하위 메시지 전부
    pub fn msgs(&self, no: u32) -> Vec<Msg> {
        self.0.iter().filter_map(|(n, v)| match v {
            Val::Bytes(b) if *n == no => Msg::parse(b).ok(),
            _ => None,
        }).collect()
    }
    /// 필드 no 의 정수 전부(포장된 것과 하나씩 든 것 모두)
    pub fn ints(&self, no: u32) -> Vec<i64> {
        let mut r = vec![];
        for (n, v) in &self.0 {
            if *n != no {
                continue;
            }
            match v {
                Val::Varint(x) => r.push(*x as i64),
                Val::Bytes(b) => {
                    let mut i = 0;
                    while i < b.len() {
                        match varint(b, &mut i) {
                            Ok(x) => r.push(x as i64),
                            Err(_) => break,
                        }
                    }
                }
                _ => {}
            }
        }
        r
    }
    pub fn remove(&mut self, no: u32) {
        self.0.retain(|(n, _)| *n != no);
    }
    pub fn push_str(&mut self, no: u32, s: &str) {
        self.0.push((no, Val::Bytes(s.as_bytes().to_vec())));
    }
    pub fn push_msg(&mut self, no: u32, m: &Msg) {
        self.0.push((no, Val::Bytes(m.bytes())));
    }
    pub fn push_int(&mut self, no: u32, x: i64) {
        self.0.push((no, Val::Varint(x as u64)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let mut m = Msg::default();
        m.push_str(1, "abc");
        m.push_int(2, 300);
        let mut sub = Msg::default();
        sub.push_int(1, 7);
        m.push_msg(3, &sub);
        let b = m.bytes();
        let p = Msg::parse(&b).unwrap();
        assert_eq!(p.bytes(), b);
        assert_eq!(p.str(1).unwrap(), "abc");
        assert_eq!(p.ints(2), vec![300]);
        assert_eq!(p.msgs(3)[0].ints(1), vec![7]);
    }
}
