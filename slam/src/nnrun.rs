//! 루프 · 재위치용 신경망(MegaLoc · XFeat) 실행기 하나: 보드는 DXRT(.dxnn, 비동기 올리기 · 받기),
//! PC 는 ONNX Runtime(.onnx). 출력은 모든 텐서를 이어 붙인 f32 버퍼 하나 + 이름별 (시작, 길이).
//!
//! 입력: MegaLoc 은 f32 조각(patches), XFeat 은 ONNX 판 f32 정규화 회색 · NPU 판 u8 회색(모델이 u8 을 받는지 [`Net::wants_u8`]).
//! DXRT 출력 순서는 모델 파일이 정하므로 이름으로 찾는다([`Net::span`]).

/// 올린 추론 하나
pub enum Ticket {
    /// ONNX: 올릴 때 이미 끝나 결과를 들고 있음
    Done(Vec<f32>),
    #[cfg(feature = "dxrt")]
    Job(i32),
}

enum Kind {
    #[cfg(feature = "onnx")]
    Onnx { sess: ort::session::Session, input: String, shape: Vec<i64>, u8_in: bool },
    #[cfg(feature = "dxrt")]
    Dx(common::dxrt::Engine),
    /// 실행기 없는 빌드(시험 · 후처리만)
    #[allow(dead_code)]
    None,
}

pub struct Net {
    kind: Kind,
    /// 이름, f32 시작, 길이
    outs: Vec<(String, usize, usize)>,
    total: usize,
}

impl Net {
    /// `.dxnn` 은 DXRT(NPU 코어 묶기 bound: 0 = 셋 다), 그 밖은 ONNX Runtime(스레드 threads)
    pub fn open(path: &str, threads: usize, bound: u32) -> Result<Net, String> {
        let _ = (threads, bound);
        #[cfg(feature = "dxrt")]
        if path.ends_with(".dxnn") {
            let e = common::dxrt::Engine::try_open(path, bound, common::dxrt::BUFFERS.load(std::sync::atomic::Ordering::Relaxed))?;
            let mut outs = Vec::new();
            for i in 0..e.tensors.len() {
                let (s, n) = e.f32_span(i);
                outs.push((e.tensors[i].0.clone(), s, n));
            }
            let total = e.out.len().div_ceil(4);
            return Ok(Net { kind: Kind::Dx(e), outs, total });
        }
        #[cfg(feature = "onnx")]
        {
            use ort::value::ValueType;
            let sess = ort::session::Session::builder(common::ort::env())
                .map_err(|e| format!("{path}: {e}"))?
                .with_intra_threads(threads.max(1))
                .map_err(|e| format!("{path}: {e}"))?
                .commit_from_file(path)
                .map_err(|e| format!("{path}: {e}"))?;
            let inp = &sess.inputs()[0];
            let (shape, u8_in) = match inp.dtype() {
                ValueType::Tensor { ty, shape, .. } => (shape.iter().copied().collect::<Vec<i64>>(), matches!(ty, ort::value::TensorElementType::Uint8)),
                _ => return Err(format!("{path}: 입력이 텐서가 아님")),
            };
            let mut outs = Vec::new();
            let mut off = 0usize;
            for o in sess.outputs() {
                let n = match o.dtype() {
                    ValueType::Tensor { shape, .. } => shape.iter().map(|&d| d.max(1) as usize).product::<usize>(),
                    _ => 0,
                };
                outs.push((o.name().to_string(), off, n));
                off += n;
            }
            let input = inp.name().to_string();
            return Ok(Net { kind: Kind::Onnx { sess, input, shape, u8_in }, outs, total: off });
        }
        #[allow(unreachable_code)]
        Err(format!("{path}: 이 빌드에는 실행기가 없다(--features onnx 또는 dxrt)"))
    }

    /// 입력을 u8 로 받는 모델인가(XFeat NPU 판 · ONNX u8 판)
    pub fn wants_u8(&self) -> bool {
        match &self.kind {
            #[cfg(feature = "onnx")]
            Kind::Onnx { u8_in, .. } => *u8_in,
            #[cfg(feature = "dxrt")]
            Kind::Dx(e) => e.input == "image" && self.outs.iter().any(|o| o.0 == "kpts"),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    pub fn has(&self, name: &str) -> bool {
        self.outs.iter().any(|o| o.0 == name)
    }
    /// 출력 이름 → (f32 시작, 길이)
    pub fn span(&self, name: &str) -> (usize, usize) {
        let o = self.outs.iter().find(|o| o.0 == name).unwrap_or_else(|| panic!("모델 출력 {name} 없음: {:?}", self.outs));
        (o.1, o.2)
    }
    pub fn out<'a>(&self, buf: &'a [f32], name: &str) -> &'a [f32] {
        let (s, n) = self.span(name);
        &buf[s..s + n]
    }
    pub fn out_len(&self) -> usize {
        self.total
    }

    /// 입력(바이트: f32 면 f32 의 바이트, u8 이면 그대로) 하나를 올림. DXRT 는 바로 돌아오고 입력 버퍼는 받을 때까지 그대로 둬야 한다
    pub fn submit(&mut self, input: &[u8]) -> Ticket {
        match &mut self.kind {
            #[cfg(feature = "dxrt")]
            Kind::Dx(e) => Ticket::Job(e.submit(input)),
            #[cfg(feature = "onnx")]
            Kind::Onnx { sess, input: name, shape, u8_in } => {
                let shape: Vec<usize> = shape.iter().map(|&d| d.max(1) as usize).collect();
                let outs = if *u8_in {
                    let v = ort::value::TensorRef::from_array_view((shape, input)).expect("입력");
                    sess.run(ort::inputs![name.as_str() => v]).expect("onnx run")
                } else {
                    // SAFETY: f32 입력은 부르는 쪽이 f32 슬라이스의 바이트로 넘긴다(정렬 4)
                    let f = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f32, input.len() / 4) };
                    let v = ort::value::TensorRef::from_array_view((shape, f)).expect("입력");
                    sess.run(ort::inputs![name.as_str() => v]).expect("onnx run")
                };
                let mut buf = vec![0f32; self.total];
                for (n, s, len) in &self.outs {
                    let (_, d) = outs[n.as_str()].try_extract_tensor::<f32>().expect("출력");
                    buf[*s..*s + len].copy_from_slice(&d[..*len]);
                }
                Ticket::Done(buf)
            }
            #[allow(unreachable_patterns)]
            _ => panic!("실행기 없음"),
        }
    }

    /// 결과 받기(받을 버퍼를 미리 넘김: 크기가 맞으면 다시 잡지 않는다)
    pub fn wait(&mut self, t: Ticket, out: &mut Vec<f32>) {
        match (&mut self.kind, t) {
            #[cfg(feature = "dxrt")]
            (Kind::Dx(e), Ticket::Job(id)) => e.waiter().wait_into(id, out),
            (_, Ticket::Done(v)) => *out = v,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// 올리고 바로 받기
    pub fn run(&mut self, input: &[u8], out: &mut Vec<f32>) {
        let t = self.submit(input);
        self.wait(t, out);
    }
}

/// f32 조각을 바이트로(DXRT · ONNX 입력)
pub fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 는 4 바이트, 정렬은 u8 이 더 느슨함
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
