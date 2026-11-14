//! 프로세스에 ONNX Runtime 환경 하나: 분할·깊이·특징점·사진 벡터(pipeline)와 글 검색(sgview)이 같은 환경에서 세션을 만든다.
//! 세션마다 스레드 수(`with_intra_threads`)는 부르는 쪽이 정한다.

pub fn env() -> &'static ort::environment::Environment {
    static ENV: std::sync::OnceLock<&'static ort::environment::Environment> = std::sync::OnceLock::new();
    ENV.get_or_init(|| Box::leak(Box::new(ort::init().build().expect("onnxruntime"))))
}
