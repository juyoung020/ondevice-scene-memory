//! ONNX 모델의 입력·출력 이름과 모양을 찍는다: onnx_info models/yolo26s-seg.onnx
use ort::session::Session;

fn main() -> ort::Result<()> {
    let path = std::env::args().nth(1).expect("usage: onnx_info <model.onnx>");
    let env = ort::init().build()?;
    let s = Session::builder(&env)?.commit_from_file(&path)?;
    for i in s.inputs() {
        println!("in  {} {:?}", i.name(), i.dtype());
    }
    for o in s.outputs() {
        println!("out {} {:?}", o.name(), o.dtype());
    }
    Ok(())
}
