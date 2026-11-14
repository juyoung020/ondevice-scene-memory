//! sgview 서버 실행 파일. 본체는 lib.rs(`sgview::run`), memmap --serve 가 같은 것을 같은 프로세스에서 부른다.
fn main() {
    sgview::run(std::env::args().skip(1).collect());
}
