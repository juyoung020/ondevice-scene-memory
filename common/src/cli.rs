//! 명령줄 읽기 하나: `--키 값`(처음 나온 것), `--표시`, 묶음 인자(`--slam "<인자들>"`)를 낱말로 나누기.
//! `ondevice` 는 명령줄을 처음에 한 번 읽어 위치(slam) · 기억(memmap) · 뷰어(serve) 설정으로 나눈다.

#[derive(Clone, Debug, Default)]
pub struct Argv(pub Vec<String>);

impl Argv {
    pub fn new(v: Vec<String>) -> Argv {
        Argv(v)
    }
    /// 이 프로세스의 명령줄
    pub fn env() -> Argv {
        Argv(std::env::args().collect())
    }
    /// `--키 값`: 처음 나온 키 다음 낱말
    pub fn get(&self, k: &str) -> Option<String> {
        self.0.iter().position(|a| a == k).and_then(|i| self.0.get(i + 1)).cloned()
    }
    /// 값을 읽어 바꿈(없으면 기본값, 못 읽으면 멈춤)
    pub fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> T {
        match self.get(k) {
            Some(s) => s.parse().unwrap_or_else(|_| panic!("{k} {s}: 숫자가 아님")),
            None => default,
        }
    }
    /// `--표시` 가 있는지
    pub fn flag(&self, k: &str) -> bool {
        self.0.iter().any(|a| a == k)
    }
    /// `--키 값` 의 값을 낱말로 나눔(묶음 인자: `--slam "--tum <폴더> --rate 0"`)
    pub fn group(&self, k: &str) -> Option<Vec<String>> {
        self.get(k).map(|s| split(&s))
    }
}

/// 빈칸으로 낱말 나누기
pub fn split(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn get_flag_group() {
        let a = Argv::new(["p", "--out", "o", "--slam", "--tum d  --rate 0", "--no-drop", "--every", "2"].map(String::from).to_vec());
        assert_eq!(a.get("--out").as_deref(), Some("o"));
        assert!(a.flag("--no-drop") && !a.flag("--loop"));
        assert_eq!(a.num("--every", 5usize), 2);
        assert_eq!(a.num("--conf", 0.35f32), 0.35);
        assert_eq!(a.group("--slam").unwrap(), ["--tum", "d", "--rate", "0"]);
    }
}
