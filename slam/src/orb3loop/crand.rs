//! glibc rand()/srand() (TYPE_3: x[i] = x[i-3] + x[i-31], 출력 >> 1) 와 DUtils::Random::RandomInt.
//! ORB-SLAM3 RGB-D 는 srand 를 부르지 않아 시작 씨앗 1(glibc 기본). 모든 호출자(Sim3Solver · MLPnPsolver 등)가
//! 한 상태를 같이 쓰므로 통합 쪽이 이 생성기 하나를 나눠 써야 원본과 같은 수열이 된다.

#[derive(Clone)]
pub struct GlibcRand {
    r: [i32; 34],
    f: usize,
    b: usize,
}

impl GlibcRand {
    pub fn new(seed: u32) -> Self {
        // __srandom_r
        let mut r = [0i32; 34];
        let seed = if seed == 0 { 1 } else { seed };
        r[0] = seed as i32;
        let mut word = seed as i32 as i64;
        for i in 1..31 {
            let hi = word / 127773;
            let lo = word % 127773;
            word = 16807 * lo - 2836 * hi;
            if word < 0 {
                word += 2147483647;
            }
            r[i] = word as i32;
        }
        let mut g = GlibcRand { r, f: 3, b: 0 };
        for _ in 0..310 {
            g.next_raw();
        }
        g
    }

    #[inline]
    fn next_raw(&mut self) -> i32 {
        // random_r: *fptr += *rptr; result = (*fptr >> 1) & 0x7fffffff
        let v = (self.r[self.f] as u32).wrapping_add(self.r[self.b] as u32);
        self.r[self.f] = v as i32;
        self.f += 1;
        if self.f >= 31 {
            self.f = 0;
            self.b += 1;
        } else {
            self.b += 1;
            if self.b >= 31 {
                self.b = 0;
            }
        }
        (v >> 1) as i32
    }

    /// rand()
    pub fn rand(&mut self) -> i32 {
        self.next_raw()
    }

    /// DUtils::Random::RandomInt(min, max)
    pub fn random_int(&mut self, min: i32, max: i32) -> i32 {
        let d = max - min + 1;
        random_int_from(self.rand(), min, d)
    }
}

impl Default for GlibcRand {
    fn default() -> Self {
        Self::new(1)
    }
}

/// RandomInt 의 변환: int(((double)rand()/((double)RAND_MAX + 1.0)) * d) + min
#[inline]
pub fn random_int_from(r: i32, min: i32, d: i32) -> i32 {
    ((r as f64 / (2147483647.0f64 + 1.0)) * d as f64) as i32 + min
}
