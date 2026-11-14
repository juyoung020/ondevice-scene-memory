//! ICM-42688-P 연결 확인: I2C 로 칩을 찾아 가속도·자이로를 읽고, 가만히 둔 동안의 중력 방향과 자이로 치우침을 찍는다.
//!   imu_probe [/dev/i2c-1] [초=5]
//! 배선(Pi 5 핀 번호): VCC → 1(3.3 V), GND → 6, SDA → 3, SCL → 5. I2C 켜기: sudo dtparam i2c_arm=on (재부팅 없이 적용)
use std::io::{Read, Write};
use std::time::{Duration, Instant};
use tools::common::atof;

/// 레지스터 쓰기·읽기(I2C 장치 하나)
trait Bus {
    /// 이 주소의 장치로 바꿈(I2C_SLAVE)
    fn select(&mut self, addr: u16) -> bool;
    fn write(&mut self, b: &[u8]) -> bool;
    fn read(&mut self, out: &mut [u8]) -> bool;
}

fn wr(bus: &mut dyn Bus, reg: u8, v: u8) -> bool {
    bus.write(&[reg, v])
}
fn rd(bus: &mut dyn Bus, reg: u8, out: &mut [u8]) -> bool {
    bus.write(&[reg]) && bus.read(out)
}

/// 12 바이트(가속도 x y z, 자이로 x y z, 큰 끝 i16) → m/s², °/s
fn convert(b: &[u8; 12]) -> ([f64; 3], [f64; 3]) {
    let acc_s = 16.0 * 9.80665 / 32768.0;
    let gyr_s = 2000.0 / 32768.0;
    let mut a = [0.0; 3];
    let mut g = [0.0; 3];
    for i in 0..3 {
        a[i] = i16::from_be_bytes([b[2 * i], b[2 * i + 1]]) as f64 * acc_s;
        g[i] = i16::from_be_bytes([b[6 + 2 * i], b[7 + 2 * i]]) as f64 * gyr_s;
    }
    (a, g)
}

/// 찾기·설정·읽기. 끝 코드(0 = 성공)와 찍을 줄을 out 에
fn probe(bus: &mut dyn Bus, secs: f64, out: &mut dyn FnMut(bool, String), sleep: &mut dyn FnMut(Duration), clock: &mut dyn FnMut() -> f64) -> i32 {
    let mut addr = 0u16;
    for a in [0x68u16, 0x69] {
        let mut who = [0u8];
        if bus.select(a) && rd(bus, 0x75, &mut who) {
            out(false, format!("주소 0x{:02x}: WHO_AM_I 0x{:02x}{}", a, who[0], if who[0] == 0x47 { " (ICM-42688-P)" } else { "" }));
            if who[0] == 0x47 {
                addr = a;
            }
        }
    }
    if addr == 0 {
        out(true, "ICM-42688-P 를 찾지 못함(0x68·0x69). 배선과 `i2cdetect -y 1` 을 확인".into());
        return 2;
    }
    bus.select(addr);
    wr(bus, 0x11, 0x01); // 소프트 리셋
    sleep(Duration::from_millis(5));
    wr(bus, 0x4F, 0x07); // 자이로 ±2000 °/s, 200 Hz
    wr(bus, 0x50, 0x07); // 가속도 ±16 g, 200 Hz
    wr(bus, 0x4E, 0x0F); // 자이로·가속도 켬(저잡음)
    sleep(Duration::from_millis(50));

    let (mut sa, mut sg) = ([0.0f64; 3], [0.0f64; 3]);
    let (mut n, mut fail) = (0i64, 0i64);
    let t0 = clock();
    let mut next = t0;
    while clock() - t0 < secs {
        let mut b = [0u8; 12];
        if rd(bus, 0x1F, &mut b) {
            let (a, g) = convert(&b);
            for i in 0..3 {
                sa[i] += a[i];
                sg[i] += g[i];
            }
            if n % 100 == 0 {
                out(false, format!("가속도 {:7.2} {:7.2} {:7.2} m/s²  자이로 {:7.2} {:7.2} {:7.2} °/s", a[0], a[1], a[2], g[0], g[1], g[2]));
            }
            n += 1;
        } else {
            fail += 1;
        }
        next += 0.005;
        let now = clock();
        if next > now {
            sleep(Duration::from_secs_f64(next - now));
        }
    }
    if n == 0 {
        out(true, format!("읽기 실패 {fail} 번"));
        return 3;
    }
    let nf = n as f64;
    let (ax, ay, az) = (sa[0] / nf, sa[1] / nf, sa[2] / nf);
    let norm = (ax * ax + ay * ay + az * az).sqrt();
    out(false, format!("표본 {} 개({:.0} Hz), 읽기 실패 {}", n, nf / secs, fail));
    out(false, format!("평균 가속도 {ax:.3} {ay:.3} {az:.3}, 크기 {norm:.3} m/s²(가만히 두면 9.81)"));
    out(false, format!("중력 방향(단위) {:.3} {:.3} {:.3}", ax / norm, ay / norm, az / norm));
    out(false, format!("자이로 치우침 {:.3} {:.3} {:.3} °/s", sg[0] / nf, sg[1] / nf, sg[2] / nf));
    0
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::os::fd::AsRawFd;
    unsafe extern "C" {
        fn ioctl(fd: i32, req: std::ffi::c_ulong, ...) -> i32;
    }
    const I2C_SLAVE: std::ffi::c_ulong = 0x0703;

    pub struct I2c(pub std::fs::File);
    impl Bus for I2c {
        fn select(&mut self, addr: u16) -> bool {
            unsafe { ioctl(self.0.as_raw_fd(), I2C_SLAVE, addr as std::ffi::c_ulong) == 0 }
        }
        fn write(&mut self, b: &[u8]) -> bool {
            self.0.write(b).is_ok_and(|n| n == b.len())
        }
        fn read(&mut self, out: &mut [u8]) -> bool {
            self.0.read(out).is_ok_and(|n| n == out.len())
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dev = args.get(1).map_or("/dev/i2c-1", |s| s.as_str());
    let secs = args.get(2).map_or(5.0, |s| atof(s));
    #[cfg(target_os = "linux")]
    {
        let f = match std::fs::OpenOptions::new().read(true).write(true).open(dev) {
            Ok(f) => f,
            Err(e) => {
                // perror 와 같은 모양: "<장치>: <strerror>"
                let msg = e.to_string();
                let msg = msg.split(" (os error").next().unwrap_or(&msg).to_string();
                eprintln!("{dev}: {msg}");
                std::process::exit(1);
            }
        };
        let mut bus = linux::I2c(f);
        let t0 = Instant::now();
        let rc = probe(
            &mut bus,
            secs,
            &mut |err, line| {
                if err {
                    eprintln!("{line}");
                } else {
                    println!("{line}");
                }
            },
            &mut std::thread::sleep,
            &mut || t0.elapsed().as_secs_f64(),
        );
        std::process::exit(rc);
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (dev, secs, Instant::now());
        eprintln!("imu_probe: Linux I2C 전용");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 가짜 칩: 0x69 에 ICM-42688-P, 0x1F 를 읽으면 정해진 12 바이트, 쓴 레지스터를 기록
    struct Fake {
        addr: u16,
        reg: u8,
        writes: Vec<(u16, u8, u8)>,
        sample: [u8; 12],
    }
    impl Bus for Fake {
        fn select(&mut self, addr: u16) -> bool {
            self.addr = addr;
            true
        }
        fn write(&mut self, b: &[u8]) -> bool {
            self.reg = b[0];
            if b.len() == 2 {
                self.writes.push((self.addr, b[0], b[1]));
            }
            self.addr == 0x69
        }
        fn read(&mut self, out: &mut [u8]) -> bool {
            match self.reg {
                0x75 => out[0] = 0x47,
                0x1F => out.copy_from_slice(&self.sample),
                _ => return false,
            }
            true
        }
    }

    #[test]
    fn fake_chip() {
        // 가속도 (0, 0, +1 g) = 2048, 자이로 (0.5, -0.25, 0) °/s ≈ 8, -4, 0
        let sample = [0, 0, 0, 0, 0x08, 0x00, 0, 8, 0xFF, 0xFC, 0, 0];
        let mut bus = Fake { addr: 0, reg: 0, writes: vec![], sample };
        let mut lines = vec![];
        let mut t = 0.0f64;
        let mut slept = Duration::ZERO;
        let rc = probe(&mut bus, 1.0, &mut |e, l| lines.push((e, l)), &mut |d| slept += d, &mut || {
            t += 0.0025;
            t
        });
        assert_eq!(rc, 0);
        assert_eq!(bus.writes, vec![(0x69, 0x11, 0x01), (0x69, 0x4F, 0x07), (0x69, 0x50, 0x07), (0x69, 0x4E, 0x0F)]);
        assert_eq!(lines[0], (false, "주소 0x69: WHO_AM_I 0x47 (ICM-42688-P)".to_string()));
        assert_eq!(lines[1].1, "가속도    0.00    0.00    9.81 m/s²  자이로    0.49   -0.24    0.00 °/s");
        let tail: Vec<&str> = lines.iter().rev().take(4).map(|l| l.1.as_str()).collect();
        assert_eq!(tail[3], "표본 200 개(200 Hz), 읽기 실패 0");
        assert_eq!(tail[2], "평균 가속도 0.000 0.000 9.807, 크기 9.807 m/s²(가만히 두면 9.81)");
        assert_eq!(tail[1], "중력 방향(단위) 0.000 0.000 1.000");
        assert_eq!(tail[0], "자이로 치우침 0.488 -0.244 0.000 °/s");
    }

    #[test]
    fn no_chip() {
        struct Dead;
        impl Bus for Dead {
            fn select(&mut self, _: u16) -> bool {
                false
            }
            fn write(&mut self, _: &[u8]) -> bool {
                false
            }
            fn read(&mut self, _: &mut [u8]) -> bool {
                false
            }
        }
        let mut lines = vec![];
        let rc = probe(&mut Dead, 1.0, &mut |e, l| lines.push((e, l)), &mut |_| {}, &mut || 0.0);
        assert_eq!(rc, 2);
        assert!(lines[0].0);
    }
}
