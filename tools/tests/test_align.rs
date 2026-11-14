//! 정확성 시험(`cargo test --release -- --nocapture` 로 표를 봄):
//!  A) 같은 입력에서 fast_align 이 옮겨 온 librealsense 코드(rsref)와 화소 하나까지 같은지(Sdk ↔ SIMD 경로, Generic ↔ 일반 경로),
//!     스레드 1·3·4 개 모두.
//!  B) 깊이로 확인 가능한 합성 장면(정면 평면·계단 경계·방)에서 정답(컬러 화소가 보는 면의 깊이)과 비교.
//!  C) 컬러 왜곡을 무시할 때 옮겨지는 거리.
use tools::common::lpad;
use tools::fast_align::{dist, DepthAligner, Extrinsics, Intrinsics, Method, Options};
use tools::rsref;
use tools::scenes::{self, *};

struct T {
    fail: i32,
}

impl T {
    fn check(&mut self, cond: bool, msg: String) {
        if !cond {
            println!("  실패: {msg}");
            self.fail += 1;
        }
    }
}

fn count_diff(a: &[u16], b: &[u16]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

struct Case {
    name: &'static str,
    d: Intrinsics,
    c: Intrinsics,
    e: Extrinsics,
}

fn yn(b: bool) -> &'static str {
    if b { "예" } else { "아니오" }
}

fn exact_case(t: &mut T, cs: &Case, z: &[u16], zname: &str) {
    let unit = 0.001f32;
    let n = (cs.c.width * cs.c.height) as usize;
    // SIMD 경로 기준
    let (mut rf, mut out, mut gref) = (vec![0u16; n], vec![0u16; n], vec![0u16; n]);
    let mut simd = rsref::SimdAlign::new(&cs.d, unit);
    simd.align_depth_to_other(z, &mut rf, &cs.c, &cs.e);
    rsref::align_z_to_other_generic(&mut gref, z, unit, &cs.d, &cs.e, &cs.c);
    for th in [1, 3, 4] {
        let mut o = Options { threads: th, method: Method::Sdk, ..Default::default() };
        let mut a = DepthAligner::new(&cs.d, &cs.c, &cs.e, unit, &o).unwrap();
        a.process(z, &mut out);
        let nd = count_diff(&out, &rf);
        println!(
            "  {} {} Sdk     스레드 {}: 다른 화소 {} / {} (사각형 {}, SDK 사각형 {}{})",
            lpad(cs.name, 34),
            lpad(zname, 6),
            th,
            nd,
            n,
            yn(a.uses_rect()),
            yn(simd.last_rect),
            if a.last_fell_back_serial() { ", 한 스레드로 다시" } else { "" }
        );
        t.check(nd == 0, format!("{} {} Sdk T={} 차이 {}", cs.name, zname, th, nd));
        t.check(a.uses_rect() == simd.last_rect, format!("{} 사각형 판단 다름", cs.name));
        o.method = Method::Generic;
        let mut g = DepthAligner::new(&cs.d, &cs.c, &cs.e, unit, &o).unwrap();
        g.process(z, &mut out);
        let nd = count_diff(&out, &gref);
        println!("  {} {} Generic 스레드 {}: 다른 화소 {} / {}", lpad(cs.name, 34), lpad(zname, 6), th, nd, n);
        t.check(nd == 0, format!("{} {} Generic T={} 차이 {}", cs.name, zname, th, nd));
    }
    // 빠른 식(exact=false): SDK 와 다른 화소 비율과, 다른 곳의 값 차이
    let o = Options { exact: false, ..Default::default() };
    let mut a = DepthAligner::new(&cs.d, &cs.c, &cs.e, unit, &o).unwrap();
    a.process(z, &mut out);
    let nd = count_diff(&out, &rf);
    let nz = out.iter().zip(&rf).filter(|(x, y)| (**x == 0) != (**y == 0)).count();
    println!(
        "  {} {} Sdk 빠른 식      : 다른 화소 {} ({:.4} %), 그중 한쪽만 빈 화소 {}",
        lpad(cs.name, 34),
        lpad(zname, 6),
        nd,
        100.0 * nd as f64 / n as f64,
        nz
    );
    t.check(nd * 200 <= n, format!("{} {} 빠른 식 차이 {} 가 0.5 % 넘음", cs.name, zname, nd));
}

/// 정답 비교 통계
#[derive(Default)]
struct GeoStat {
    truth: usize,
    filled: usize,
    ok: usize,
    hole: usize,
    bleed: usize,
    edge_n: usize,
    edge_ok: usize,
}

fn geo_compare(out: &[u16], truth: &[f32], ci: &Intrinsics, unit: f32) -> GeoStat {
    let mut s = GeoStat::default();
    let (w, h) = (ci.width as usize, ci.height as usize);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let t = truth[i];
            if t <= 0.0 {
                continue;
            }
            // 경계: 3×3 안 정답이 5 cm 넘게 다르거나 없음
            let mut edge = false;
            for dy in 0..3 {
                for dx in 0..3 {
                    let tn = truth[(y + dy - 1) * w + x + dx - 1];
                    if tn <= 0.0 || (tn - t).abs() > 0.05 {
                        edge = true;
                    }
                }
            }
            let o = out[i] as f32 * unit;
            let tol = if 0.003f32 < 0.01 * t { 0.01 * t } else { 0.003 };
            let good = out[i] != 0 && (o - t).abs() <= tol;
            if edge {
                s.edge_n += 1;
                s.edge_ok += good as usize;
                continue;
            }
            s.truth += 1;
            if out[i] == 0 {
                s.hole += 1;
                continue;
            }
            s.filled += 1;
            if good {
                s.ok += 1;
            } else if o > t + 0.05 {
                s.bleed += 1; // 뒤쪽 값이 앞을 덮음
            }
        }
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn geo_case(t: &mut T, name: &str, sc: &Scene, di: &Intrinsics, ci: &Intrinsics, e: &Extrinsics, min_ok: f64, max_hole_sdk: f64, sdk_strict: bool) {
    let unit = 0.001f32;
    let z = render_depth(sc, di, unit, 0.0, 0.0, 1);
    let truth = truth_in_color(sc, di, ci, e);
    let mut out = vec![0u16; (ci.width * ci.height) as usize];
    for variant in 0..3 {
        if variant == 2 && sdk_strict {
            break; // 컬러 왜곡 켠 Sdk 는 C 에서만
        }
        let m = if variant == 1 { Method::Generic } else { Method::Sdk };
        let o = Options { method: m, color_distortion: variant == 2, ..Default::default() };
        let mut a = DepthAligner::new(di, ci, e, unit, &o).unwrap();
        a.process(&z, &mut out);
        let s = geo_compare(&out, &truth, ci, unit);
        let okr = if s.filled > 0 { s.ok as f64 / s.filled as f64 } else { 0.0 };
        let holer = if s.truth > 0 { s.hole as f64 / s.truth as f64 } else { 0.0 };
        let edger = if s.edge_n > 0 { s.edge_ok as f64 / s.edge_n as f64 } else { 1.0 };
        println!(
            "  {} {} 안쪽 {} 화소: 값 맞음 {:.4}, 구멍 {:.4}, 뒤값 덮음 {} | 경계 {} 화소 중 맞음 {:.3}",
            lpad(name, 26),
            lpad(["Sdk", "Generic", "Sdk+왜곡"][variant], 7),
            s.truth,
            okr,
            holer,
            s.bleed,
            s.edge_n,
            edger
        );
        if m == Method::Generic || sdk_strict || variant == 2 {
            t.check(okr >= min_ok, format!("{name} 값 맞음 {okr:.4} < {min_ok:.4}"));
        }
        if m == Method::Sdk {
            t.check(holer <= max_hole_sdk, format!("{name} Sdk 구멍 {holer:.4} > {max_hole_sdk:.4}"));
        }
        if m == Method::Generic {
            t.check(holer <= 0.001, format!("{name} Generic 구멍 {holer:.4}"));
        }
    }
}

#[test]
fn test_align() {
    let mut t = T { fail: 0 };
    println!("== A. librealsense 이식본과 화소 단위 비교 ==");
    let mut c640_nodist = d455_color_640();
    c640_nodist.model = dist::NONE;
    let mut c640_mbc = d455_color_640();
    c640_mbc.model = dist::MODIFIED_BROWN_CONRADY;
    let mut d640_ibc = d455_depth_640(); // 깊이에 왜곡이 있는 경우(표가 화소마다)
    d640_ibc.model = dist::INVERSE_BROWN_CONRADY;
    d640_ibc.coeffs[0] = 0.02;
    d640_ibc.coeffs[1] = -0.01;
    d640_ibc.coeffs[2] = 0.001;
    let c640x360 = make_intrin(640, 360, 320.0, 320.0, 320.0, 180.0, dist::NONE, &[]);
    let d640_hi = make_intrin(640, 480, 300.0, 300.0, 320.0, 240.0, dist::NONE, &[]); // 깊이가 더 넓게 봄 → 사각형 채움
    let cases = [
        Case { name: "D455 640x480 → 640x480(IBC)", d: d455_depth_640(), c: d455_color_640(), e: d455_extrin() },
        Case { name: "D455 848x480 → 640x480(IBC)", d: d455_depth_848(), c: d455_color_640(), e: d455_extrin() },
        Case { name: "D455 640x480 → 1280x720(IBC)", d: d455_depth_640(), c: d455_color_1280x720(), e: d455_extrin() },
        Case { name: "640x480 → 640x480 왜곡 없음", d: d455_depth_640(), c: c640_nodist, e: d455_extrin() },
        Case { name: "640x480 → 640x480 MBC", d: d455_depth_640(), c: c640_mbc, e: d455_extrin() },
        Case { name: "깊이 IBC → 640x480(화소별 표)", d: d640_ibc, c: d455_color_640(), e: d455_extrin() },
        Case { name: "640x480 → 640x360(특수 해상도)", d: d455_depth_640(), c: c640x360, e: d455_extrin() },
        Case { name: "넓은 깊이 → 640x480(사각형)", d: d640_hi, c: c640_mbc, e: d455_extrin() },
    ];
    for cs in &cases {
        let z1 = render_depth(&room(), &cs.d, 0.001, 0.004, 0.03, 7);
        exact_case(&mut t, cs, &z1, "방");
        let z2 = random_depth(&cs.d, 11);
        exact_case(&mut t, cs, &z2, "무작위");
    }

    println!("\n== B. 합성 장면 정답 비교(컬러 왜곡 없음, 정답 = 컬러 화소 중심이 보는 면의 깊이 카메라 z) ==");
    // 정면 평면 2 m, 옆으로 59 mm: 모든 값이 2000
    {
        let (di, ci) = (d455_depth_640(), c640_nodist);
        let z = render_depth(&front_plane(2.0), &di, 0.001, 0.0, 0.0, 1);
        let mut out = vec![0u16; (ci.width * ci.height) as usize];
        for m in [Method::Sdk, Method::Generic] {
            let o = Options { method: m, ..Default::default() };
            let mut a = DepthAligner::new(&di, &ci, &shift_x(-0.059), 0.001, &o).unwrap();
            a.process(&z, &mut out);
            let filled = out.iter().filter(|&&v| v != 0).count();
            let wrong = out.iter().filter(|&&v| v != 0 && v != 2000).count();
            println!(
                "  정면 평면 2 m {}: 채움 {:.4}, 2000 아닌 값 {}",
                lpad(if m == Method::Sdk { "Sdk" } else { "Generic" }, 7),
                filled as f64 / out.len() as f64,
                wrong
            );
            t.check(wrong == 0, format!("정면 평면 값 다름 {wrong}"));
        }
    }
    geo_case(&mut t, "계단(1 m 판 / 2 m 벽)", &step(), &d455_depth_640(), &c640_nodist, &d455_extrin(), 0.995, 0.08, true);
    geo_case(&mut t, "방(벽·바닥·상자)", &room(), &d455_depth_640(), &c640_nodist, &d455_extrin(), 0.99, 0.08, true);
    geo_case(&mut t, "방 848x480 깊이", &room(), &d455_depth_848(), &c640_nodist, &d455_extrin(), 0.99, 0.02, true);

    println!("\n== C. D455 컬러 왜곡(Inverse Brown-Conrady) 그대로: Sdk 는 컬러 왜곡을 무시(SDK SIMD 경로와 같음) ==");
    geo_case(&mut t, "방, 컬러 IBC", &room(), &d455_depth_640(), &d455_color_640(), &d455_extrin(), 0.99, 0.08, false);
    // 컬러 왜곡을 무시할 때 화소가 얼마나 옮겨지는지(같은 광선을 왜곡 넣고/빼고 투영)
    {
        let ci = scenes::d455_color_640();
        let mut cn = ci;
        cn.model = dist::NONE;
        let (mut mx, mut sum) = (0.0f64, 0.0f64);
        let (mut n, mut over1) = (0usize, 0usize);
        for y in (0..ci.height).step_by(4) {
            for x in (0..ci.width).step_by(4) {
                let ray = rsref::rs2_deproject_pixel_to_point(&ci, &[x as f32, y as f32], 1.0);
                let a = rsref::rs2_project_point_to_pixel(&ci, &ray);
                let b = rsref::rs2_project_point_to_pixel(&cn, &ray);
                let d = f64::hypot((a[0] - b[0]) as f64, (a[1] - b[1]) as f64);
                mx = if mx < d { d } else { mx };
                sum += d;
                n += 1;
                over1 += (d > 1.0) as usize;
            }
        }
        println!(
            "  컬러 왜곡 무시로 옮겨지는 거리(640x480 가정 계수): 평균 {:.2} 화소, 최대 {:.2} 화소, 1 화소 넘는 곳 {:.1} %",
            sum / n as f64,
            mx,
            100.0 * over1 as f64 / n as f64
        );
    }

    println!("\n{} (실패 {})", if t.fail > 0 { "실패" } else { "모두 통과" }, t.fail);
    assert_eq!(t.fail, 0);
}
