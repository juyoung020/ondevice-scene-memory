//! librealsense v2.59.1 의 정렬 코드를 시험·비교용으로 옮겨 온 것(느리지만 SDK 와 같은 결과의 기준).
//!
//! 출처(Apache License 2.0, Copyright(c) 2015-2024 RealSense, Inc.):
//! - <https://github.com/realsenseai/librealsense/blob/v2.59.1/src/rs.cpp>  rs2_project_point_to_pixel,
//!   rs2_deproject_pixel_to_point, rs2_transform_point_to_point
//! - <https://github.com/realsenseai/librealsense/blob/v2.59.1/src/proc/align.cpp>  align_images, align_z_to_other
//! - <https://github.com/realsenseai/librealsense/blob/v2.59.1/src/proc/sse/sse-align.cpp>  (x86 SIMD 경로)
//! - <https://github.com/realsenseai/librealsense/blob/v2.59.1/src/proc/neon/neon-align.cpp>  (ARM NEON 경로)
//!
//! 바꾼 점: 프레임·프로파일 객체 대신 보정값과 버퍼를 직접 받음, OpenMP 줄 없음, SIMD 레인 연산을 화소마다 같은 순서의
//! 스칼라 연산으로 씀(레인끼리 섞지 않으므로 결과 같음), 정수 변환은 그 명령(x86 cvttps → 범위 밖 i32::MIN,
//! aarch64 fcvtzs → 포화)과 같게. 계산식과 연산 순서는 원본 그대로 둔다(비트 단위 비교용).
use crate::fast_align::{dist, fov_deg, Extrinsics, Intrinsics};

const NEON: bool = cfg!(target_arch = "aarch64");

/// C 의 (int)float 과 SIMD 정수 변환: x86 은 범위 밖·NaN 이 i32::MIN, aarch64 는 포화(NaN 은 0)
#[inline(always)]
fn cvt(v: f32) -> i32 {
    if NEON {
        v as i32
    } else {
        // 분기 없이(벡터화): 범위 안이면 자른 값, 아니면 i32::MIN
        let ok = v >= -2147483648.0 && v < 2147483648.0;
        let c = v.max(-2147483648.0).min(2147483520.0);
        // SAFETY: c 는 유한하고 i32 범위 안
        let i = unsafe { c.to_int_unchecked::<i32>() };
        if ok { i } else { i32::MIN }
    }
}

// ---- src/rs.cpp ----
pub fn rs2_project_point_to_pixel(intrin: &Intrinsics, point: &[f32; 3]) -> [f32; 2] {
    let mut x = point[0] / point[2];
    let mut y = point[1] / point[2];
    let c = &intrin.coeffs;
    if intrin.model == dist::MODIFIED_BROWN_CONRADY || intrin.model == dist::INVERSE_BROWN_CONRADY {
        let r2 = x * x + y * y;
        let f = 1.0 + c[0] * r2 + c[1] * r2 * r2 + c[4] * r2 * r2 * r2;
        x *= f;
        y *= f;
        let dx = x + 2.0 * c[2] * x * y + c[3] * (r2 + 2.0 * x * x);
        let dy = y + 2.0 * c[3] * x * y + c[2] * (r2 + 2.0 * y * y);
        x = dx;
        y = dy;
    }
    if intrin.model == dist::BROWN_CONRADY {
        let r2 = x * x + y * y;
        let f = 1.0 + c[0] * r2 + c[1] * r2 * r2 + c[4] * r2 * r2 * r2;
        let xf = x * f;
        let yf = y * f;
        let dx = xf + 2.0 * c[2] * x * y + c[3] * (r2 + 2.0 * x * x);
        let dy = yf + 2.0 * c[3] * x * y + c[2] * (r2 + 2.0 * y * y);
        x = dx;
        y = dy;
    }
    [x * intrin.fx + intrin.ppx, y * intrin.fy + intrin.ppy]
}

pub fn is_intrinsics_distortion_zero(i: &Intrinsics) -> bool {
    i.coeffs.iter().all(|c| c.abs() < f32::EPSILON)
}

pub fn rs2_deproject_pixel_to_point(intrin: &Intrinsics, pixel: &[f32; 2], depth: f32) -> [f32; 3] {
    let mut x = (pixel[0] - intrin.ppx) / intrin.fx;
    let mut y = (pixel[1] - intrin.ppy) / intrin.fy;
    let (xo, yo) = (x, y);
    let c = &intrin.coeffs;
    if !is_intrinsics_distortion_zero(intrin) {
        if intrin.model == dist::INVERSE_BROWN_CONRADY {
            for _ in 0..10 {
                let r2 = x * x + y * y;
                let icdist = 1.0f32 / (1.0 + ((c[4] * r2 + c[1]) * r2 + c[0]) * r2);
                let xq = x / icdist;
                let yq = y / icdist;
                let delta_x = 2.0 * c[2] * xq * yq + c[3] * (r2 + 2.0 * xq * xq);
                let delta_y = 2.0 * c[3] * xq * yq + c[2] * (r2 + 2.0 * yq * yq);
                x = (xo - delta_x) * icdist;
                y = (yo - delta_y) * icdist;
            }
        }
        if intrin.model == dist::BROWN_CONRADY {
            for _ in 0..10 {
                let r2 = x * x + y * y;
                let icdist = 1.0f32 / (1.0 + ((c[4] * r2 + c[1]) * r2 + c[0]) * r2);
                let delta_x = 2.0 * c[2] * x * y + c[3] * (r2 + 2.0 * x * x);
                let delta_y = 2.0 * c[3] * x * y + c[2] * (r2 + 2.0 * y * y);
                x = (xo - delta_x) * icdist;
                y = (yo - delta_y) * icdist;
            }
        }
    }
    [depth * x, depth * y, depth]
}

pub fn rs2_transform_point_to_point(e: &Extrinsics, p: &[f32; 3]) -> [f32; 3] {
    let r = &e.rotation;
    let t = &e.translation;
    [
        r[0] * p[0] + r[3] * p[1] + r[6] * p[2] + t[0],
        r[1] * p[0] + r[4] * p[1] + r[7] * p[2] + t[1],
        r[2] * p[0] + r[5] * p[1] + r[8] * p[2] + t[2],
    ]
}

// ---- src/proc/align.cpp (일반 경로) ----
pub fn align_z_to_other_generic(out_z: &mut [u16], z_pixels: &[u16], z_scale: f32, z_intrin: &Intrinsics, z_to_other: &Extrinsics, other: &Intrinsics) {
    out_z[..(other.width * other.height) as usize].fill(0);
    for depth_y in 0..z_intrin.height {
        let mut idx = (depth_y * z_intrin.width) as usize;
        for depth_x in 0..z_intrin.width {
            let depth = z_scale * z_pixels[idx] as f32;
            if depth != 0.0 {
                let p = rs2_deproject_pixel_to_point(z_intrin, &[depth_x as f32 - 0.5, depth_y as f32 - 0.5], depth);
                let o = rs2_transform_point_to_point(z_to_other, &p);
                let px = rs2_project_point_to_pixel(other, &o);
                let (x0, y0) = (cvt(px[0] + 0.5), cvt(px[1] + 0.5));
                let p = rs2_deproject_pixel_to_point(z_intrin, &[depth_x as f32 + 0.5, depth_y as f32 + 0.5], depth);
                let o = rs2_transform_point_to_point(z_to_other, &p);
                let px = rs2_project_point_to_pixel(other, &o);
                let (x1, y1) = (cvt(px[0] + 0.5), cvt(px[1] + 0.5));
                if !(x0 < 0 || y0 < 0 || x1 >= other.width || y1 >= other.height) {
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            let oi = (y * other.width + x) as usize;
                            let zv = z_pixels[idx];
                            out_z[oi] = if out_z[oi] != 0 { out_z[oi].min(zv) } else { zv };
                        }
                    }
                }
            }
            idx += 1;
        }
    }
}

// ---- src/proc/sse/sse-align.cpp · src/proc/neon/neon-align.cpp (SIMD 경로) ----
#[derive(Clone, Copy, Default)]
struct Int2 {
    x: i32,
    y: i32,
}

pub fn is_special_resolution(d: &Intrinsics, to: &Intrinsics) -> bool {
    (d.width == 640 && d.height == 240 && to.width == 320 && to.height == 180)
        || (d.width == 640 && d.height == 480 && to.width == 640 && to.height == 360)
}

/// distorte_x_y<RS2_DISTORTION_MODIFIED_BROWN_CONRADY>
#[inline(always)]
fn distort_mbc(x: f32, y: f32, c: &[f32; 5]) -> (f32, f32) {
    if NEON {
        let r2 = x * x + y * y;
        let f = r2.mul_add(r2.mul_add(r2.mul_add(c[4], c[1]), c[0]), 1.0);
        let (x_f, y_f) = (x * f, y * f);
        let dx = c[3].mul_add(r2, x_f * 2.0f32.mul_add(c[3].mul_add(x_f, c[2] * y_f), 1.0));
        let dy = c[2].mul_add(r2, y_f * 2.0f32.mul_add(c[2].mul_add(y_f, c[3] * x_f), 1.0));
        (dx, dy)
    } else {
        let r2_0 = x * x + y * y;
        let r3_0 = c[1] * (r2_0 * r2_0) + c[4] * (r2_0 * (r2_0 * r2_0));
        let f_0 = 1.0 + (c[0] * r2_0 + r3_0);
        let x_f0 = x * f_0;
        let y_f0 = y * f_0;
        let r4_0 = c[3] * (r2_0 + 2.0 * (x_f0 * x_f0));
        let d_x0 = x_f0 + (2.0 * (c[2] * (x_f0 * y_f0)) + r4_0);
        let r5_0 = c[2] * (r2_0 + 2.0 * (y_f0 * y_f0));
        let d_y0 = y_f0 + (2.0 * (c[3] * (x_f0 * y_f0)) + r5_0);
        (d_x0, d_y0)
    }
}

/// get_texture_map_simd: 화소마다 정수 좌표(깊이 0 이면 (0, 0))
fn get_texture_map(depth: &[u16], depth_scale: f32, mapx: &[f32], mapy: &[f32], out: &mut [Int2], to: &Intrinsics, e: &Extrinsics, mbc: bool) {
    let r = &e.rotation;
    let t = &e.translation;
    for i in 0..depth.len() {
        let d = depth[i] as i32 as f32 * depth_scale;
        let px = d * mapx[i];
        let py = d * mapy[i];
        let (mut x, mut y, z);
        if NEON {
            x = r[0].mul_add(px, r[3].mul_add(py, r[6].mul_add(d, t[0])));
            y = r[1].mul_add(px, r[4].mul_add(py, r[7].mul_add(d, t[1])));
            z = r[2].mul_add(px, r[5].mul_add(py, r[8].mul_add(d, t[2])));
        } else {
            x = r[0] * px + (r[3] * py + (r[6] * d + t[0]));
            y = r[1] * px + (r[4] * py + (r[7] * d + t[1]));
            z = r[2] * px + (r[5] * py + (r[8] * d + t[2]));
        }
        x /= z;
        y /= z;
        if mbc {
            (x, y) = distort_mbc(x, y, &to.coeffs);
        }
        let (u, v) = if NEON { (x.mul_add(to.fx, to.ppx), y.mul_add(to.fy, to.ppy)) } else { (x * to.fx + to.ppx, y * to.fy + to.ppy) };
        // SSE: cmpneq(depth, 0) 로 가림, NEON: cgt(depth, 0) 로 가림(깊이 단위가 양수라 같음)
        let keep = if NEON { d > 0.0 } else { d != 0.0 };
        out[i] = if keep { Int2 { x: cvt(u + 0.5), y: cvt(v + 0.5) } } else { Int2 { x: 0, y: 0 } };
    }
}

/// image_transform / align_neon_helper: 표를 한 번 만들고 장면마다 align_depth_to_other
pub struct SimdAlign {
    depth: Intrinsics,
    depth_scale: f32,
    x_tl: Vec<f32>,
    y_tl: Vec<f32>,
    x_br: Vec<f32>,
    y_br: Vec<f32>,
    tl: Vec<Int2>,
    br: Vec<Int2>,
    pub last_rect: bool,
}

impl SimdAlign {
    pub fn new(from: &Intrinsics, depth_scale: f32) -> SimdAlign {
        let n = (from.width * from.height) as usize;
        let mut s = SimdAlign {
            depth: *from,
            depth_scale,
            x_tl: vec![],
            y_tl: vec![],
            x_br: vec![],
            y_br: vec![],
            tl: vec![Int2::default(); n],
            br: vec![Int2::default(); n],
            last_rect: false,
        };
        (s.x_tl, s.y_tl) = s.pre_compute_x_y_map(-0.5);
        (s.x_br, s.y_br) = s.pre_compute_x_y_map(0.5);
        s
    }

    fn pre_compute_x_y_map(&self, offset: f32) -> (Vec<f32>, Vec<f32>) {
        let d = &self.depth;
        let n = (d.width * d.height) as usize;
        let (mut mx, mut my) = (vec![0.0f32; n], vec![0.0f32; n]);
        for h in 0..d.height {
            for w in 0..d.width {
                let pixel = [w as f32 + offset, h as f32 + offset];
                let mut x = (pixel[0] - d.ppx) / d.fx;
                let mut y = (pixel[1] - d.ppy) / d.fy;
                if d.model == dist::INVERSE_BROWN_CONRADY {
                    let c = &d.coeffs;
                    let r2 = x * x + y * y;
                    let f = if NEON { 1.0 + (c[0] + (c[1] + c[4] * r2) * r2) * r2 } else { 1.0 + c[0] * r2 + c[1] * r2 * r2 + c[4] * r2 * r2 * r2 };
                    let ux = x * f + 2.0 * c[2] * x * y + c[3] * (r2 + 2.0 * x * x);
                    let uy = y * f + 2.0 * c[3] * x * y + c[2] * (r2 + 2.0 * y * y);
                    x = ux;
                    y = uy;
                }
                mx[(h * d.width + w) as usize] = x;
                my[(h * d.width + w) as usize] = y;
            }
        }
        (mx, my)
    }

    pub fn align_depth_to_other(&mut self, z_pixels: &[u16], dest: &mut [u16], to: &Intrinsics, e: &Extrinsics) {
        dest[..(to.width * to.height) as usize].fill(0);
        let mbc = to.model == dist::MODIFIED_BROWN_CONRADY;
        let n = (self.depth.width * self.depth.height) as usize;
        let z = &z_pixels[..n];
        get_texture_map(z, self.depth_scale, &self.x_tl, &self.y_tl, &mut self.tl, to, e, mbc);
        let fov = fov_deg(&self.depth);
        let ppa_d = [self.depth.width as f32 / fov[0], self.depth.height as f32 / fov[1]];
        let fov = fov_deg(to);
        let ppa_t = [to.width as f32 / fov[0], to.height as f32 / fov[1]];
        if ppa_d[0] < ppa_t[0] || ppa_d[1] < ppa_t[1] || is_special_resolution(&self.depth, to) {
            self.last_rect = true;
            get_texture_map(z, self.depth_scale, &self.x_br, &self.y_br, &mut self.br, to, e, mbc);
            move_depth_to_other(&self.depth, z, dest, to, &self.tl, &self.br);
        } else {
            self.last_rect = false;
            move_depth_to_other(&self.depth, z, dest, to, &self.tl, &self.tl);
        }
    }
}

fn move_depth_to_other(d: &Intrinsics, z: &[u16], dest: &mut [u16], to: &Intrinsics, tl: &[Int2], br: &[Int2]) {
    for i in 0..(d.width * d.height) as usize {
        let zz = z[i];
        if zz == 0 {
            continue;
        }
        // 원본은 화면 밖 줄·열을 하나씩 건너뜀: 같은 화소만 쓰도록 범위를 화면으로 묶어 돈다
        for oy in tl[i].y.max(0)..=br[i].y.min(to.height - 1) {
            for ox in tl[i].x.max(0)..=br[i].x.min(to.width - 1) {
                let oi = (oy * to.width + ox) as usize;
                dest[oi] = if dest[oi] != 0 { dest[oi].min(zz) } else { zz };
            }
        }
    }
}
