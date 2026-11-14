//! D400 보정표 해석: librealsense v2.59.1 `src/ds/d400/d400-private.cpp`·`src/pose.h`(Apache-2.0)의 같은 계산을 같은 f32 연산 순서로 옮김.
//! - 깊이 내부 보정: 새 표(RECPARAMSGET, 펌웨어 5.11.9.5 이상)에 해상도가 있으면 그것, 없으면 계수표(GETINTCAL 25)의 rect_params
//! - 컬러 내부 보정: RGB 표(GETINTCAL 32)의 정규화 행렬을 화면비로 맞춤, 왜곡 = Inverse Brown-Conrady + 표의 계수
//! - 깊이→컬러: RGB 표의 rotation_matrix_rect·translation_rect(mm → m, 부호 반대) = 컬러→깊이, 그 역
pub use tools::fast_align::{Extrinsics, Intrinsics, dist};

pub const HEADER: usize = 16;
pub const COEFF_TABLE_SIZE: usize = 576;
pub const RGB_TABLE_SIZE: usize = 256;

/// rsutils::number::calc_crc32 (다항식 0xEDB88320, 초기값 0xFFFFFFFF, 끝에서 반전)
pub fn crc32(buf: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in buf {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
        }
    }
    !c
}

fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

/// check_calib: 머리 16 바이트, table_size 가 응답 안에 들어감, 머리 뒤 전체의 CRC32 = 머리의 crc32
pub fn check_calib(raw: &[u8], struct_size: usize) -> Result<(), String> {
    if raw.len() < HEADER {
        return Err(format!("보정표가 너무 짧음: {} 바이트", raw.len()));
    }
    let table_size = u32_at(raw, 4) as usize;
    if table_size + HEADER > raw.len() {
        return Err(format!("보정표 크기 {} 가 응답 {} 바이트를 넘음", table_size, raw.len() - HEADER));
    }
    if u32_at(raw, 12) != crc32(&raw[HEADER..]) {
        return Err("보정표 CRC 오류".into());
    }
    if raw.len() < struct_size {
        return Err(format!("보정표가 구조체({struct_size} 바이트)보다 짧음: {}", raw.len()));
    }
    Ok(())
}

/// ds_rect_resolutions 번호 순서의 (가로, 세로). 10·11 은 예약(목록에 없음)
const RECT_RES: [(u32, u32); 16] = [
    (1920, 1080),
    (1280, 720),
    (640, 480),
    (848, 480),
    (640, 360),
    (424, 240),
    (320, 240),
    (480, 270),
    (1280, 800),
    (960, 540),
    (0, 0),
    (0, 0),
    (640, 400),
    (576, 576),
    (720, 720),
    (1152, 1152),
];

fn rect_index(w: u32, h: u32) -> Option<usize> {
    RECT_RES.iter().position(|&(rw, rh)| rw != 0 && rw == w && rh == h)
}

/// get_d400_intrinsic_by_resolution_coefficients_table
pub fn depth_intrinsics_coeff(raw: &[u8], width: u32, height: u32) -> Result<Intrinsics, String> {
    check_calib(raw, COEFF_TABLE_SIZE)?;
    let rect = |r: usize| -> [f32; 4] {
        let o = 256 + r * 16;
        [f32_at(raw, o), f32_at(raw, o + 4), f32_at(raw, o + 8), f32_at(raw, o + 12)]
    };
    let mut res = rect_index(width, height);
    if width == 848 && height == 100 {
        res = rect_index(848, 480);
    }
    let mut k = Intrinsics { model: dist::BROWN_CONRADY, ..Default::default() };
    if let Some(r) = res {
        k.width = RECT_RES[r].0 as i32;
        k.height = RECT_RES[r].1 as i32;
        let mut p = rect(r);
        if p[0] == p[1] && p[2] == p[3] {
            p[0] = k.width as f32 * 1.5f32;
            p[1] = p[0];
            p[2] = k.width as f32 * 0.5f32;
            p[3] = k.height as f32 * 0.5f32;
        }
        k.fx = p[0];
        k.fy = p[1];
        k.ppx = p[2];
        k.ppy = p[3];
        if width == 848 && height == 100 {
            k.height = 100;
            k.ppy -= 190.0;
        }
    } else {
        k.width = width as i32;
        k.height = height as i32;
        let mut p = rect(0);
        if p[0] == p[1] && p[2] == p[3] {
            p[0] = k.width as f32 * 1.5f32;
            p[1] = p[0];
            p[2] = k.width as f32 * 0.5f32;
            p[3] = k.height as f32 * 0.5f32;
        }
        if width == 256 && height == 144 {
            k.fx = p[0];
            k.fy = p[1];
            k.ppx = p[2] - 832.0;
            k.ppy = p[3] - 468.0;
        } else {
            k.fx = p[0] * width as f32 / 1920.0;
            k.fy = p[1] * height as f32 / 1080.0;
            k.ppx = p[2] * width as f32 / 1920.0;
            k.ppy = p[3] * height as f32 / 1080.0;
        }
    }
    Ok(k)
}

/// try_get_d400_intrinsic_by_resolution_new: 20 바이트 항목(u16 w, u16 h, fx, fy, ppx, ppy)의 나열
pub fn depth_intrinsics_new(raw: &[u8], width: u32, height: u32) -> Option<Intrinsics> {
    for it in raw.chunks_exact(20) {
        if u16_at(it, 0) as u32 == width && u16_at(it, 2) as u32 == height {
            return Some(Intrinsics {
                width: width as i32,
                height: height as i32,
                fx: f32_at(it, 4),
                fy: f32_at(it, 8),
                ppx: f32_at(it, 12),
                ppy: f32_at(it, 16),
                model: dist::BROWN_CONRADY,
                coeffs: [0.0; 5],
            });
        }
    }
    None
}

/// d400_depth_sensor::get_intrinsics: 새 표 먼저, 없으면 계수표
pub fn depth_intrinsics(new_table: &[u8], coeff: &[u8], width: u32, height: u32) -> Result<Intrinsics, String> {
    match depth_intrinsics_new(new_table, width, height) {
        Some(k) => Ok(k),
        None => depth_intrinsics_coeff(coeff, width, height),
    }
}

/// get_d400_color_stream_intrinsic. 행렬은 열 우선 float3x3(i,j) = 평면 [j*3+i]
pub fn color_intrinsics(raw: &[u8], width: u32, height: u32) -> Result<Intrinsics, String> {
    check_calib(raw, RGB_TABLE_SIZE)?;
    let mut m = [0f32; 9];
    for (i, v) in m.iter_mut().enumerate() {
        *v = f32_at(raw, 16 + i * 4);
    }
    let (cw, ch) = (u16_at(raw, 144), u16_at(raw, 146));
    let mut calib_aspect = 9.0f32 / 16.0f32;
    if cw != 0 && ch != 0 {
        calib_aspect = ch as f32 / cw as f32;
    }
    let actual = height as f32 / width as f32;
    if actual < calib_aspect {
        m[4] *= calib_aspect / actual; // (1,1)
        m[5] *= calib_aspect / actual; // (2,1)
    } else {
        m[0] *= actual / calib_aspect; // (0,0)
        m[2] *= actual / calib_aspect; // (2,0)
    }
    let mut coeffs = [0f32; 5];
    for (i, c) in coeffs.iter_mut().enumerate() {
        *c = f32_at(raw, 52 + i * 4);
    }
    Ok(Intrinsics {
        width: width as i32,
        height: height as i32,
        ppx: ((1.0 + m[2]) * width as f32) / 2.0,
        ppy: ((1.0 + m[5]) * height as f32) / 2.0,
        fx: m[0] * width as f32 / 2.0,
        fy: m[4] * height as f32 / 2.0,
        model: dist::INVERSE_BROWN_CONRADY,
        coeffs,
    })
}

/// get_d400_color_stream_extrinsic → from_pose: 컬러 → 깊이
pub fn color_to_depth(raw: &[u8]) -> Result<Extrinsics, String> {
    check_calib(raw, RGB_TABLE_SIZE)?;
    let mut e = Extrinsics::default();
    for i in 0..9 {
        e.rotation[i] = f32_at(raw, 184 + i * 4);
    }
    for i in 0..3 {
        e.translation[i] = f32_at(raw, 220 + i * 4) * -0.001f32;
    }
    Ok(e)
}

/// pose.h inverse(rs2_extrinsics): {Rᵀ, Rᵀ·(t·-1)}. float3x3·float3 = (x·b.x + y·b.y) + z·b.z 를 성분마다.
/// librealsense v2.59.1 은 모든 리눅스 빌드(aarch64 포함)를 `-ffp-contract=off` 로 하므로 곱·더하기를 합치지 않는다(Rust 도 합치지 않음).
pub fn inverse(e: &Extrinsics) -> Extrinsics {
    let r = &e.rotation;
    // transpose: inv 열 k = 원래 행 k
    let inv = [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];
    let q = [e.translation[0] * -1.0f32, e.translation[1] * -1.0f32, e.translation[2] * -1.0f32];
    let mut t = [0f32; 3];
    for k in 0..3 {
        t[k] = mat_vec_component(inv[k], inv[3 + k], inv[6 + k], q);
    }
    Extrinsics { rotation: inv, translation: t }
}

fn mat_vec_component(x: f32, y: f32, z: f32, q: [f32; 3]) -> f32 {
    (x * q[0] + y * q[1]) + z * q[2]
}

/// 깊이 → 컬러 (librealsense 가 컬러→깊이 간선을 뒤집어 돌려주는 값)
pub fn depth_to_color(rgb_table: &[u8]) -> Result<Extrinsics, String> {
    Ok(inverse(&color_to_depth(rgb_table)?))
}

/// depth_scale_option::query: (float)(0.000001 * (float)depth_units)
pub fn depth_scale_from_units(depth_units_um: u32) -> f32 {
    (0.000001f64 * (depth_units_um as f32) as f64) as f32
}

/// 장치 정보(GVD 응답)
#[derive(Clone, Debug, Default)]
pub struct Gvd {
    pub serial: String,
    pub asic_serial: String,
    pub firmware: String,
    pub fw: [u8; 4],
}

/// d400_device::get_fw_details: 시리얼 = 48 부터 6 바이트 16진, ASIC = 64 부터, 펌웨어 = 12 부터 4 바이트 거꾸로
pub fn parse_gvd(g: &[u8]) -> Result<Gvd, String> {
    if g.len() < 70 {
        return Err(format!("GVD 응답이 짧음: {}", g.len()));
    }
    let hex = |o: usize| g[o..o + 6].iter().map(|b| format!("{b:02x}")).collect::<String>();
    let fw = [g[15], g[14], g[13], g[12]];
    Ok(Gvd { serial: hex(48), asic_serial: hex(64), firmware: format!("{}.{}.{}.{}", fw[0], fw[1], fw[2], fw[3]), fw })
}

pub fn fw_at_least(fw: [u8; 4], min: [u8; 4]) -> bool {
    fw >= min
}
