//! HW 모니터 명령(librealsense `src/hw-monitor.cpp`·`platform/uvc-option.cpp` 의 command_transfer_over_xu).
//! 깊이 XU(unit 3)의 제어 1(DS5_HWMONITOR)에 1024 바이트를 SET_CUR 로 쓰고 GET_CUR 로 1024 바이트를 읽는다.
//! 응답: 끝 4 바이트(1020..1024) = 자료 길이, 앞 4 바이트 = 보낸 명령 번호(오류면 음수 코드), 그 뒤가 자료.

pub const BUFFER_SIZE: usize = 1024;
pub const MAGIC: u16 = 0xCDAB;

pub const GVD: u32 = 0x10;
pub const GETINTCAL: u32 = 0x15;
pub const GET_ADV: u32 = 0x2C;
pub const UAMG: u32 = 0x30;
pub const RECPARAMSGET: u32 = 0x7E;

pub const COEFFICIENTS_TABLE_ID: u32 = 25;
pub const RGB_CALIBRATION_ID: u32 = 32;
pub const ET_DEPTH_TABLE_CONTROL: u32 = 9;

/// fill_usb_buffer + 1024 바이트로 0 채움
pub fn build(op: u32, p: [u32; 4], data: &[u8]) -> Vec<u8> {
    let mut b = vec![0u8; BUFFER_SIZE];
    let mut i = 2;
    b[i..i + 2].copy_from_slice(&MAGIC.to_le_bytes());
    i += 2;
    for v in [op, p[0], p[1], p[2], p[3]] {
        b[i..i + 4].copy_from_slice(&v.to_le_bytes());
        i += 4;
    }
    let n = data.len().min(BUFFER_SIZE - i);
    b[i..i + n].copy_from_slice(&data[..n]);
    i += n;
    b[0..2].copy_from_slice(&((i - 4) as u16).to_le_bytes());
    b
}

/// GET_CUR 로 읽은 1024 바이트에서 자료를 꺼냄
pub fn parse(op: u32, resp: &[u8]) -> Result<Vec<u8>, String> {
    if resp.len() < BUFFER_SIZE {
        return Err(format!("HWM 응답이 짧음: {}", resp.len()));
    }
    let size = u32::from_le_bytes(resp[1020..1024].try_into().unwrap()) as usize + 4;
    if !(4..=BUFFER_SIZE).contains(&size) {
        return Err(format!("HWM 응답 길이 이상: {size}"));
    }
    let got = u32::from_le_bytes(resp[0..4].try_into().unwrap());
    if got != op {
        return Err(format!("HWM 명령 0x{op:02x} 오류 코드 {}", got as i32));
    }
    Ok(resp[4..size].to_vec())
}
