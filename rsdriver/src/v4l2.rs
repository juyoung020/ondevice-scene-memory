//! 리눅스 V4L2·uvcvideo ioctl (64 비트 리눅스: x86_64·aarch64 같은 배치). libc 만 씀.
use std::ffi::CString;
use std::io;

pub const BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
pub const BUF_TYPE_META_CAPTURE: u32 = 13;
pub const MEMORY_MMAP: u32 = 1;
pub const CAP_VIDEO_CAPTURE: u32 = 0x0000_0001;
pub const CAP_META_CAPTURE: u32 = 0x0080_0000;

pub const fn fourcc(s: &[u8; 4]) -> u32 {
    (s[0] as u32) | ((s[1] as u32) << 8) | ((s[2] as u32) << 16) | ((s[3] as u32) << 24)
}
pub const FMT_Z16: u32 = fourcc(b"Z16 ");
pub const FMT_YUYV: u32 = fourcc(b"YUYV");
pub const FMT_META_UVC: u32 = fourcc(b"UVCH");

// 제어 번호
pub const CID_GAIN: u32 = 0x0098_0913;
pub const CID_EXPOSURE_AUTO: u32 = 0x009a_0901;
pub const CID_EXPOSURE_ABSOLUTE: u32 = 0x009a_0902;
pub const CID_EXPOSURE_AUTO_PRIORITY: u32 = 0x009a_0903;
pub const EXPOSURE_MANUAL: i32 = 1;
pub const EXPOSURE_APERTURE_PRIORITY: i32 = 3;

pub const UVC_SET_CUR: u8 = 0x01;
pub const UVC_GET_CUR: u8 = 0x81;

const fn iowr(nr: u32, size: u32) -> libc::c_ulong {
    ((3u32 << 30) | (size << 16) | ((b'V' as u32) << 8) | nr) as libc::c_ulong
}
const fn iow(nr: u32, size: u32) -> libc::c_ulong {
    ((1u32 << 30) | (size << 16) | ((b'V' as u32) << 8) | nr) as libc::c_ulong
}
const fn ior(nr: u32, size: u32) -> libc::c_ulong {
    ((2u32 << 30) | (size << 16) | ((b'V' as u32) << 8) | nr) as libc::c_ulong
}
pub const VIDIOC_QUERYCAP: libc::c_ulong = ior(0, 104);
pub const VIDIOC_S_FMT: libc::c_ulong = iowr(5, 208);
pub const VIDIOC_REQBUFS: libc::c_ulong = iowr(8, 20);
pub const VIDIOC_QUERYBUF: libc::c_ulong = iowr(9, 88);
pub const VIDIOC_QBUF: libc::c_ulong = iowr(15, 88);
pub const VIDIOC_DQBUF: libc::c_ulong = iowr(17, 88);
pub const VIDIOC_STREAMON: libc::c_ulong = iow(18, 4);
pub const VIDIOC_STREAMOFF: libc::c_ulong = iow(19, 4);
pub const VIDIOC_S_PARM: libc::c_ulong = iowr(22, 204);
pub const VIDIOC_G_CTRL: libc::c_ulong = iowr(27, 8);
pub const VIDIOC_S_CTRL: libc::c_ulong = iowr(28, 8);
/// _IOWR('u', 0x21, struct uvc_xu_control_query) — 16 바이트
pub const UVCIOC_CTRL_QUERY: libc::c_ulong = ((3u32 << 30) | (16 << 16) | ((b'u' as u32) << 8) | 0x21) as libc::c_ulong;

#[repr(C)]
pub struct Capability {
    pub driver: [u8; 16],
    pub card: [u8; 32],
    pub bus_info: [u8; 32],
    pub version: u32,
    pub capabilities: u32,
    pub device_caps: u32,
    pub reserved: [u32; 3],
}

/// struct v4l2_format: type + 200 바이트 공용체(8 바이트 정렬)
#[repr(C, align(8))]
pub struct Format {
    pub typ: u32,
    pub _pad: u32,
    pub raw: [u8; 200],
}

#[repr(C)]
pub struct StreamParm {
    pub typ: u32,
    pub raw: [u8; 200],
}

#[repr(C)]
#[derive(Default)]
pub struct RequestBuffers {
    pub count: u32,
    pub typ: u32,
    pub memory: u32,
    pub capabilities: u32,
    pub flags: u8,
    pub reserved: [u8; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Buffer {
    pub index: u32,
    pub typ: u32,
    pub bytesused: u32,
    pub flags: u32,
    pub field: u32,
    pub _pad: u32,
    pub timestamp: libc::timeval,
    pub timecode: [u32; 4],
    pub sequence: u32,
    pub memory: u32,
    pub offset: u64,
    pub length: u32,
    pub reserved2: u32,
    pub request_fd: u32,
    pub _pad2: u32,
}

#[repr(C)]
pub struct Control {
    pub id: u32,
    pub value: i32,
}

#[repr(C)]
pub struct XuQuery {
    pub unit: u8,
    pub selector: u8,
    pub query: u8,
    pub size: u16,
    pub data: *mut u8,
}

const _: () = assert!(std::mem::size_of::<Capability>() == 104);
const _: () = assert!(std::mem::size_of::<Format>() == 208);
const _: () = assert!(std::mem::size_of::<StreamParm>() == 204);
const _: () = assert!(std::mem::size_of::<RequestBuffers>() == 20);
const _: () = assert!(std::mem::size_of::<Buffer>() == 88);
const _: () = assert!(std::mem::size_of::<XuQuery>() == 16);

impl Buffer {
    pub fn new(typ: u32, index: u32) -> Buffer {
        let mut b: Buffer = unsafe { std::mem::zeroed() };
        b.typ = typ;
        b.memory = MEMORY_MMAP;
        b.index = index;
        b
    }
}

/// EINTR 이면 다시 부름
pub fn xioctl<T>(fd: i32, req: libc::c_ulong, arg: *mut T) -> io::Result<()> {
    loop {
        let r = unsafe { libc::ioctl(fd, req as _, arg) };
        if r >= 0 {
            return Ok(());
        }
        let e = io::Error::last_os_error();
        if e.raw_os_error() != Some(libc::EINTR) {
            return Err(e);
        }
    }
}

/// 열린 V4L2 노드
pub struct Node {
    pub fd: i32,
    pub path: String,
}

impl Node {
    pub fn open(path: &str) -> io::Result<Node> {
        let c = CString::new(path).unwrap();
        let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDWR | libc::O_NONBLOCK | libc::O_CLOEXEC) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Node { fd, path: path.to_string() })
    }
    pub fn device_caps(&self) -> io::Result<u32> {
        let mut c: Capability = unsafe { std::mem::zeroed() };
        xioctl(self.fd, VIDIOC_QUERYCAP, &mut c)?;
        Ok(c.device_caps)
    }
    /// 영상 형식(가로·세로·fourcc). 돌려줌: (bytesperline, sizeimage)
    pub fn set_video_format(&self, w: u32, h: u32, fourcc: u32) -> io::Result<(u32, u32)> {
        let mut f = Format { typ: BUF_TYPE_VIDEO_CAPTURE, _pad: 0, raw: [0; 200] };
        // v4l2_pix_format: width, height, pixelformat, field, bytesperline, sizeimage, ...
        f.raw[0..4].copy_from_slice(&w.to_le_bytes());
        f.raw[4..8].copy_from_slice(&h.to_le_bytes());
        f.raw[8..12].copy_from_slice(&fourcc.to_le_bytes());
        f.raw[12..16].copy_from_slice(&1u32.to_le_bytes()); // V4L2_FIELD_NONE
        xioctl(self.fd, VIDIOC_S_FMT, &mut f)?;
        let g = |o: usize| u32::from_le_bytes(f.raw[o..o + 4].try_into().unwrap());
        if g(0) != w || g(4) != h || g(8) != fourcc {
            return Err(io::Error::other(format!("{}: 형식 {}x{} {:08x} 을 받지 않음({}x{} {:08x})", self.path, w, h, fourcc, g(0), g(4), g(8))));
        }
        Ok((g(16), g(20)))
    }
    pub fn set_meta_format(&self) -> io::Result<u32> {
        let mut f = Format { typ: BUF_TYPE_META_CAPTURE, _pad: 0, raw: [0; 200] };
        f.raw[0..4].copy_from_slice(&FMT_META_UVC.to_le_bytes());
        xioctl(self.fd, VIDIOC_S_FMT, &mut f)?;
        Ok(u32::from_le_bytes(f.raw[4..8].try_into().unwrap()))
    }
    /// timeperframe = 1/fps
    pub fn set_fps(&self, fps: u32) -> io::Result<()> {
        let mut p = StreamParm { typ: BUF_TYPE_VIDEO_CAPTURE, raw: [0; 200] };
        // v4l2_captureparm: capability, capturemode, timeperframe{numerator, denominator}
        p.raw[8..12].copy_from_slice(&1u32.to_le_bytes());
        p.raw[12..16].copy_from_slice(&fps.to_le_bytes());
        xioctl(self.fd, VIDIOC_S_PARM, &mut p)
    }
    pub fn set_ctrl(&self, id: u32, value: i32) -> io::Result<()> {
        let mut c = Control { id, value };
        xioctl(self.fd, VIDIOC_S_CTRL, &mut c)
    }
    pub fn get_ctrl(&self, id: u32) -> io::Result<i32> {
        let mut c = Control { id, value: 0 };
        xioctl(self.fd, VIDIOC_G_CTRL, &mut c)?;
        Ok(c.value)
    }
    /// UVC 확장 단위 제어(UVCIOC_CTRL_QUERY)
    pub fn xu(&self, unit: u8, selector: u8, query: u8, data: &mut [u8]) -> io::Result<()> {
        let mut q = XuQuery { unit, selector, query, size: data.len() as u16, data: data.as_mut_ptr() };
        xioctl(self.fd, UVCIOC_CTRL_QUERY, &mut q)
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd) };
    }
}

/// mmap 한 V4L2 버퍼 묶음과 흐름
pub struct Stream {
    pub node: Node,
    pub typ: u32,
    pub maps: Vec<(*mut u8, usize)>,
    pub on: bool,
}

unsafe impl Send for Stream {}

impl Stream {
    pub fn new(node: Node, typ: u32, count: u32) -> io::Result<Stream> {
        let mut rb = RequestBuffers { count, typ, memory: MEMORY_MMAP, ..Default::default() };
        xioctl(node.fd, VIDIOC_REQBUFS, &mut rb)?;
        let mut maps = Vec::new();
        for i in 0..rb.count {
            let mut b = Buffer::new(typ, i);
            xioctl(node.fd, VIDIOC_QUERYBUF, &mut b)?;
            let p = unsafe { libc::mmap(std::ptr::null_mut(), b.length as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, node.fd, (b.offset as u32) as libc::off_t) };
            if p == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            maps.push((p as *mut u8, b.length as usize));
        }
        let s = Stream { node, typ, maps, on: false };
        for i in 0..s.maps.len() as u32 {
            s.queue(i)?;
        }
        Ok(s)
    }
    pub fn queue(&self, index: u32) -> io::Result<()> {
        let mut b = Buffer::new(self.typ, index);
        xioctl(self.node.fd, VIDIOC_QBUF, &mut b)
    }
    pub fn start(&mut self) -> io::Result<()> {
        let mut t = self.typ as i32;
        xioctl(self.node.fd, VIDIOC_STREAMON, &mut t)?;
        self.on = true;
        Ok(())
    }
    /// 준비된 버퍼 하나(없으면 None, EAGAIN)
    pub fn dequeue(&self) -> io::Result<Option<Buffer>> {
        let mut b = Buffer::new(self.typ, 0);
        match xioctl(self.node.fd, VIDIOC_DQBUF, &mut b) {
            Ok(()) => Ok(Some(b)),
            Err(e) if e.raw_os_error() == Some(libc::EAGAIN) => Ok(None),
            Err(e) => Err(e),
        }
    }
    /// 버퍼 내용(mmap, 복사 없음)
    pub fn data(&self, b: &Buffer) -> &[u8] {
        let (p, len) = self.maps[b.index as usize];
        unsafe { std::slice::from_raw_parts(p, (b.bytesused as usize).min(len)) }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.on {
            let mut t = self.typ as i32;
            let _ = xioctl(self.node.fd, VIDIOC_STREAMOFF, &mut t);
        }
        for &(p, len) in &self.maps {
            unsafe { libc::munmap(p as *mut libc::c_void, len) };
        }
        let mut rb = RequestBuffers { count: 0, typ: self.typ, memory: MEMORY_MMAP, ..Default::default() };
        let _ = xioctl(self.node.fd, VIDIOC_REQBUFS, &mut rb);
    }
}

/// 여러 fd 중 읽을 것이 생길 때까지 기다림(ms). 돌려줌: 준비된 fd 표시
pub fn poll(fds: &[i32], timeout_ms: i32) -> io::Result<Vec<bool>> {
    let mut p: Vec<libc::pollfd> = fds.iter().map(|&fd| libc::pollfd { fd, events: libc::POLLIN, revents: 0 }).collect();
    let r = unsafe { libc::poll(p.as_mut_ptr(), p.len() as _, timeout_ms) };
    if r < 0 {
        let e = io::Error::last_os_error();
        if e.raw_os_error() == Some(libc::EINTR) {
            return Ok(vec![false; fds.len()]);
        }
        return Err(e);
    }
    Ok(p.iter().map(|x| x.revents & libc::POLLIN != 0).collect())
}
