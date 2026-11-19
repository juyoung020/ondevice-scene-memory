//! 위치 프로그램 설정 · 열기 · 따로 도는 실행(`ondevice slam`). 같은 프로세스(`ondevice --slam "…"`)는 [`open`] 으로 백엔드와 입력을 받아
//! 처리 루프가 장면마다 돌린다.
use crate::backend::SlamBackend;
use crate::io::FrameSource;
use std::sync::atomic::{AtomicBool, Ordering};

/// 위치 프로그램 설정(명령줄을 `parse` 로 한 번 읽은 것)
#[derive(Clone)]
pub struct Args {
    /// 녹화 재생 폴더(--tum · --rsdrec)
    tum: String,
    realsense: bool,
    rate: f64,
    serial: String,
    w: u32,
    h: u32,
    /// --color-size: 컬러 크기(0 = --size 와 같음)
    color: (u32, u32),
    fps: u32,
    exposure_ms: f64,
    ae_cap_ms: f64,
    camera_profile: String,
    gain: i32,
    /// 적외선 레이저(점무늬) 켬(기본 끔)
    emitter: bool,
    out: String,
    poses: String,
    final_poses: String,
    send_every: i64,
    queue: usize,
    block: i32,
    reset_file: String,
    pub max_frames: i64,
    /// IMU: I2C 장치(실시간)
    imu: String,
    /// IMU-카메라 보정 yaml(빈 값 = 저장소 docs/imu_cam_calib.yaml 을 빌드에 넣은 것)
    imu_calib: String,
    /// 녹화 IMU(Kalibr imu0.csv)
    imu_csv: String,
    /// ORB-SLAM3 단어장(빈 값 = $ORB_SLAM3_DIR 또는 ~/ext/ORB_SLAM3 의 Vocabulary/ORBvoc.txt)
    voc: String,
}

impl Args {
    /// 받는 쪽을 기다리는지(--block | --drop, 기본: 최대 속도 재생이면 기다림)
    pub fn block_mode(&self) -> bool {
        let replay_max = !self.tum.is_empty() && self.rate == 0.0;
        if self.block < 0 { replay_max } else { self.block == 1 }
    }
}

fn usage() {
    common::info!(
        "ondevice slam (--tum|--rsdrec <녹화 폴더(적외선 좌우 포함)> [--rate R] --imu-csv <imu0.csv> | --realsense [--serial S] [--size 640x480] [--color-size 1280x720] [--fps 30] [--camera-profile librealsense|tracking] [--exposure ms | --ae-cap ms] [--gain 0..128] [--emitter on|off] --imu /dev/i2c-1)\n            [--imu-calib <yaml>] [--voc <ORBvoc.txt>] [--out -|<파일>] [--send-every N] [--queue N] [--block|--drop] [--poses <파일>] [--final-poses <파일>] [--max-frames N] [--reset-file <파일>]"
    );
}

pub fn parse(argv: &[String]) -> Result<Args, i32> {
    let mut a = Args {
        tum: String::new(),
        realsense: false,
        rate: 0.0,
        serial: String::new(),
        w: 640,
        h: 480,
        color: (0, 0),
        fps: 30,
        exposure_ms: 0.0,
        ae_cap_ms: 0.0, // 노출 상한(ms). 0 = 카메라 자동 노출
        camera_profile: "librealsense".into(),
        gain: -1,
        emitter: false, // 레이저 점무늬는 카메라와 함께 움직여 위치 추적을 흔듦(레이저 켠 녹화에서 추적 실패가 약 4 배)
        out: "-".into(),
        poses: String::new(),
        final_poses: String::new(),
        send_every: 1,
        queue: 4,
        block: -1, // -1: 입력에 따라(재생 = 기다림, 카메라 = 버림)
        reset_file: String::new(),
        max_frames: -1,
        imu: String::new(),
        imu_calib: String::new(),
        imu_csv: String::new(),
        voc: String::new(),
    };
    let mut i = 1;
    let num = |s: &str| -> Result<f64, i32> {
        s.parse::<f64>().map_err(|_| {
            common::info!("숫자가 아님: {s}");
            2
        })
    };
    let wh = |s: &str| s.split_once('x').and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)));
    while i < argv.len() {
        let k = argv[i].as_str();
        let mut val = || -> Result<String, i32> {
            if i + 1 >= argv.len() {
                usage();
                return Err(2);
            }
            i += 1;
            Ok(argv[i].clone())
        };
        match k {
            "--tum" | "--rsdrec" => a.tum = val()?,
            "--realsense" => a.realsense = true,
            "--rate" => a.rate = num(&val()?)?,
            "--serial" => a.serial = val()?,
            "--size" => {
                if let Some((w, h)) = wh(&val()?) {
                    (a.w, a.h) = (w, h);
                }
            }
            "--color-size" => a.color = wh(&val()?).unwrap_or((0, 0)),
            "--fps" => a.fps = num(&val()?)? as u32,
            "--exposure" => a.exposure_ms = num(&val()?)?,
            "--ae-cap" => a.ae_cap_ms = num(&val()?)?,
            "--camera-profile" => a.camera_profile = val()?,
            "--gain" => a.gain = num(&val()?)? as i32,
            "--emitter" => a.emitter = val()? == "on",
            "--out" => a.out = val()?,
            "--poses" => a.poses = val()?,
            "--final-poses" => a.final_poses = val()?,
            "--send-every" => a.send_every = (num(&val()?)? as i64).max(1),
            "--queue" => a.queue = (num(&val()?)? as i64).max(1) as usize,
            "--block" => a.block = 1,
            "--drop" => a.block = 0,
            "--reset-file" => a.reset_file = val()?,
            "--max-frames" => a.max_frames = num(&val()?)? as i64,
            "--imu" => a.imu = val()?,
            "--imu-calib" => a.imu_calib = val()?,
            "--imu-csv" => a.imu_csv = val()?,
            "--voc" => a.voc = val()?,
            // 프로세스 설정(ondevice 가 처음에 읽음): 실시간 배치·메모리 묶기·조용히
            "--rt" => {
                let _ = val()?;
            }
            "--mlock" | "--quiet" => {}
            "-h" | "--help" => {
                usage();
                return Err(0);
            }
            _ => {
                common::info!("모르는 인자: {k}");
                usage();
                return Err(2);
            }
        }
        i += 1;
    }
    if a.tum.is_empty() == !a.realsense {
        usage();
        return Err(2);
    }
    Ok(a)
}

/// D435 입력(rsdriver: V4L2 직접, 리눅스 빌드)
fn open_camera(a: &Args) -> Result<Box<dyn FrameSource>, String> {
    #[cfg(feature = "rsdriver")]
    {
        let o = crate::io::rsdriver::RsOpen { w: a.w, h: a.h, color: a.color, fps: a.fps, serial: &a.serial, exposure_ms: a.exposure_ms, gain: a.gain, ae_cap_ms: a.ae_cap_ms, profile: &a.camera_profile, emitter: a.emitter };
        Ok(Box::new(crate::io::RsDriverSource::new(o)?))
    }
    #[cfg(not(feature = "rsdriver"))]
    {
        let _ = a;
        Err("D435 입력은 리눅스 빌드(rsdriver 기능)에만 있음".into())
    }
}

/// 단어장 경로: --voc, 없으면 $ORB_SLAM3_DIR(또는 ~/ext/ORB_SLAM3)/Vocabulary/ORBvoc.txt
fn voc_path(a: &Args) -> String {
    if !a.voc.is_empty() {
        return a.voc.clone();
    }
    let dir = std::env::var("ORB_SLAM3_DIR").unwrap_or_else(|_| format!("{}/ext/ORB_SLAM3", std::env::var("HOME").unwrap_or_default()));
    format!("{dir}/Vocabulary/ORBvoc.txt")
}

/// 위치 백엔드와 장면 입력을 연다: 입력 · IMU(실시간 읽기 스레드 또는 녹화 재생) · 보정(적외선 + Kalibr) → ORB-SLAM3 설정 → 백엔드
pub fn open(a: Args) -> Result<(Box<dyn SlamBackend>, Box<dyn FrameSource>), String> {
    let mut src: Box<dyn FrameSource> = if a.realsense { open_camera(&a)? } else { Box::new(crate::io::TumSource::new(&a.tum, a.rate)?) };
    common::info!("입력: {}", src.describe());
    let calib_text = if a.imu_calib.is_empty() { crate::imu::DEFAULT_CALIB.to_string() } else { std::fs::read_to_string(&a.imu_calib).map_err(|e| format!("{}: {e}", a.imu_calib))? };
    let calib = crate::imu::Calib::parse(&calib_text)?;
    // IMU 를 주면(실시간 --imu, 재생 --imu-csv) 스테레오-관성, 아니면 스테레오
    let mut ring = None;
    let mut replay = None;
    if a.realsense && !a.imu.is_empty() {
        let r = crate::imu::Ring::new();
        crate::imu::spawn_reader(&a.imu, r.clone())?;
        ring = Some(r);
    } else if !a.realsense && !a.imu_csv.is_empty() {
        let (r, csv) = (crate::imu::Ring::new(), crate::imu::CsvReplay::load(&a.imu_csv, calib.shift)?);
        csv.count_into(&r);
        common::info!("IMU 재생: {} 표본, {:.1} 초", csv.len(), csv.span().1 - csv.span().0);
        match src.caller_paces() {
            Some((rate, clock)) => csv.spawn_timed(r.clone(), clock, rate),
            None => replay = Some(csv),
        }
        ring = Some(r);
    }
    let st = src.stereo().ok_or("입력에 적외선 좌우가 없음(D435 --realsense 또는 rsd_rec --ir 녹화)")?;
    // 적외선 왼쪽 → IMU = (IMU → 컬러)⁻¹ · (적외선 왼쪽 → 컬러)
    let t_b_c1 = crate::backend::mul(&crate::backend::inv(&calib.t_cam_imu), &st.color_from_ir);
    let settings = crate::orb3::settings_yaml(&st, &t_b_c1, a.fps as f64);
    let block = a.block_mode();
    let settings_path = std::env::temp_dir().join(format!("orb3_{}.yaml", std::process::id()));
    let o = OpenParts { voc: voc_path(&a), settings, settings_path, ring, replay, shift: calib.shift, color_from_ir: st.color_from_ir, cap: a.queue, block, send_every: a.send_every, reset_file: a.reset_file.clone(), poses: a.poses.clone(), final_poses: a.final_poses.clone() };
    Ok((crate::backend::make(o)?, src))
}

/// 백엔드를 만드는 데 필요한 것(설정 · IMU · 보정 · 출력)
pub struct OpenParts {
    pub voc: String,
    pub settings: String,
    pub settings_path: std::path::PathBuf,
    /// IMU 표본(있으면 스테레오-관성)
    pub ring: Option<std::sync::Arc<crate::imu::Ring>>,
    pub replay: Option<crate::imu::CsvReplay>,
    pub shift: f64,
    pub color_from_ir: crate::backend::T34,
    pub cap: usize,
    pub block: bool,
    pub send_every: i64,
    pub reset_file: String,
    pub poses: String,
    pub final_poses: String,
}

static STOP: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
mod sys {
    unsafe extern "C" {
        fn dup(fd: i32) -> i32;
        fn dup2(a: i32, b: i32) -> i32;
        fn signal(sig: i32, h: usize) -> usize;
    }
    extern "C" fn on_signal(_: i32) {
        super::STOP.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    /// 표준출력을 떼어 내 스트림용 파일로 돌려주고, 표준출력은 표준오류로 돌린다(ORB-SLAM3 의 출력이 장면 바이트에 섞이지 않게)
    pub fn take_stdout() -> std::fs::File {
        use std::os::fd::FromRawFd;
        let _ = std::io::Write::flush(&mut std::io::stdout());
        // SAFETY: 표준출력 기술자를 복제하고 그 자리를 표준오류로
        unsafe {
            let fd = dup(1);
            dup2(2, 1);
            std::fs::File::from_raw_fd(fd)
        }
    }
    pub fn on_stop_signals() {
        // SAFETY: 신호 처리기는 원자 값 하나만 씀
        unsafe {
            signal(2, on_signal as extern "C" fn(i32) as usize); // SIGINT
            signal(15, on_signal as extern "C" fn(i32) as usize); // SIGTERM
        }
    }
}

/// 명령줄 그대로 실행(`ondevice slam …`): 장면 스트림을 파일 · 표준출력으로 쓴다
pub fn run(argv: &[String]) -> Result<i32, String> {
    let a = match parse(argv) {
        Ok(a) => a,
        Err(c) => return Ok(c),
    };
    #[cfg(unix)]
    sys::on_stop_signals();
    let fd = if a.out == "-" {
        #[cfg(unix)]
        {
            Some(sys::take_stdout())
        }
        #[cfg(not(unix))]
        {
            None
        }
    } else {
        None
    };
    let (block, max, queue) = (a.block_mode(), a.max_frames, a.queue);
    let mut w = crate::stream::Writer::new(&a.out, fd, queue, block)?;
    let (mut r, mut src) = open(a)?;
    while !STOP.load(Ordering::Relaxed) && (max < 0 || r.frames() < max) {
        let Some(f) = src.next()? else { break };
        let ms = src.last_read_ms();
        if !r.step(f, ms) {
            break;
        }
        while let Some(o) = r.pop() {
            if !w.push(o) {
                return Err("스트림 받는 쪽이 닫힘".into());
            }
        }
    }
    w.close();
    common::info!("보냄 {}, 밀려 버림 {}", w.written(), w.dropped());
    r.finish()
}
