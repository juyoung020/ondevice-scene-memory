//! Frame(ORB-SLAM3 src/Frame.cc) · MapPoint 일부 옮김: RGB-D 장면 만들기, 격자, 위치 행렬, 시야 판정, 영역 안 특징점, 깊이로 3D.
//! 왜곡 0(우리 설정)이라 mvKeysUn = mvKeys, 경계는 영상 크기 그대로.
use super::types::*;
use std::sync::Arc;

impl FrameCtx {
    /// 원본 첫 장면에서 한 번 하는 계산(ComputeImageBounds · 격자 칸 역수, 왜곡 0)
    pub fn new(cam: Camera, pyr: Pyramid, w: i32, h: i32) -> FrameCtx {
        let (min_x, max_x, min_y, max_y) = (0.0f32, w as f32, 0.0f32, h as f32);
        FrameCtx { cam, pyr, min_x, max_x, min_y, max_y, grid_inv_w: GRID_COLS as f32 / (max_x - min_x), grid_inv_h: GRID_ROWS as f32 / (max_y - min_y) }
    }
}

impl Camera {
    /// Pinhole 값(fx · fy · cx · cy, bf = 기준선 × fx, th_depth = 설정 ThDepth(기준선 배수))
    pub fn pinhole(fx: f32, fy: f32, cx: f32, cy: f32, bf: f32, th_depth_factor: f32, w: i32, h: i32) -> Camera {
        // Tracking: mThDepth = mbf * ThDepth / fx, Frame: mb = mbf / fx
        Camera { fx, fy, cx, cy, invfx: 1.0 / fx, invfy: 1.0 / fy, bf, b: bf / fx, th_depth: bf * th_depth_factor / fx, width: w, height: h }
    }
    /// Pinhole::project: fx·x/z + cx
    #[inline]
    pub fn project(&self, p: &V3) -> [f32; 2] {
        [self.fx * p[0] / p[2] + self.cx, self.fy * p[1] / p[2] + self.cy]
    }
}

/// 깊이 영상(16 비트) → 미터(float): cv::Mat::convertTo(CV_32F, 1/DepthMapFactor)
pub fn depth_to_float(raw: &[u16], depth_map_factor: f32) -> Vec<f32> {
    let s = (1.0f64 / depth_map_factor as f64) as f32;
    raw.iter().map(|&d| d as f32 * s).collect()
}

impl Frame {
    /// RGB-D 장면(Frame 생성자): 이미 뽑은 특징점 · 기술자 + 깊이(미터, w 칸 줄)
    pub fn new_rgbd(id: u64, timestamp: f64, keys: Vec<KeyPoint>, desc: Vec<Desc>, depth: &[f32], w: usize, ctx: Arc<FrameCtx>) -> Frame {
        let n = keys.len();
        let mut f = Frame { id, timestamp, ctx, ..Default::default() };
        f.u_right = vec![-1.0; n];
        f.depth = vec![-1.0; n];
        let bf = f.ctx.cam.bf;
        for (i, k) in keys.iter().enumerate() {
            // imDepth.at<float>(v, u): float → int 잘라냄
            let d = depth[k.y as i32 as usize * w + k.x as i32 as usize];
            if d > 0.0 {
                f.depth[i] = d;
                f.u_right[i] = k.x - bf / d;
            }
        }
        f.map_points = vec![None; n];
        f.outlier = vec![false; n];
        f.keys = keys;
        f.desc = desc;
        f.assign_grid();
        f
    }

    fn assign_grid(&mut self) {
        self.grid = vec![Vec::new(); GRID_COLS * GRID_ROWS];
        for i in 0..self.keys.len() {
            if let Some((x, y)) = self.pos_in_grid(&self.keys[i]) {
                self.grid[x * GRID_ROWS + y].push(i as u32);
            }
        }
    }

    /// PosInGrid: round((x - minX) · 역수)(std::round = 0.5 는 0 에서 멀어지게)
    pub fn pos_in_grid(&self, k: &KeyPoint) -> Option<(usize, usize)> {
        let c = &self.ctx;
        let x = ((k.x - c.min_x) * c.grid_inv_w).round() as i32;
        let y = ((k.y - c.min_y) * c.grid_inv_h).round() as i32;
        if x < 0 || x >= GRID_COLS as i32 || y < 0 || y >= GRID_ROWS as i32 {
            return None;
        }
        Some((x as usize, y as usize))
    }

    /// SetPose + UpdatePoseMatrices
    pub fn set_pose(&mut self, tcw: Se3) {
        let twc = tcw.inverse();
        self.rwc = twc.rot();
        self.ow = twc.t;
        self.rcw = tcw.rot();
        self.tcw_t = tcw.t;
        self.tcw = Some(tcw);
    }

    /// GetFeaturesInArea(왼쪽 영상). max_level < 0 = 위 제한 없음
    pub fn features_in_area(&self, x: f32, y: f32, r: f32, min_level: i32, max_level: i32) -> Vec<u32> {
        let mut out = Vec::new();
        let c = &self.ctx;
        let nminx = 0.max(((x - c.min_x - r) * c.grid_inv_w).floor() as i32);
        if nminx >= GRID_COLS as i32 {
            return out;
        }
        let nmaxx = (GRID_COLS as i32 - 1).min(((x - c.min_x + r) * c.grid_inv_w).ceil() as i32);
        if nmaxx < 0 {
            return out;
        }
        let nminy = 0.max(((y - c.min_y - r) * c.grid_inv_h).floor() as i32);
        if nminy >= GRID_ROWS as i32 {
            return out;
        }
        let nmaxy = (GRID_ROWS as i32 - 1).min(((y - c.min_y + r) * c.grid_inv_h).ceil() as i32);
        if nmaxy < 0 {
            return out;
        }
        let check = min_level > 0 || max_level >= 0;
        for ix in nminx..=nmaxx {
            for iy in nminy..=nmaxy {
                for &j in self.grid[ix as usize * GRID_ROWS + iy as usize].iter() {
                    let k = &self.keys[j as usize];
                    if check {
                        if k.octave < min_level {
                            continue;
                        }
                        if max_level >= 0 && k.octave > max_level {
                            continue;
                        }
                    }
                    if (k.x - x).abs() < r && (k.y - y).abs() < r {
                        out.push(j);
                    }
                }
            }
        }
        out
    }

    /// UnprojectStereo: 깊이 있는 점의 세계 좌표
    pub fn unproject_stereo(&self, i: usize) -> Option<V3> {
        let z = self.depth[i];
        if z > 0.0 {
            let c = &self.ctx.cam;
            let (u, v) = (self.keys[i].x, self.keys[i].y);
            let x = (u - c.cx) * z * c.invfx;
            let y = (v - c.cy) * z * c.invfy;
            Some(add(&mat_vec(&self.rwc, &[x, y, z]), &self.ow))
        } else {
            None
        }
    }

    /// isInFrustum(왼쪽만): 지도 점의 추적용 칸을 채움
    pub fn is_in_frustum(&self, mp: &mut MapPoint, view_cos_limit: f32) -> bool {
        mp.track_in_view = false;
        mp.track_proj_x = -1.0;
        mp.track_proj_y = -1.0;
        let p = mp.pos;
        let pc = add(&mat_vec(&self.rcw, &p), &self.tcw_t);
        let pc_dist = norm(&pc);
        let pcz = pc[2];
        let invz = 1.0f32 / pcz;
        if pcz < 0.0 {
            return false;
        }
        let uv = self.ctx.cam.project(&pc);
        let c = &self.ctx;
        if uv[0] < c.min_x || uv[0] > c.max_x || uv[1] < c.min_y || uv[1] > c.max_y {
            return false;
        }
        mp.track_proj_x = uv[0];
        mp.track_proj_y = uv[1];
        let max_d = mp.max_distance_invariance();
        let min_d = mp.min_distance_invariance();
        let po = sub(&p, &self.ow);
        let dist = norm(&po);
        if dist < min_d || dist > max_d {
            return false;
        }
        let view_cos = dot(&po, &mp.normal) / dist;
        if view_cos < view_cos_limit {
            return false;
        }
        let level = mp.predict_scale(dist, &c.pyr);
        mp.track_in_view = true;
        mp.track_proj_x = uv[0];
        mp.track_proj_xr = uv[0] - c.cam.bf * invz;
        mp.track_depth = pc_dist;
        mp.track_proj_y = uv[1];
        mp.track_scale_level = level;
        mp.track_view_cos = view_cos;
        true
    }
}

impl MapPoint {
    /// MapPoint(Pos, Map, Frame, idx): 장면에서 만든 점(UpdateLastFrame 의 임시 점 · 키프레임 점)
    pub fn from_frame(id: MpId, map: MapId, pos: V3, f: &Frame, idx: usize) -> MapPoint {
        let ow = f.ow;
        let nv = sub(&pos, &ow);
        let nn = norm(&nv);
        let normal = [nv[0] / nn, nv[1] / nn, nv[2] / nn];
        let dist = norm(&sub(&pos, &ow));
        let level = f.keys[idx].octave as usize;
        let pyr = &f.ctx.pyr;
        let max_distance = dist * pyr.scale[level];
        let min_distance = max_distance / pyr.scale[pyr.nlevels as usize - 1];
        MapPoint { id, map, pos, normal, desc: f.desc[idx], first_frame: f.id, visible: 1, found: 1, min_dist: min_distance, max_dist: max_distance, ..Default::default() }
    }
    pub fn min_distance_invariance(&self) -> f32 {
        0.8f32 * self.min_dist
    }
    pub fn max_distance_invariance(&self) -> f32 {
        1.2f32 * self.max_dist
    }
    /// PredictScale: ceil(logf(max/거리) / logf(배율)), 0..층 수-1
    pub fn predict_scale(&self, current_dist: f32, pyr: &Pyramid) -> i32 {
        let ratio = self.max_dist / current_dist;
        let n = (ratio.ln() / pyr.log_scale_factor).ceil() as i32;
        n.clamp(0, pyr.nlevels - 1)
    }
}

impl KeyFrame {
    /// KeyFrame(Frame&, Map*, KFDB*): 장면 내용을 그대로 옮겨 받음(위치 · 점 · 격자 · BoW)
    pub fn from_frame(id: KfId, map: MapId, f: &Frame) -> KeyFrame {
        let mut k = KeyFrame {
            id,
            frame_id: f.id,
            map,
            timestamp: f.timestamp,
            keys: f.keys.clone(),
            desc: f.desc.clone(),
            u_right: f.u_right.clone(),
            depth: f.depth.clone(),
            map_points: f.map_points.clone(),
            grid: f.grid.clone(),
            ctx: f.ctx.clone(),
            bow: f.bow.clone(),
            feat: f.feat.clone(),
            ..Default::default()
        };
        k.set_pose(f.tcw.expect("키프레임 위치"));
        k
    }
    /// SetPose: mTcw, mTwc = mTcw.inverse()
    pub fn set_pose(&mut self, tcw: Se3) {
        self.tcw = tcw;
        self.twc = tcw.inverse();
    }
    pub fn camera_center(&self) -> V3 {
        self.twc.t
    }
}

impl Camera {
    pub fn k(&self) -> M3 {
        [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
    }
    /// Pinhole::epipolarConstrain: 2 영상 점이 1 의 극선에서 떨어진 거리² < 3.84 · unc
    pub fn epipolar_constrain(&self, cam2: &Camera, kp1: &KeyPoint, kp2: &KeyPoint, r12: &M3, t12: &V3, unc: f32) -> bool {
        let f12 = mat_mul(&mat_mul(&mat_mul(&mat_inv(&transpose(&self.k())), &hat(t12)), r12), &mat_inv(&cam2.k()));
        let a = kp1.x * f12[0][0] + kp1.y * f12[1][0] + f12[2][0];
        let b = kp1.x * f12[0][1] + kp1.y * f12[1][1] + f12[2][1];
        let c = kp1.x * f12[0][2] + kp1.y * f12[1][2] + f12[2][2];
        let num = a * kp2.x + b * kp2.y + c;
        let den = a * a + b * b;
        if den == 0.0 {
            return false;
        }
        let dsqr = num * num / den;
        (dsqr as f64) < 3.84 * unc as f64
    }
}
