//! ORBmatcher(ORB-SLAM3 src/ORBmatcher.cc) 옮김. 단안 · 입체 왼쪽 경로만(우리는 RGB-D: Nleft = -1, mpCamera2 없음).
//! 지도 점은 id 로 받고 Atlas 의 표(BTreeMap)에서 찾는다.
use super::types::*;
use std::collections::BTreeMap;

pub const TH_HIGH: i32 = 100;
pub const TH_LOW: i32 = 50;
pub const HISTO_LENGTH: usize = 30;

pub struct Matcher {
    pub nn_ratio: f32,
    pub check_ori: bool,
}

/// ComputeThreeMaxima: 회전 막대그래프에서 가장 많은 세 칸(없으면 -1)
pub fn three_maxima(h: &[Vec<u32>]) -> (i32, i32, i32) {
    let (mut m1, mut m2, mut m3) = (0usize, 0usize, 0usize);
    let (mut i1, mut i2, mut i3) = (-1i32, -1i32, -1i32);
    for (i, v) in h.iter().enumerate() {
        let s = v.len();
        if s > m1 {
            m3 = m2;
            m2 = m1;
            m1 = s;
            i3 = i2;
            i2 = i1;
            i1 = i as i32;
        } else if s > m2 {
            m3 = m2;
            m2 = s;
            i3 = i2;
            i2 = i as i32;
        } else if s > m3 {
            m3 = s;
            i3 = i as i32;
        }
    }
    if (m2 as i32 as f32) < 0.1f32 * m1 as i32 as f32 {
        i2 = -1;
        i3 = -1;
    } else if (m3 as i32 as f32) < 0.1f32 * m1 as i32 as f32 {
        i3 = -1;
    }
    (i1, i2, i3)
}

/// RadiusByViewingCos
pub fn radius_by_viewing_cos(view_cos: f32) -> f32 {
    if view_cos as f64 > 0.998 { 2.5 } else { 4.0 }
}

/// 회전 막대 칸: round((각1 − 각2 (+360)) / 30)
#[inline]
fn rot_bin(a1: f32, a2: f32) -> usize {
    let factor = 1.0f32 / HISTO_LENGTH as f32;
    let mut rot = a1 - a2;
    if (rot as f64) < 0.0 {
        rot += 360.0f32;
    }
    let mut bin = (rot * factor).round() as i32;
    if bin == HISTO_LENGTH as i32 {
        bin = 0;
    }
    bin as usize
}

impl Matcher {
    pub fn new(nn_ratio: f32, check_ori: bool) -> Matcher {
        Matcher { nn_ratio, check_ori }
    }

    /// SearchByProjection(CurrentFrame, LastFrame, th, bMono): 앞 장면의 지도 점을 지금 장면에 투영해 짝짓기(등속 모형 추적)
    pub fn search_by_projection_last(&self, cur: &mut Frame, last: &Frame, mps: &BTreeMap<MpId, MapPoint>, th: f32, mono: bool) -> i32 {
        let mut nmatches = 0;
        let mut rot_hist: Vec<Vec<u32>> = vec![Vec::new(); HISTO_LENGTH];
        let tcw = cur.tcw.expect("지금 장면 위치");
        let twc = tcw.inverse().t;
        let tlw = last.tcw.expect("앞 장면 위치");
        let tlc = tlw.mul_point(&twc);
        let b = cur.ctx.cam.b;
        let forward = tlc[2] > b && !mono;
        let backward = -tlc[2] > b && !mono;
        let ctx = cur.ctx.clone();
        for i in 0..last.keys.len() {
            let Some(mid) = last.map_points[i] else { continue };
            if last.outlier[i] {
                continue;
            }
            let mp = &mps[&mid];
            let x3dc = tcw.mul_point(&mp.pos);
            let invzc = (1.0f64 / x3dc[2] as f64) as f32;
            if invzc < 0.0 {
                continue;
            }
            let uv = ctx.cam.project(&x3dc);
            if uv[0] < ctx.min_x || uv[0] > ctx.max_x || uv[1] < ctx.min_y || uv[1] > ctx.max_y {
                continue;
            }
            let last_oct = last.keys[i].octave;
            let radius = th * ctx.pyr.scale[last_oct as usize];
            let idx2 = if forward {
                cur.features_in_area(uv[0], uv[1], radius, last_oct, -1)
            } else if backward {
                cur.features_in_area(uv[0], uv[1], radius, 0, last_oct)
            } else {
                cur.features_in_area(uv[0], uv[1], radius, last_oct - 1, last_oct + 1)
            };
            if idx2.is_empty() {
                continue;
            }
            let dmp = &mp.desc;
            let mut best_dist = 256;
            let mut best_idx2 = -1i32;
            for &i2 in idx2.iter() {
                let i2u = i2 as usize;
                if let Some(m2) = cur.map_points[i2u] {
                    if mps[&m2].n_obs > 0 {
                        continue;
                    }
                }
                if cur.u_right[i2u] > 0.0 {
                    let ur = uv[0] - ctx.cam.bf * invzc;
                    let er = (ur - cur.u_right[i2u]).abs();
                    if er > radius {
                        continue;
                    }
                }
                let dist = desc_dist(dmp, &cur.desc[i2u]);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx2 = i2 as i32;
                }
            }
            if best_dist <= TH_HIGH {
                cur.map_points[best_idx2 as usize] = Some(mid);
                nmatches += 1;
                if self.check_ori {
                    rot_hist[rot_bin(last.keys[i].angle, cur.keys[best_idx2 as usize].angle)].push(best_idx2 as u32);
                }
            }
        }
        if self.check_ori {
            let (i1, i2, i3) = three_maxima(&rot_hist);
            for (i, v) in rot_hist.iter().enumerate() {
                let i = i as i32;
                if i != i1 && i != i2 && i != i3 {
                    for &j in v {
                        cur.map_points[j as usize] = None;
                        nmatches -= 1;
                    }
                }
            }
        }
        nmatches
    }

    /// SearchByProjection(F, 지역 지도 점들, th, bFarPoints, thFarPoints): isInFrustum 으로 채운 추적 칸을 써서 짝짓기
    pub fn search_by_projection_local(&self, f: &mut Frame, points: &[MpId], mps: &BTreeMap<MpId, MapPoint>, th: f32, far_points: bool, th_far: f32) -> i32 {
        let mut nmatches = 0;
        let factor = th != 1.0;
        let scale = f.ctx.pyr.scale.clone();
        for &mid in points {
            let mp = &mps[&mid];
            if !mp.track_in_view {
                continue;
            }
            if far_points && mp.track_depth > th_far {
                continue;
            }
            if mp.bad {
                continue;
            }
            let level = mp.track_scale_level;
            let mut r = radius_by_viewing_cos(mp.track_view_cos);
            if factor {
                r *= th;
            }
            let rs = r * scale[level as usize];
            let idx = f.features_in_area(mp.track_proj_x, mp.track_proj_y, rs, level - 1, level);
            if idx.is_empty() {
                continue;
            }
            let (mut best, mut best_lv, mut best2, mut best_lv2, mut best_idx) = (256, -1, 256, -1, -1i32);
            for &i in idx.iter() {
                let iu = i as usize;
                if let Some(m2) = f.map_points[iu] {
                    if mps[&m2].n_obs > 0 {
                        continue;
                    }
                }
                if f.u_right[iu] > 0.0 {
                    let er = (mp.track_proj_xr - f.u_right[iu]).abs();
                    if er > r * scale[level as usize] {
                        continue;
                    }
                }
                let dist = desc_dist(&mp.desc, &f.desc[iu]);
                if dist < best {
                    best2 = best;
                    best = dist;
                    best_lv2 = best_lv;
                    best_lv = f.keys[iu].octave;
                    best_idx = i as i32;
                } else if dist < best2 {
                    best_lv2 = f.keys[iu].octave;
                    best2 = dist;
                }
            }
            if best <= TH_HIGH {
                if best_lv == best_lv2 && best as f32 > self.nn_ratio * best2 as f32 {
                    continue;
                }
                if best_lv != best_lv2 || best as f32 <= self.nn_ratio * best2 as f32 {
                    f.map_points[best_idx as usize] = Some(mid);
                    nmatches += 1;
                }
            }
        }
        nmatches
    }

    /// SearchByBoW(KF, F): 같은 BoW 노드에 든 점끼리 짝짓기(재위치 · 기준 키프레임 추적). 돌려줌: (짝 수, F 의 점마다 지도 점)
    pub fn search_by_bow_kf_frame(&self, kf: &KeyFrame, f: &Frame, mps: &BTreeMap<MpId, MapPoint>) -> (i32, Vec<Option<MpId>>) {
        let mut matches: Vec<Option<MpId>> = vec![None; f.keys.len()];
        let mut nmatches = 0;
        let mut rot_hist: Vec<Vec<u32>> = vec![Vec::new(); HISTO_LENGTH];
        let kv: Vec<(&u32, &Vec<u32>)> = kf.feat.iter().collect();
        let fv: Vec<(&u32, &Vec<u32>)> = f.feat.iter().collect();
        let (mut a, mut b) = (0usize, 0usize);
        while a < kv.len() && b < fv.len() {
            if kv[a].0 == fv[b].0 {
                for &ikf in kv[a].1.iter() {
                    let Some(mid) = kf.map_points[ikf as usize] else { continue };
                    let mp = &mps[&mid];
                    if mp.bad {
                        continue;
                    }
                    let dkf = &kf.desc[ikf as usize];
                    let (mut best1, mut best_idx, mut best2) = (256, -1i32, 256);
                    for &i_f in fv[b].1.iter() {
                        if matches[i_f as usize].is_some() {
                            continue;
                        }
                        let dist = desc_dist(dkf, &f.desc[i_f as usize]);
                        if dist < best1 {
                            best2 = best1;
                            best1 = dist;
                            best_idx = i_f as i32;
                        } else if dist < best2 {
                            best2 = dist;
                        }
                    }
                    if best1 <= TH_LOW && (best1 as f32) < self.nn_ratio * best2 as f32 {
                        matches[best_idx as usize] = Some(mid);
                        if self.check_ori {
                            rot_hist[rot_bin(kf.keys[ikf as usize].angle, f.keys[best_idx as usize].angle)].push(best_idx as u32);
                        }
                        nmatches += 1;
                    }
                }
                a += 1;
                b += 1;
            } else if kv[a].0 < fv[b].0 {
                let key = *fv[b].0;
                a = kv.partition_point(|x| *x.0 < key);
            } else {
                let key = *kv[a].0;
                b = fv.partition_point(|x| *x.0 < key);
            }
        }
        if self.check_ori {
            let (i1, i2, i3) = three_maxima(&rot_hist);
            for (i, v) in rot_hist.iter().enumerate() {
                let i = i as i32;
                if i == i1 || i == i2 || i == i3 {
                    continue;
                }
                for &j in v {
                    matches[j as usize] = None;
                    nmatches -= 1;
                }
            }
        }
        (nmatches, matches)
    }

    /// SearchByProjection(F, KF, 이미 찾은 점, th, ORBdist): 재위치 뒤 키프레임 점을 더 찾기
    pub fn search_by_projection_kf(&self, cur: &mut Frame, kf: &KeyFrame, already: &std::collections::BTreeSet<MpId>, mps: &BTreeMap<MpId, MapPoint>, th: f32, orb_dist: i32) -> i32 {
        let mut nmatches = 0;
        let tcw = cur.tcw.expect("지금 장면 위치");
        let ow = tcw.inverse().t;
        let mut rot_hist: Vec<Vec<u32>> = vec![Vec::new(); HISTO_LENGTH];
        let ctx = cur.ctx.clone();
        for (i, m) in kf.map_points.iter().enumerate() {
            let Some(mid) = *m else { continue };
            let mp = &mps[&mid];
            if mp.bad || already.contains(&mid) {
                continue;
            }
            let x3dc = tcw.mul_point(&mp.pos);
            let uv = ctx.cam.project(&x3dc);
            if uv[0] < ctx.min_x || uv[0] > ctx.max_x || uv[1] < ctx.min_y || uv[1] > ctx.max_y {
                continue;
            }
            let dist3d = norm(&sub(&mp.pos, &ow));
            if dist3d < mp.min_distance_invariance() || dist3d > mp.max_distance_invariance() {
                continue;
            }
            let lv = mp.predict_scale(dist3d, &ctx.pyr);
            let radius = th * ctx.pyr.scale[lv as usize];
            let idx2 = cur.features_in_area(uv[0], uv[1], radius, lv - 1, lv + 1);
            if idx2.is_empty() {
                continue;
            }
            let (mut best, mut best_idx) = (256, -1i32);
            for &i2 in idx2.iter() {
                if cur.map_points[i2 as usize].is_some() {
                    continue;
                }
                let d = desc_dist(&mp.desc, &cur.desc[i2 as usize]);
                if d < best {
                    best = d;
                    best_idx = i2 as i32;
                }
            }
            if best <= orb_dist {
                cur.map_points[best_idx as usize] = Some(mid);
                nmatches += 1;
                if self.check_ori {
                    rot_hist[rot_bin(kf.keys[i].angle, cur.keys[best_idx as usize].angle)].push(best_idx as u32);
                }
            }
        }
        if self.check_ori {
            let (i1, i2, i3) = three_maxima(&rot_hist);
            for (i, v) in rot_hist.iter().enumerate() {
                let i = i as i32;
                if i != i1 && i != i2 && i != i3 {
                    for &j in v {
                        cur.map_points[j as usize] = None;
                        nmatches -= 1;
                    }
                }
            }
        }
        nmatches
    }

    /// SearchForTriangulation(KF1, KF2, bOnlyStereo, bCoarse): 지도 점이 없는 점끼리 BoW 노드 안에서 짝(극선 조건). 돌려줌: (idx1, idx2) 쌍
    pub fn search_for_triangulation(&self, kf1: &KeyFrame, kf2: &KeyFrame, only_stereo: bool, coarse: bool) -> Vec<(usize, usize)> {
        let cw = kf1.camera_center();
        let c2 = kf2.tcw.mul_point(&cw);
        let ep = kf2.ctx.cam.project(&c2);
        let t12 = kf1.tcw.mul(&kf2.twc);
        let r12 = t12.rot();
        let tt12 = t12.t;
        let (cam1, cam2) = (&kf1.ctx.cam, &kf2.ctx.cam);
        let mut matched2 = vec![false; kf2.keys.len()];
        let mut m12 = vec![-1i32; kf1.keys.len()];
        let mut nmatches = 0;
        let mut rot_hist: Vec<Vec<u32>> = vec![Vec::new(); HISTO_LENGTH];
        let v1: Vec<(&u32, &Vec<u32>)> = kf1.feat.iter().collect();
        let v2: Vec<(&u32, &Vec<u32>)> = kf2.feat.iter().collect();
        let (mut a, mut b) = (0usize, 0usize);
        while a < v1.len() && b < v2.len() {
            if v1[a].0 == v2[b].0 {
                for &i1 in v1[a].1.iter() {
                    let idx1 = i1 as usize;
                    if kf1.map_points[idx1].is_some() {
                        continue;
                    }
                    let stereo1 = kf1.u_right[idx1] >= 0.0;
                    if only_stereo && !stereo1 {
                        continue;
                    }
                    let kp1 = &kf1.keys[idx1];
                    let d1 = &kf1.desc[idx1];
                    let mut best = TH_LOW;
                    let mut best_idx2 = -1i32;
                    for &i2 in v2[b].1.iter() {
                        let idx2 = i2 as usize;
                        if matched2[idx2] || kf2.map_points[idx2].is_some() {
                            continue;
                        }
                        let stereo2 = kf2.u_right[idx2] >= 0.0;
                        if only_stereo && !stereo2 {
                            continue;
                        }
                        let dist = desc_dist(d1, &kf2.desc[idx2]);
                        if dist > TH_LOW || dist > best {
                            continue;
                        }
                        let kp2 = &kf2.keys[idx2];
                        if !stereo1 && !stereo2 {
                            let dex = ep[0] - kp2.x;
                            let dey = ep[1] - kp2.y;
                            if dex * dex + dey * dey < 100.0 * kf2.ctx.pyr.scale[kp2.octave as usize] {
                                continue;
                            }
                        }
                        if coarse || cam1.epipolar_constrain(cam2, kp1, kp2, &r12, &tt12, kf2.ctx.pyr.sigma2[kp2.octave as usize]) {
                            best_idx2 = idx2 as i32;
                            best = dist;
                        }
                    }
                    if best_idx2 >= 0 {
                        m12[idx1] = best_idx2;
                        nmatches += 1;
                        if self.check_ori {
                            rot_hist[rot_bin(kp1.angle, kf2.keys[best_idx2 as usize].angle)].push(idx1 as u32);
                        }
                    }
                }
                a += 1;
                b += 1;
            } else if v1[a].0 < v2[b].0 {
                let key = *v2[b].0;
                a = v1.partition_point(|x| *x.0 < key);
            } else {
                let key = *v1[a].0;
                b = v2.partition_point(|x| *x.0 < key);
            }
        }
        if self.check_ori {
            let (i1, i2, i3) = three_maxima(&rot_hist);
            for (i, v) in rot_hist.iter().enumerate() {
                let i = i as i32;
                if i == i1 || i == i2 || i == i3 {
                    continue;
                }
                for &j in v {
                    m12[j as usize] = -1;
                    nmatches -= 1;
                }
            }
        }
        let _ = nmatches;
        m12.iter().enumerate().filter(|(_, j)| **j >= 0).map(|(i, j)| (i, *j as usize)).collect()
    }

    /// SearchByBoW(KF1, KF2): 두 키프레임의 지도 점끼리 BoW 노드 안에서 짝(루프 · 합치기 후보 검사). 돌려줌: (짝 수, KF1 점마다 KF2 지도 점)
    pub fn search_by_bow_kf_kf(&self, kf1: &KeyFrame, kf2: &KeyFrame, mps: &BTreeMap<MpId, MapPoint>) -> (i32, Vec<Option<MpId>>) {
        let mut m12: Vec<Option<MpId>> = vec![None; kf1.map_points.len()];
        let mut matched2 = vec![false; kf2.map_points.len()];
        let mut rot_hist: Vec<Vec<u32>> = vec![Vec::new(); HISTO_LENGTH];
        let mut nmatches = 0;
        let v1: Vec<(&u32, &Vec<u32>)> = kf1.feat.iter().collect();
        let v2: Vec<(&u32, &Vec<u32>)> = kf2.feat.iter().collect();
        let (mut a, mut b) = (0usize, 0usize);
        while a < v1.len() && b < v2.len() {
            if v1[a].0 == v2[b].0 {
                for &i1 in v1[a].1.iter() {
                    let idx1 = i1 as usize;
                    let Some(m1) = kf1.map_points[idx1] else { continue };
                    if mps[&m1].bad {
                        continue;
                    }
                    let d1 = &kf1.desc[idx1];
                    let (mut best1, mut best_idx2, mut best2) = (256, -1i32, 256);
                    for &i2 in v2[b].1.iter() {
                        let idx2 = i2 as usize;
                        if matched2[idx2] {
                            continue;
                        }
                        let Some(m2) = kf2.map_points[idx2] else { continue };
                        if mps[&m2].bad {
                            continue;
                        }
                        let dist = desc_dist(d1, &kf2.desc[idx2]);
                        if dist < best1 {
                            best2 = best1;
                            best1 = dist;
                            best_idx2 = idx2 as i32;
                        } else if dist < best2 {
                            best2 = dist;
                        }
                    }
                    if best1 < TH_LOW && (best1 as f32) < self.nn_ratio * best2 as f32 {
                        m12[idx1] = kf2.map_points[best_idx2 as usize];
                        matched2[best_idx2 as usize] = true;
                        if self.check_ori {
                            rot_hist[rot_bin(kf1.keys[idx1].angle, kf2.keys[best_idx2 as usize].angle)].push(idx1 as u32);
                        }
                        nmatches += 1;
                    }
                }
                a += 1;
                b += 1;
            } else if v1[a].0 < v2[b].0 {
                let key = *v2[b].0;
                a = v1.partition_point(|x| *x.0 < key);
            } else {
                let key = *v1[a].0;
                b = v2.partition_point(|x| *x.0 < key);
            }
        }
        if self.check_ori {
            let (i1, i2, i3) = three_maxima(&rot_hist);
            for (i, v) in rot_hist.iter().enumerate() {
                let i = i as i32;
                if i == i1 || i == i2 || i == i3 {
                    continue;
                }
                for &j in v {
                    m12[j as usize] = None;
                    nmatches -= 1;
                }
            }
        }
        (nmatches, m12)
    }

    /// SearchByProjection(KF, Scw, 점들, 짝, th, ratioHamming): 루프 · 합치기 후보 키프레임에 Sim3 로 투영(카메라 project).
    /// matched 는 키프레임 점마다 이미 짝지어진 지도 점(들어 있는 점은 건너뜀)
    #[allow(clippy::too_many_arguments)]
    pub fn search_by_projection_sim3(&self, kf: &KeyFrame, scw: &Sim3, points: &[MpId], matched: &mut [Option<MpId>], mps: &BTreeMap<MpId, MapPoint>, th: i32, ratio_hamming: f32) -> i32 {
        self.sim3_proj(kf, scw, points, None, matched, mps, th, ratio_hamming, false)
    }
    /// SearchByProjection(KF, Scw, 점들, 점의 키프레임들, 짝, 짝 키프레임, th, ratioHamming): 투영은 fx·(x·(1/z))+cx 식
    #[allow(clippy::too_many_arguments)]
    pub fn search_by_projection_sim3_kfs(&self, kf: &KeyFrame, scw: &Sim3, points: &[MpId], points_kfs: &[KfId], matched: &mut [Option<MpId>], matched_kf: &mut [Option<KfId>], mps: &BTreeMap<MpId, MapPoint>, th: i32, ratio_hamming: f32) -> i32 {
        self.sim3_proj(kf, scw, points, Some((points_kfs, matched_kf)), matched, mps, th, ratio_hamming, true)
    }
    #[allow(clippy::too_many_arguments)]
    fn sim3_proj(&self, kf: &KeyFrame, scw: &Sim3, points: &[MpId], mut kfs: Option<(&[KfId], &mut [Option<KfId>])>, matched: &mut [Option<MpId>], mps: &BTreeMap<MpId, MapPoint>, th: i32, ratio_hamming: f32, manual: bool) -> i32 {
        let ctx = &kf.ctx;
        let cam = &ctx.cam;
        let tcw = scw.to_se3();
        let ow = tcw.inverse().t;
        let already: std::collections::BTreeSet<MpId> = matched.iter().flatten().copied().collect();
        let mut nmatches = 0;
        for (imp, &mid) in points.iter().enumerate() {
            let mp = &mps[&mid];
            if mp.bad || already.contains(&mid) {
                continue;
            }
            let p3dw = mp.pos;
            let p3dc = tcw.mul_point(&p3dw);
            if p3dc[2] < 0.0 {
                continue;
            }
            let (u, v) = if manual {
                let invz = 1.0 / p3dc[2];
                let x = p3dc[0] * invz;
                let y = p3dc[1] * invz;
                (cam.fx * x + cam.cx, cam.fy * y + cam.cy)
            } else {
                let uv = cam.project(&p3dc);
                (uv[0], uv[1])
            };
            if !kf.is_in_image(u, v) {
                continue;
            }
            let po = sub(&p3dw, &ow);
            let dist = norm(&po);
            if dist < mp.min_distance_invariance() || dist > mp.max_distance_invariance() {
                continue;
            }
            if (dot(&po, &mp.normal) as f64) < 0.5 * dist as f64 {
                continue;
            }
            let lv = mp.predict_scale(dist, &ctx.pyr);
            let radius = th as f32 * ctx.pyr.scale[lv as usize];
            let idxs = kf.features_in_area(u, v, radius);
            if idxs.is_empty() {
                continue;
            }
            let (mut best, mut best_idx) = (256, -1i32);
            for &idx in idxs.iter() {
                let idx = idx as usize;
                if matched[idx].is_some() {
                    continue;
                }
                let kl = kf.keys[idx].octave;
                if kl < lv - 1 || kl > lv {
                    continue;
                }
                let d = desc_dist(&mp.desc, &kf.desc[idx]);
                if d < best {
                    best = d;
                    best_idx = idx as i32;
                }
            }
            if best as f32 <= TH_LOW as f32 * ratio_hamming {
                matched[best_idx as usize] = Some(mid);
                if let Some((pk, mk)) = kfs.as_mut() {
                    mk[best_idx as usize] = Some(pk[imp]);
                }
                nmatches += 1;
            }
        }
        nmatches
    }

    /// Fuse(KF, 지도 점들, th, bRight = false): 키프레임에 투영해 이미 점이 있으면 합치고(Replace) 없으면 관측을 더함
    pub fn fuse(&self, kf_id: KfId, kfs: &mut BTreeMap<KfId, KeyFrame>, points: &[Option<MpId>], mps: &mut BTreeMap<MpId, MapPoint>, th: f32) -> i32 {
        let (tcw, ow, ctx) = {
            let kf = &kfs[&kf_id];
            (kf.tcw, kf.camera_center(), kf.ctx.clone())
        };
        let bf = ctx.cam.bf;
        let mut nfused = 0;
        for p in points {
            let Some(pid) = *p else { continue };
            let mp = &mps[&pid];
            if mp.bad || mp.obs.contains_key(&kf_id) {
                continue;
            }
            let p3dw = mp.pos;
            let p3dc = tcw.mul_point(&p3dw);
            if p3dc[2] < 0.0 {
                continue;
            }
            let invz = 1.0 / p3dc[2];
            let uv = ctx.cam.project(&p3dc);
            let kf = &kfs[&kf_id];
            if !kf.is_in_image(uv[0], uv[1]) {
                continue;
            }
            let ur = uv[0] - bf * invz;
            let po = sub(&p3dw, &ow);
            let dist3d = norm(&po);
            if dist3d < mp.min_distance_invariance() || dist3d > mp.max_distance_invariance() {
                continue;
            }
            if (dot(&po, &mp.normal) as f64) < 0.5 * dist3d as f64 {
                continue;
            }
            let lv = mp.predict_scale(dist3d, &ctx.pyr);
            let radius = th * ctx.pyr.scale[lv as usize];
            let idxs = kf.features_in_area(uv[0], uv[1], radius);
            if idxs.is_empty() {
                continue;
            }
            let (mut best, mut best_idx) = (256, -1i32);
            for &idx in idxs.iter() {
                let idx = idx as usize;
                let kp = &kf.keys[idx];
                let kl = kp.octave;
                if kl < lv - 1 || kl > lv {
                    continue;
                }
                let ex = uv[0] - kp.x;
                let ey = uv[1] - kp.y;
                if kf.u_right[idx] >= 0.0 {
                    let er = ur - kf.u_right[idx];
                    let e2 = ex * ex + ey * ey + er * er;
                    if (e2 * ctx.pyr.inv_sigma2[kl as usize]) as f64 > 7.8 {
                        continue;
                    }
                } else {
                    let e2 = ex * ex + ey * ey;
                    if (e2 * ctx.pyr.inv_sigma2[kl as usize]) as f64 > 5.99 {
                        continue;
                    }
                }
                let d = desc_dist(&mp.desc, &kf.desc[idx]);
                if d < best {
                    best = d;
                    best_idx = idx as i32;
                }
            }
            if best <= TH_LOW {
                let bi = best_idx as usize;
                if let Some(q) = kf.map_points[bi] {
                    if !mps[&q].bad {
                        if mps[&q].n_obs > mps[&pid].n_obs {
                            replace_map_point(pid, q, kfs, mps);
                        } else {
                            replace_map_point(q, pid, kfs, mps);
                        }
                    }
                } else {
                    mps.get_mut(&pid).unwrap().add_observation(&kfs[&kf_id], bi);
                    kfs.get_mut(&kf_id).unwrap().map_points[bi] = Some(pid);
                }
                nfused += 1;
            }
        }
        nfused
    }

    /// Fuse(KF, Scw, 점들, th, vpReplacePoint): Sim3 로 투영해 합침. 이미 점이 있으면 replace[i] 에 적고(바꾸기는 부른 쪽), 없으면 관측 추가
    #[allow(clippy::too_many_arguments)]
    pub fn fuse_sim3(&self, kf_id: KfId, kfs: &mut BTreeMap<KfId, KeyFrame>, scw: &Sim3, points: &[MpId], mps: &mut BTreeMap<MpId, MapPoint>, th: f32, replace: &mut [Option<MpId>]) -> i32 {
        let tcw = scw.to_se3();
        let ow = tcw.inverse().t;
        let (already, ctx) = {
            let kf = &kfs[&kf_id];
            (kf.map_point_set(mps), kf.ctx.clone())
        };
        let mut nfused = 0;
        for (imp, &pid) in points.iter().enumerate() {
            let mp = &mps[&pid];
            if mp.bad || already.contains(&pid) {
                continue;
            }
            let p3dw = mp.pos;
            let p3dc = tcw.mul_point(&p3dw);
            if p3dc[2] < 0.0 {
                continue;
            }
            let uv = ctx.cam.project(&p3dc);
            let kf = &kfs[&kf_id];
            if !kf.is_in_image(uv[0], uv[1]) {
                continue;
            }
            let po = sub(&p3dw, &ow);
            let dist3d = norm(&po);
            if dist3d < mp.min_distance_invariance() || dist3d > mp.max_distance_invariance() {
                continue;
            }
            if (dot(&po, &mp.normal) as f64) < 0.5 * dist3d as f64 {
                continue;
            }
            let lv = mp.predict_scale(dist3d, &ctx.pyr);
            let radius = th * ctx.pyr.scale[lv as usize];
            let idxs = kf.features_in_area(uv[0], uv[1], radius);
            if idxs.is_empty() {
                continue;
            }
            let (mut best, mut best_idx) = (i32::MAX, -1i32);
            for &idx in idxs.iter() {
                let idx = idx as usize;
                let kl = kf.keys[idx].octave;
                if kl < lv - 1 || kl > lv {
                    continue;
                }
                let d = desc_dist(&mp.desc, &kf.desc[idx]);
                if d < best {
                    best = d;
                    best_idx = idx as i32;
                }
            }
            if best <= TH_LOW {
                let bi = best_idx as usize;
                if let Some(q) = kf.map_points[bi] {
                    if !mps[&q].bad {
                        replace[imp] = Some(q);
                    }
                } else {
                    mps.get_mut(&pid).unwrap().add_observation(&kfs[&kf_id], bi);
                    kfs.get_mut(&kf_id).unwrap().map_points[bi] = Some(pid);
                }
                nfused += 1;
            }
        }
        nfused
    }

    /// SearchBySim3(KF1, KF2, 짝12, S12, th): 양쪽 지도 점을 서로에게 투영해 서로 고른 짝만 더함
    pub fn search_by_sim3(&self, kf1: &KeyFrame, kf2: &KeyFrame, matches12: &mut [Option<MpId>], s12: &Sim3, th: f32, mps: &BTreeMap<MpId, MapPoint>) -> i32 {
        let cam = &kf1.ctx.cam;
        let (fx, fy, cx, cy) = (cam.fx, cam.fy, cam.cx, cam.cy);
        let s21 = s12.inverse();
        let (n1, n2) = (kf1.map_points.len(), kf2.map_points.len());
        let mut am1 = vec![false; n1];
        let mut am2 = vec![false; n2];
        for i in 0..n1 {
            if let Some(m) = matches12[i] {
                am1[i] = true;
                let idx2 = mps[&m].obs.get(&kf2.id).map(|x| x.0).unwrap_or(-1);
                if idx2 >= 0 && (idx2 as usize) < n2 {
                    am2[idx2 as usize] = true;
                }
            }
        }
        let side = |src: &KeyFrame, dst: &KeyFrame, s: &Sim3, am: &[bool]| -> Vec<i32> {
            let mut out = vec![-1i32; src.map_points.len()];
            for (i, m) in src.map_points.iter().enumerate() {
                let Some(mid) = *m else { continue };
                if am[i] {
                    continue;
                }
                let mp = &mps[&mid];
                if mp.bad {
                    continue;
                }
                let pc = s.mul_point(&src.tcw.mul_point(&mp.pos));
                if pc[2] < 0.0 {
                    continue;
                }
                let invz = (1.0f64 / pc[2] as f64) as f32;
                let x = pc[0] * invz;
                let y = pc[1] * invz;
                let u = fx * x + cx;
                let v = fy * y + cy;
                if !dst.is_in_image(u, v) {
                    continue;
                }
                let dist3d = norm(&pc);
                if dist3d < mp.min_distance_invariance() || dist3d > mp.max_distance_invariance() {
                    continue;
                }
                let lv = mp.predict_scale(dist3d, &dst.ctx.pyr);
                let radius = th * dst.ctx.pyr.scale[lv as usize];
                let idxs = dst.features_in_area(u, v, radius);
                if idxs.is_empty() {
                    continue;
                }
                let (mut best, mut best_idx) = (i32::MAX, -1i32);
                for &idx in idxs.iter() {
                    let idx = idx as usize;
                    let kl = dst.keys[idx].octave;
                    if kl < lv - 1 || kl > lv {
                        continue;
                    }
                    let d = desc_dist(&mp.desc, &dst.desc[idx]);
                    if d < best {
                        best = d;
                        best_idx = idx as i32;
                    }
                }
                if best <= TH_HIGH {
                    out[i] = best_idx;
                }
            }
            out
        };
        let m1 = side(kf1, kf2, &s21, &am1);
        let m2 = side(kf2, kf1, s12, &am2);
        let mut found = 0;
        for i1 in 0..n1 {
            let idx2 = m1[i1];
            if idx2 >= 0 && m2[idx2 as usize] == i1 as i32 {
                matches12[i1] = kf2.map_points[idx2 as usize];
                found += 1;
            }
        }
        found
    }
}

/// MapPoint::Replace(old → new): old 의 관측을 new 로 옮기고 old 는 나쁨 표시(지도에서 지우기는 부른 쪽 Map 몫)
pub fn replace_map_point(old: MpId, new: MpId, kfs: &mut BTreeMap<KfId, KeyFrame>, mps: &mut BTreeMap<MpId, MapPoint>) {
    if old == new {
        return;
    }
    let (obs, nvisible, nfound) = {
        let o = mps.get_mut(&old).unwrap();
        o.bad = true;
        o.replaced = Some(new);
        (std::mem::take(&mut o.obs), o.visible, o.found)
    };
    for (k, (l, r)) in obs {
        let in_kf = mps[&new].obs.contains_key(&k);
        for idx in [l, r] {
            if idx == -1 {
                continue;
            }
            if !in_kf {
                kfs.get_mut(&k).unwrap().map_points[idx as usize] = Some(new);
                mps.get_mut(&new).unwrap().add_observation(&kfs[&k], idx as usize);
            } else {
                kfs.get_mut(&k).unwrap().map_points[idx as usize] = None;
            }
        }
    }
    let n = mps.get_mut(&new).unwrap();
    n.found += nfound;
    n.visible += nvisible;
    n.compute_distinctive_descriptors(kfs);
}
