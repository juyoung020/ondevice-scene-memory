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
}
