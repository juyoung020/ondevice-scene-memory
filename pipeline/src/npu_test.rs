//! NPU 끝 알림 시험(보드): `ondevice npu-test --model <dxnn> [--n 300] [--jobs 1,2,4] [--wake thread,callback]`
//!
//! 같은 입력을 동기 실행(`dxrt_engine_run`)으로 한 번 돌린 출력을 기준으로, 끝 알림 길(`--npu-wake`)마다 K 장씩 함께 올려
//! ① 받을 버퍼에 쓰인 결과가 기준과 비트까지 같은지 ② 올림 → 알림 지연(p50 · p95 · 최대) ③ 알림이 도는 스레드 이름 · 코어
//! ④ (콜백 길) `wait` 없이도 작업 자리가 풀리는지(버퍼 수보다 많이 올려 2 초 안에 끝나는지)를 잰다.
//! 줄마다 JSON 하나, 끝 줄 `{"callback_pass": …}` = 콜백 길이 모든 K 에서 비트 같고 끝까지 돌았는가(`--npu-wake callback` 을 기본으로 바꿀 근거).
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(target_os = "linux")]
fn here() -> (String, i32) {
    unsafe extern "C" {
        fn sched_getcpu() -> i32;
    }
    let name = std::fs::read_to_string("/proc/thread-self/comm").map(|s| s.trim().to_string()).unwrap_or_default();
    (name, unsafe { sched_getcpu() })
}
#[cfg(not(target_os = "linux"))]
fn here() -> (String, i32) {
    (String::new(), -1)
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * p) as usize]
}

pub fn run(v: &common::cli::Argv) {
    use common::dxrt::{Engine, Wake};
    let model = v.get("--model").expect("--model <dxnn>");
    let n: usize = v.get("--n").and_then(|s| s.parse().ok()).unwrap_or(300);
    let jobs: Vec<usize> = v.get("--jobs").unwrap_or("1,2,4".into()).split(',').filter_map(|s| s.parse().ok()).collect();
    let wakes: Vec<Wake> = v.get("--wake").unwrap_or("thread,callback".into()).split(',').map(|s| if s == "callback" { Wake::Callback } else { Wake::Thread }).collect();
    // 기준: 동기 실행(알림 없는 엔진)
    let mut base = Engine::try_open(&model, 0, 2).unwrap_or_else(|e| panic!("{e}"));
    let mut seed = 0x9e3779b97f4a7c15u64;
    let input: Vec<u8> = (0..base.in_len)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 24) as u8
        })
        .collect();
    let mut want = Vec::new();
    base.run_into(&input, &mut want);
    drop(base);
    let mut cb_pass = true;
    let mut cb_ran = false;
    for &wake in &wakes {
        for &k in &jobs {
            let k = k.max(1);
            let mut e = match Engine::try_open(&model, 0, (k + 1) as i32) {
                Ok(e) => e,
                Err(err) => {
                    println!("{}", serde_json::json!({"wake": format!("{wake:?}"), "jobs": k, "error": err}));
                    continue;
                }
            };
            // 알림: (표, 받은 때, 스레드 이름, 코어)
            let got: Arc<(Mutex<Vec<(u64, Instant, String, i32)>>, std::sync::Condvar)> = Default::default();
            let g2 = got.clone();
            e.on_done(wake, "test", Arc::new(move |tag| {
                let (name, cpu) = here();
                g2.0.lock().unwrap().push((tag, Instant::now(), name, cpu));
                g2.1.notify_all();
            }));
            let mut outs: Vec<Vec<f32>> = (0..k).map(|_| Vec::new()).collect();
            let mut fly: Vec<Option<(i32, Instant)>> = vec![None; k];
            let (mut sent, mut done, mut bad) = (0usize, 0usize, 0usize);
            let (mut lat, mut threads, mut cpus) = (Vec::new(), std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
            let t0 = Instant::now();
            let mut stalled = false;
            while done < n {
                for (slot, f) in fly.iter_mut().enumerate() {
                    if f.is_none() && sent < n {
                        let id = e.submit_tagged(&input, &mut outs[slot], slot as u64);
                        *f = Some((id, Instant::now()));
                        sent += 1;
                    }
                }
                let mut g = got.0.lock().unwrap();
                if g.is_empty() {
                    let (g2, to) = got.1.wait_timeout(g, Duration::from_secs(5)).unwrap();
                    g = g2;
                    if to.timed_out() && g.is_empty() {
                        stalled = true;
                        break;
                    }
                }
                let batch: Vec<_> = g.drain(..).collect();
                drop(g);
                for (tag, at, name, cpu) in batch {
                    let slot = tag as usize;
                    let Some((id, c)) = fly[slot].take() else { continue };
                    if e.callback_mode() {
                        e.waiter().wait(id);
                    }
                    lat.push((at - c).as_secs_f64() * 1e3);
                    bad += (outs[slot].len() != want.len() || outs[slot].iter().zip(&want).any(|(a, b)| a.to_bits() != b.to_bits())) as usize;
                    threads.insert(name);
                    cpus.insert(cpu);
                    done += 1;
                }
            }
            let secs = t0.elapsed().as_secs_f64();
            // 콜백 길: wait 없이 버퍼 수(k+1)보다 많이 올려도 2 초 안에 모두 끝나는가(작업 자리가 콜백만으로 풀림)
            let no_wait_ok = (wake == Wake::Callback && !stalled).then(|| {
                got.0.lock().unwrap().clear();
                let m = 3 * (k + 1);
                let mut bufs: Vec<Vec<f32>> = (0..m).map(|_| Vec::new()).collect();
                let c = Instant::now();
                for (i, b) in bufs.iter_mut().enumerate() {
                    let _ = e.submit_tagged(&input, b, 1000 + i as u64);
                }
                let mut g = got.0.lock().unwrap();
                while g.len() < m && c.elapsed() < Duration::from_secs(2) {
                    g = got.1.wait_timeout(g, Duration::from_millis(200)).unwrap().0;
                }
                let ok = g.len() >= m;
                drop(g);
                if !ok {
                    std::mem::forget(bufs); // 아직 쓰는 중일 수 있는 버퍼는 놓지 않음
                }
                ok
            });
            if wake == Wake::Callback {
                cb_ran = true;
                cb_pass &= !stalled && bad == 0 && done == n;
            }
            let mut l2 = lat.clone();
            println!(
                "{}",
                serde_json::json!({"wake": format!("{wake:?}").to_lowercase(), "jobs": k, "n": done, "mismatch": bad, "stalled": stalled, "per_s": done as f64 / secs.max(1e-9),
                    "lat_ms": {"p50": pct(&mut l2, 0.5), "p95": pct(&mut l2, 0.95), "max": pct(&mut l2, 1.0)},
                    "notify_threads": threads, "notify_cpus": cpus, "no_wait_ok": no_wait_ok})
            );
            if stalled {
                // 멈춘 엔진은 닫다가 기다릴 수 있어 버림
                std::mem::forget(e);
            }
        }
    }
    println!("{}", serde_json::json!({"callback_pass": cb_ran && cb_pass}));
}
