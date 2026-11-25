# 1 등 · 2 등 코사인 차 문턱별 맞음(시험 도구): 마스크 토큰 이름, 본 횟수 >= 12 만. 사용: margin_eval.py <out> <labels> <model> <vocab>
import json, sys, numpy as np, onnxruntime as ort
from PIL import Image
from maphead import Head
out, lab, model, voc = sys.argv[1:5]
labels = {int(k): v for k, v in json.load(open(lab, encoding="utf-8")).items()}
b = open(voc, "rb").read(); p = 0
def line():
    global p
    e = b.index(b"\n", p); s = b[p:e].decode(); p = e + 1; return s
n = int(line().split()[2]); names = [line().split(":")[0] for _ in range(n)]
V = np.frombuffer(b[p:], dtype="<f4").reshape(n, 768)
H = Head(model); s = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
objs = {o["id"]: o for o in json.load(open(f"{out}/view.json"))["objects"] if o.get("rgbd")}
rows = []
for i, ok in labels.items():
    o = objs[i]
    if o["n_obs"] < 12: continue
    rgb = Image.open(f"{out}/{o['rgbd']['rgb']}").convert("RGB"); mk = Image.open(f"{out}/{o['rgbd']['mask']}").convert("L").resize(rgb.size)
    x = (np.asarray(rgb.resize((256, 256), Image.BILINEAR), np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
    pm = np.asarray(mk.resize((256, 256), Image.BILINEAR), np.float32).reshape(8, 32, 8, 32).mean((1, 3)).reshape(64) / 255
    lh = s.run(["last_hidden_state"], {s.get_inputs()[0].name: x})[0][0]
    e = H(lh, pm if pm.max() > 0 else None); e /= np.linalg.norm(e)
    c = V @ e; j = np.argsort(-c)[:2]
    rows.append((c[j[0]] - c[j[1]], names[j[0]] in ok, i, names[j[0]]))
for t in (0.003, 0.005, 0.008, 0.01, 0.015, 0.02):
    k = [r for r in rows if r[0] >= t]; c = sum(r[1] for r in k)
    print(f"문턱 {t}: 붙음 {len(k)} 맞음 {c} ({c/max(1,len(k)):.0%})")
for r in sorted(rows): print(f"{r[0]:.4f} {'O' if r[1] else 'X'} O{r[2]} {r[3]}")
