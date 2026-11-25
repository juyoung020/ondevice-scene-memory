# 마스크 안쪽으로 이름 붙이기 비교(시험 도구): <out>/objects 대표 사진 · 마스크로 사진 벡터를 여러 방식으로 만들어 눈 정답(labels json)과 대조.
# 정답 "?" = 무엇이든 이름 붙으면 틀림(안 붙이면 맞음 아님 · 틀림 아님). 사용: mask_eval.py <out> <labels.json> <model.onnx> <vocab.voc>
import json, sys, numpy as np, onnxruntime as ort
from PIL import Image
from maphead import Head
out, lab, model, voc = sys.argv[1:5]
labels = {int(k): v for k, v in json.load(open(lab, encoding="utf-8")).items()}
b = open(voc, "rb").read(); p = 0
def line():
    global p
    e = b.index(b"\n", p); s = b[p:e].decode(); p = e + 1; return s
h = line().split(); n = int(h[2])
names = [line().split(":")[0] for _ in range(n)]
V = np.frombuffer(b[p:], dtype="<f4").reshape(n, 768)
sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
outs = [o.name for o in sess.get_outputs()]
H = Head(model)
def run(im):
    a = np.asarray(im.convert("RGB").resize((256, 256), Image.BILINEAR), np.float32) / 127.5 - 1
    return dict(zip(outs, sess.run(None, {inp: a.transpose(2, 0, 1)[None]})))
def nz(v): return v / np.linalg.norm(v)
d = json.load(open(f"{out}/view.json"))
objs = {o["id"]: o for o in d["objects"] if o.get("rgbd")}
def dim(a, m, k): return Image.fromarray(np.where(m, a, a + (128 - a) * k).astype(np.uint8))
res = {}
for i, ok in labels.items():
    o = objs[i]; rgb = Image.open(f"{out}/{o['rgbd']['rgb']}").convert("RGB"); mk = Image.open(f"{out}/{o['rgbd']['mask']}").convert("L").resize(rgb.size)
    a = np.asarray(rgb, np.float32); m = (np.asarray(mk) > 127)[..., None]
    vs = {}
    pm = np.asarray(mk.resize((256, 256), Image.BILINEAR), np.float32).reshape(8, 32, 8, 32).mean((1, 3)).reshape(64) / 255
    for src, k in (("raw", 0.0), ("ctx60", 0.6)):
        r = run(dim(a, m, k)); lh = r["last_hidden_state"][0]
        vs[src] = nz(r["pooler_output"].reshape(-1))
        for eps in (0.0, 0.05, 0.2):
            w = pm + eps
            if w.max() <= 0: w = np.ones(64)
            vs[f"{src}_mp{eps}"] = nz(H(lh, w))
    for k, e in vs.items():
        c = V @ e; j = np.argsort(-c)[:2]; nm = names[j[0]]; sure = c[j[0]] - c[j[1]] >= 0.003
        r = res.setdefault(k, [0, 0, 0, []])
        if not sure: r[2] += 1
        elif nm in ok: r[0] += 1
        else: r[1] += 1; r[3].append(f"O{i}:{nm}")
for k, (c, w, u, ws) in res.items():
    print(f"{k:22s} 맞음 {c} 틀림 {w} 안붙임 {u} | 붙은 것 중 {c/max(1,c+w):.0%} 전체 {c/len(labels):.0%}  {' '.join(ws)}")
