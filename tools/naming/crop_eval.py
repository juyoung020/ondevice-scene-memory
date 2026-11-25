# 이름 붙이기 사진 전처리 비교(시험 도구): <out>/objects 의 사진 · 마스크로 같은 물체 묶음을 전처리만 바꿔 이름을 붙이고
# 사람이 사진을 보고 적은 정답(labels json: {"id": ["chair", ...]})과 대조. 사용: crop_eval.py <out> <labels.json> <model.onnx> <vocab.voc>
import json, sys, numpy as np, onnxruntime as ort
from PIL import Image, ImageFilter
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
def emb(im):
    a = np.asarray(im.convert("RGB").resize((256, 256), Image.BILINEAR), np.float32) / 127.5 - 1
    o = sess.run(None, {inp: a.transpose(2, 0, 1)[None]})
    v = o[-1].reshape(-1)[:768] if o[-1].size == 768 else o[0].reshape(-1)[:768]
    return v / np.linalg.norm(v)
d = json.load(open(f"{out}/view.json"))
objs = {o["id"]: o for o in d["objects"] if o.get("rgbd")}
def variants(rgb, mk):
    a = np.asarray(rgb, np.float32); m = (np.asarray(mk.resize(rgb.size)) > 127)[..., None]
    yield "raw", rgb
    ys, xs = np.nonzero(m[..., 0])
    if len(xs):
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        yield "stretch", rgb.crop((x0, y0, x1, y1))
        L = max(x1 - x0, y1 - y0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for f in (1.0, 1.15, 1.6):
            q = L * f / 2
            yield f"sq{f}", rgb.crop((int(cx - q), int(cy - q), int(cx + q), int(cy + q)))
    if 0: yield "dim50", Image.fromarray(np.where(m, a, (a + 128) / 2).astype(np.uint8))
    if 0: yield "dim75", Image.fromarray(np.where(m, a, a * 0.25 + 128 * 0.75).astype(np.uint8))
    bl = np.asarray(rgb.filter(ImageFilter.GaussianBlur(6)), np.float32)
    if 0: yield "blur", Image.fromarray(np.where(m, a, bl).astype(np.uint8))
    if 0: yield "blurdim", Image.fromarray(np.where(m, a, (bl + 128) / 2).astype(np.uint8))
res = {}
for i, ok in labels.items():
    o = objs[i]; rgb = Image.open(f"{out}/{o['rgbd']['rgb']}").convert("RGB"); mk = Image.open(f"{out}/{o['rgbd']['mask']}").convert("L")
    vs = {k: emb(im) for k, im in variants(rgb, mk)}
    vs["raw+flip"] = vs["raw"] + emb(rgb.transpose(Image.FLIP_LEFT_RIGHT))
    if "stretch" in vs:
        vs["ens3"] = vs["raw"] + vs["stretch"] + vs["sq1.0"]
        vs["ens2"] = vs["raw"] + vs["sq1.0"]
    for k, e in vs.items():
        c = V @ (e / np.linalg.norm(e)); j = np.argsort(-c)[:2]; nm = names[j[0]]; sure = c[j[0]] - c[j[1]] >= 0.003
        r = res.setdefault(k, [0, 0, 0, []])
        if not sure: r[2] += 1
        elif nm in ok: r[0] += 1
        else: r[1] += 1; r[3].append(f"O{i}:{nm}")
for k, (c, w, u, ws) in res.items():
    print(f"{k:8s} 맞음 {c} 틀림 {w} 안붙임 {u} | 붙은 것 중 {c/max(1,c+w):.0%} 전체 {c/len(labels):.0%}  {' '.join(ws)}")
