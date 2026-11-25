# 중간 층 가리기 비교(시험 도구): midmask.py 로 만든 모델들(층 L 부터 마스크 밖 키를 가림)과 원본을 같은 대표 사진 · 정답으로 견줌. 이름 = 마스크 토큰 머리(maphead)
# 사용: mid_eval.py <out> <labels.json> <원본.onnx> <vocab.voc> <L모델.onnx>...
import json, sys, numpy as np, onnxruntime as ort
from PIL import Image
from maphead import Head
out, lab, model, voc = sys.argv[1:5]; mids = sys.argv[5:]
labels = {int(k): v for k, v in json.load(open(lab, encoding="utf-8")).items()}
b = open(voc, "rb").read(); p = 0
def line():
    global p
    e = b.index(b"\n", p); s = b[p:e].decode(); p = e + 1; return s
n = int(line().split()[2]); names = [line().split(":")[0] for _ in range(n)]
V = np.frombuffer(b[p:], dtype="<f4").reshape(n, 768)
H = Head(model)
sess = {"none": ort.InferenceSession(model, providers=["CPUExecutionProvider"])}
for m in mids: sess[m.split("_")[-1].split(".")[0]] = ort.InferenceSession(m, providers=["CPUExecutionProvider"])
objs = {o["id"]: o for o in json.load(open(f"{out}/view.json"))["objects"] if o.get("rgbd")}
res = {}
for i, ok in labels.items():
    o = objs[i]; rgb = Image.open(f"{out}/{o['rgbd']['rgb']}").convert("RGB"); mk = Image.open(f"{out}/{o['rgbd']['mask']}").convert("L").resize(rgb.size)
    x = (np.asarray(rgb.resize((256, 256), Image.BILINEAR), np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
    pm = np.asarray(mk.resize((256, 256), Image.BILINEAR), np.float32).reshape(8, 32, 8, 32).mean((1, 3)).reshape(64) / 255
    if pm.max() <= 0: pm = np.ones(64, np.float32)
    kb = np.where(pm > 0, np.log(np.maximum(pm, 1e-9)), -1e4).astype(np.float32).reshape(1, 1, 1, 64)
    for k, s in sess.items():
        feed = {s.get_inputs()[0].name: x}
        if k != "none": feed["key_bias"] = kb
        lh = s.run(["last_hidden_state"], feed)[0][0]
        e = H(lh, pm); e = e / np.linalg.norm(e)
        c = V @ e; j = np.argsort(-c)[:2]; nm = names[j[0]]; sure = c[j[0]] - c[j[1]] >= 0.003
        r = res.setdefault(k, [0, 0, 0, []])
        if not sure: r[2] += 1
        elif nm in ok: r[0] += 1
        else: r[1] += 1; r[3].append(f"O{i}:{nm}")
for k, (c, w, u, ws) in res.items():
    print(f"{k:6s} 맞음 {c} 틀림 {w} 안붙임 {u} | 붙은 것 중 {c/max(1,c+w):.0%}  {' '.join(ws)}")
