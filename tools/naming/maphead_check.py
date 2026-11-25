# Rust maskpool 시험(head_matches_onnx)과 같은 입력 · 무게로 파이썬 머리 값을 출력(두 출력 비교용)
import sys, numpy as np, onnxruntime as ort
from maphead import Head
P = sys.argv[1]
x = np.sin(np.arange(3 * 256 * 256, dtype=np.float32) * np.float32(0.001)).reshape(1, 3, 256, 256)
s = ort.InferenceSession(P, providers=["CPUExecutionProvider"])
lh, po = s.run(["last_hidden_state", "pooler_output"], {s.get_inputs()[0].name: x})
H = Head(P)
w = np.array([(k % 4 + 1) / 4 if k % 8 < 4 and k // 8 < 4 else 0 for k in range(64)], np.float32)
m = H(lh[0], w)
print("python: 모든 무게 1 최대 차", np.abs(H(lh[0]) - po[0]).max(), "마스크 판 앞 4 값", m[:4], "합", m.sum())
