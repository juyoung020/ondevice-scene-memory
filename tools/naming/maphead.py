# SigLIP2 그림 인코더의 끝 머리(주의 풀링 MAP head)를 numpy 로: 마지막 토큰(last_hidden_state)에서 고른 토큰만 보고 사진 벡터를 만든다(시험 도구)
import numpy as np, onnx
from onnx import numpy_helper
class Head:
    def __init__(s, path):
        m = onnx.load(path)
        I = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
        g = lambda k: I[k].astype(np.float32)
        s.probe = g("/vision_model/head/Expand_output_0").reshape(768)
        s.wq, s.bq = g("/vision_model/head/attention/Transpose_2_output_0"), g("/vision_model/head/attention/Slice_3_output_0")
        s.wkv, s.bkv = g("/vision_model/head/attention/Transpose_3_output_0"), g("/vision_model/head/attention/Slice_4_output_0")
        s.wo, s.bo = g("vision_model.head.attention.out_proj.weight"), g("vision_model.head.attention.out_proj.bias")
        s.lw, s.lb = g("vision_model.head.layernorm.weight"), g("vision_model.head.layernorm.bias")
        s.w1, s.b1 = g("onnx::MatMul_2025"), g("vision_model.head.mlp.fc1.bias")
        s.w2, s.b2 = g("onnx::MatMul_2026"), g("vision_model.head.mlp.fc2.bias")
    def __call__(s, h, w=None):
        """h: [N, 768] 토큰, w: [N] 토큰 무게(0 = 안 봄, None = 모두 1). 돌려줌 [768]"""
        q = (s.probe @ s.wq + s.bq).reshape(12, 64)
        kv = h @ s.wkv + s.bkv
        k, v = kv[:, :768].reshape(-1, 12, 64), kv[:, 768:].reshape(-1, 12, 64)
        a = np.einsum("hd,nhd->hn", q, k) / 8.0
        if w is not None:
            a = a + np.log(np.maximum(w, 1e-9))[None]
        a = np.exp(a - a.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
        o = np.einsum("hn,nhd->hd", a, v).reshape(768)
        r = o @ s.wo.T + s.bo
        x = (r - r.mean()) / np.sqrt(r.var() + 1e-6) * s.lw + s.lb
        u = x @ s.w1 + s.b1
        u = 0.5 * u * (1 + np.tanh(0.7978845608 * (u + 0.044715 * u ** 3)))
        return r + u @ s.w2 + s.b2
