# 인코더 중간 층부터 마스크 밖 토큰을 가리는 시험(시험 도구): vision_model.onnx 의 층 L.. 의 Softmax 앞에 키 무게 bias([1,1,1,64], 입력 key_bias)를 더한 모델을 만든다.
# 사용: midmask.py <원본.onnx> <L> <출력.onnx>
import sys, onnx
from onnx import helper, TensorProto
src, L, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = onnx.load(src)
g = m.graph
g.input.append(helper.make_tensor_value_info("key_bias", TensorProto.FLOAT, [1, 1, 1, 64]))
new = []
for n in g.node:
    if n.op_type == "Softmax" and "/encoder/layers." in n.name and int(n.name.split("layers.")[1].split("/")[0]) >= L:
        t = n.input[0] + "_kb"
        new.append(helper.make_node("Add", [n.input[0], "key_bias"], [t], name=n.name + "_kb"))
        n.input[0] = t
    new.append(n)
del g.node[:]
g.node.extend(new)
onnx.save(m, dst)
