"""YOLO26s-seg 정합성 기준과 시작 가중치.

1) 시작 가중치: yolo26s-seg.pt(COCO) → FSOW, 클래스 출력층(cv3·one2one_cv3 마지막 1×1, semseg 마지막)을 nc 로 새로 만든다.
     python pytools/ref26.py weights --pt yolo26s-seg.pt --out yolo26s-seg-nc1.fsow
2) 기준: 같은 가중치로 Ultralytics 를 fp32 학습 모드로 한 번 정방향, 출력마다 고정 난수 G 를 곱해 더한 값을 역방향.
   층 출력·머리 출력·proto·semseg·모든 가중치 기울기·BN running 통계를 남긴다(tests/parity26 이 비교).
     python pytools/ref26.py ref --weights yolo26s-seg-nc1.fsow --out ref26.fsow [--imgsz 256 --batch 2]
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import fsow  # noqa: E402


def build(nc=1):
    from ultralytics.nn.tasks import SegmentationModel
    m = SegmentationModel('yolo26s-seg.yaml', nc=nc, verbose=False).float()
    m.args = type('A', (), {})()
    return m


def cmd_weights(a):
    from ultralytics import YOLO
    src = YOLO(a.pt).model.float()
    t = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in src.state_dict().items() if 'num_batches' not in k}
    dst = build(a.nc)
    ref = {k: v for k, v in dst.state_dict().items() if 'num_batches' not in k}
    head = dst.model[-1]
    torch.manual_seed(0)
    for k, v in ref.items():
        if k not in t or t[k].shape != tuple(v.shape):
            # 클래스 수가 다른 층: 새 nn.Conv2d 기본 초기화(시드 0), 점수 bias 는 Ultralytics Detect.bias_init 과 같이
            w = v.detach().clone()
            if k.endswith('.weight'):
                torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            else:
                lvl = int(k.split('.')[3]) if ('cv3.' in k) else None
                if lvl is not None:
                    s = head.stride[lvl].item() if head.stride.sum() > 0 else (8, 16, 32)[lvl]
                    w[:] = math.log(5 / a.nc / (640 / s) ** 2)
                else:
                    w.zero_()
            print(f'{k}: {t.get(k, np.zeros(0)).shape} → {tuple(w.shape)}')
            t[k] = w.numpy().astype(np.float32)
    t = {k: t[k] for k in ref}
    fsow.save(a.out, t)
    print(f'{a.out}: {len(t)} tensors, {sum(x.size for x in t.values()):,} values')


def smooth_images(b, s, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) / s
    ims = []
    for _ in range(b):
        im = np.stack([0.5 + 0.3 * np.sin(6.28 * (xx * rng.uniform(1, 5) + yy * rng.uniform(1, 5)) + rng.uniform(0, 6))
                       for _ in range(3)], 0)
        ims.append(np.clip(im + 0.05 * rng.standard_normal(im.shape).astype(np.float32), 0, 1))
    return np.stack(ims).astype(np.float32)


def cmd_ref(a):
    torch.manual_seed(0)
    w = fsow.load(a.weights)
    m = build(a.nc)
    sd = {k: torch.from_numpy(np.asarray(v)) for k, v in w.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if 'num_batches' not in k] and not unexpected, (missing, unexpected)
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    x = torch.from_numpy(smooth_images(a.batch, a.imgsz))
    acts = {}
    for i, layer in enumerate(m.model[:-1]):
        layer.register_forward_hook(lambda mod, inp, out, i=i: acts.__setitem__(i, out.detach().clone()))
    out = m(x)
    o2m, o2o = out['one2many'], out['one2one']
    proto, sem = o2m['proto']
    tens = {
        'o2m.boxes': o2m['boxes'], 'o2m.scores': o2m['scores'], 'o2m.coef': o2m['mask_coefficient'],
        'o2o.boxes': o2o['boxes'], 'o2o.scores': o2o['scores'], 'o2o.coef': o2o['mask_coefficient'],
        'proto': proto, 'sem': sem,
    }
    g = torch.Generator().manual_seed(1)
    G = {k: torch.randn(v.shape, generator=g) for k, v in tens.items()}
    loss = sum((v * G[k]).sum() for k, v in tens.items())
    loss.backward()
    r = {'in': x.numpy()}
    for k, v in tens.items():
        r['out.' + k] = v.detach().numpy()
        r['G.' + k] = G[k].numpy()
    for i, v in acts.items():
        r[f'act.{i}'] = v.numpy()
    for n, p in m.named_parameters():
        r['grad.' + n] = p.grad.numpy() if p.grad is not None else np.zeros(p.shape, np.float32)
    for n, b in m.named_buffers():
        if 'running' in n:
            r['run.' + n] = b.numpy()
    fsow.save(a.out, {k: np.asarray(v, np.float32) for k, v in r.items()})
    print(f'{a.out}: {len(r)} tensors; loss {loss.item():.4f}; params {sum(p.numel() for p in m.parameters()):,}')


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    w = sp.add_parser('weights')
    w.add_argument('--pt', default='yolo26s-seg.pt')
    w.add_argument('--out', default='yolo26s-seg-nc1.fsow')
    w.add_argument('--nc', type=int, default=1)
    r = sp.add_parser('ref')
    r.add_argument('--weights', default='yolo26s-seg-nc1.fsow')
    r.add_argument('--out', default='ref26.fsow')
    r.add_argument('--imgsz', type=int, default=256)
    r.add_argument('--batch', type=int, default=2)
    r.add_argument('--nc', type=int, default=1)
    a = ap.parse_args()
    {'weights': cmd_weights, 'ref': cmd_ref}[a.cmd](a)


if __name__ == '__main__':
    main()
