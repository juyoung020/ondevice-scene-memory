"""학습기 EMA 가중치(FSOW, PyTorch 이름) → Ultralytics .pt (추론·ONNX/TensorRT/DX-COM 내보내기용).

  .venv/bin/python pytools/fsow_to_pt.py runs/r1/ema_last.fsow runs/r1/model.pt

구조는 가중치 이름으로 고른다:
  - YOLO26s-seg(클래스 1): one2one 머리(model.23.one2one_cv2…)가 있으면. 정의는 Ultralytics 의 yolo26s-seg.yaml.
  - YOLOv8s-seg(FastSAM-s 와 같은 구조, 클래스 1): 그 밖. reg_max 는 상자 가지 출력(4·reg_max)에서 읽는다.
"""
import argparse

import torch

import fsow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--yaml-from', default='weights/FastSAM-s.pt', help='v8: 모델 정의(yaml)를 가져올 Ultralytics 파일')
    a = ap.parse_args()
    from ultralytics import YOLO
    from ultralytics.nn.tasks import SegmentationModel
    t = fsow.load(a.src)
    if 'model.23.one2one_cv2.0.2.weight' in t:
        y = YOLO('yolo26s-seg.yaml', task='segment')
        m = SegmentationModel('yolo26s-seg.yaml', ch=3, nc=1, verbose=False)
        what = 'YOLO26s-seg'
    else:
        y = YOLO(a.yaml_from)
        cfg = dict(y.model.yaml)
        cfg['reg_max'] = int(t['model.22.cv2.0.2.weight'].shape[0] // 4)
        m = SegmentationModel(cfg, ch=3, nc=1, verbose=False)
        what = f'YOLOv8s-seg reg_max {cfg["reg_max"]}'
    sd = m.state_dict()
    for k, v in t.items():
        assert k in sd, f'unknown tensor {k}'
        assert tuple(sd[k].shape) == v.shape, f'{k}: {tuple(sd[k].shape)} vs {v.shape}'
        sd[k] = torch.from_numpy(v).to(sd[k].dtype)
    missing = [k for k in sd if k not in t and 'num_batches' not in k and 'dfl' not in k]
    assert not missing, f'missing in {a.src}: {missing[:5]}'
    m.load_state_dict(sd)
    m.names = {0: 'object'}
    y.model = m
    y.save(a.dst)
    print(f'{a.dst}: {len(t)} tensors from {a.src}, {what}')


if __name__ == '__main__':
    main()
