"""orb_maps*.onnx → .dxnn (DX-COM 2.4.1 파이썬 API, CPU 보정). 시험 · 변환 도구.
쓰기: ~/dx241/bin/python compile_dx.py <model.onnx> <mosdir> <outdir> [보정 장수=64] [--diag]
보정 자료 = fr1_desk 모자이크(orbnpu_bench mosaic)를 고르게 건너뛰며."""
import sys, os, numpy as np, torch
import dx_com


class Mos(torch.utils.data.Dataset):
    def __init__(self, d, n):
        W, H = map(int, open(f"{d}/layout.txt").readline().split()[:2])
        fs = sorted(f for f in os.listdir(d) if f.endswith(".raw"))
        step = max(1, len(fs) // n)
        self.fs = [os.path.join(d, f) for f in fs[::step][:n]]
        self.W, self.H = W, H

    def __len__(self):
        return len(self.fs)

    def __getitem__(self, i):
        a = np.fromfile(self.fs[i], np.uint8).reshape(1, self.H, self.W).astype(np.float32)
        return torch.from_numpy(a)


def main():
    onnx, md, out = sys.argv[1:4]
    n = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 64
    dl = torch.utils.data.DataLoader(Mos(md, n), batch_size=1)
    dx_com.compile(model=onnx, output_dir=out, dataloader=dl, calibration_method="ema", calibration_num=n,
                   quantization_device="cpu", quant_diagnosis="--diag" in sys.argv)


if __name__ == "__main__":
    main()
