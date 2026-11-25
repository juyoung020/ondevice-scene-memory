#!/bin/bash
# Rust 하네스 비교: run_rs.sh <n> <기준.bin> — CPU fast · ONNX+fast_npu · ONNX+fast_npu(NOISE)
cd ~/orbnpu_out
B=~/tgt-orbnpu/release/orbnpu_rs
C="$HOME/sfm/bin/python /mnt/c/ondevice-orb-npu/tools/orbnpu/cmp_kp.py"
n=$1; ref=$2
echo "== Rust CPU fast"; $B mos $n rs_cpu.bin && $C $ref rs_cpu.bin
echo "== Rust ONNX S + fast_npu"; $B mos $n rs_npu.bin orb_maps.onnx && $C $ref rs_npu.bin
echo "== 같은 + NOISE"; NOISE=1 $B mos $n rs_noise.bin orb_maps.onnx && $C $ref rs_noise.bin
