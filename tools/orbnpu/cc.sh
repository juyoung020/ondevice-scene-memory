#!/bin/bash
# 컴파일을 WSL 안에서 떨어뜨려 돌린다(호출 창이 끊겨도 계속)
cd ~/orbnpu_out
rm -rf dx_maps
{ time ~/dx241/bin/python /mnt/c/ondevice-orb-npu/tools/orbnpu/compile_dx.py orb_maps.onnx mos dx_maps 64 ; } > dx_compile.log 2>&1
echo DONE >> dx_compile.log
