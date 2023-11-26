#!/bin/sh

arch=$1
resume=$2

if [ $# -eq 3 ]; then
  partition=$3
else
  partition=VI_Face_1080TI
fi

printf '==================== Market1501 ===================\n'
./scripts/market/test.sh ${arch} ${resume} ${partition}
printf '==================== Market1501 ===================\n\n'

printf '===================== DukeMTMC ====================\n'
./scripts/duke/test.sh ${arch} ${resume} ${partition}
printf '===================== DukeMTMC ====================\n\n'

printf '====================== CUHK03 =====================\n'
./scripts/cuhk03/test.sh ${arch} ${resume} ${partition}
printf '====================== CUHK03 =====================\n\n'

printf '====================== MSMT17 =====================\n'
./scripts/msmt17/test.sh ${arch} ${resume} ${partition}
printf '====================== MSMT17 =====================\n\n'
