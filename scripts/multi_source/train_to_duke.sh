#!/bin/sh
ARCH=$1
PARTITION=$2
NUM_GPUs=$3
DESC=$4
SEED=0
if [[ $# -eq 5 ]]; then
  port=${5}
else
  port=23456
fi

export PATH=~/.local/bin/:$PATH

GLOG_vmodule=MemcachedClient=-1 \
srun  -p ${PARTITION} \
      -n${NUM_GPUs} --gres=gpu:${NUM_GPUs} --ntasks-per-node=8 \
      --job-name=R-SC210077.00108 \
python -u examples/train_multi_source.py -a ${ARCH} --seed ${SEED} --margin 0.3 \
	--num-instances 4 -b 32 -j 4 --warmup-step 1000 --lr 0.001 --milestones 4000 8000 --iters 12000 --port ${port} \
	--alpha 0.1 --alpha-scheduler constant --alpha-milestones 4000 8000 \
	--logs-dir logs/MAML_to_duke/${ARCH}-${DESC} \
	--train-lists /mnt/lustre/viface/share/ReID/data_list/MSMT17_V1/train.txt \
	/mnt/lustre/viface/share/ReID/data_list/market1501/train.txt \
	/mnt/lustre/viface/share/ReID/data_list/cuhk03_1/train.txt \
	--root /mnt/lustre/viface/share/ReID/ReID_dataset/datasets8 \
	--query-list /mnt/lustre/viface/share/ReID/data_list/dukemtmc-reid/probe.txt \
  --gallery-list /mnt/lustre/viface/share/ReID/data_list/dukemtmc-reid/gallery.txt \
  --validate