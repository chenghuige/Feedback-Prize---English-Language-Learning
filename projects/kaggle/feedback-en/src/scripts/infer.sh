ddp ./main.py \
  --pretrain=$1 \
  --pretrain_restart=0 \
  --mn=$1.infer \
  --restore_configs \
  --mode=test \
  --save_final=0 \
  --use_v1 \
  $*

