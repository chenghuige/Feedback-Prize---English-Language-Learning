ddp ./main.py \
  --pretrain=$1 \
  --pretrain_restart=0 \
  --mn=$1.eval \
  --restore_configs \
  --mode=valid \
  --save_final=0 \
  $*

