#!/bin/sh

export MKL_NUM_THREADS=3

SEED=670352551
MEMORY=1024
CEMB_SIZE=10
WEMBED_SIZE=100
DEMBED_SIZE=10
HIDDEN_SIZE=500
N_LAYER=1
MLP_SIZE=100
EPOCH=10
OPTIMIZER="Adam"
LOSS="average"
DICT="./data/feat_dict.pickle"
RATE=0.2

TRAIN="./data/atb-train_14mf+bw.conll"
DEV="./data/atb-dev_14mf+bw.conll"

if [ $1 = "joint" ]; then
  if [ $2 = "no_dict" ]; then
    nohup python src/joint.py --dynet-seed $SEED --dynet-mem $MEMORY\
    $CEMB_SIZE $WEMBED_SIZE $DEMBED_SIZE $HIDDEN_SIZE $N_LAYER $MLP_SIZE\
    --train $TRAIN --dev $DEV --epoch $EPOCH --optimizer $OPTIMIZER\
    --loss $LOSS >/dev/null 2>&1 &
  elif [ $2 = "dict_all" ]; then
    nohup python src/joint.py --dynet-seed $SEED --dynet-mem $MEMORY\
    $CEMB_SIZE $WEMBED_SIZE $DEMBED_SIZE $HIDDEN_SIZE $N_LAYER $MLP_SIZE\
    --train $TRAIN --dev $DEV --epoch $EPOCH --optimizer $OPTIMIZER\
    --loss $LOSS --use_dict --tag_dict $DICT --dict_all>/dev/null 2>&1 &
  elif [ $2 = "dict_one" ]; then
    for FEAT in "POS" "PRC3" "PRC2" "PRC1" "PRC0" "PER" "ASP"\
    "VOX" "MOD" "GEN" "NUM" "STT" "CAS" "ENC0"
    do
      nohup python src/joint.py --dynet-seed $SEED --dynet-mem $MEMORY\
      $CEMB_SIZE $WEMBED_SIZE $DEMBED_SIZE $HIDDEN_SIZE $N_LAYER $MLP_SIZE\
      --train $TRAIN --dev $DEV --epoch $EPOCH --optimizer $OPTIMIZER\
      --loss $LOSS --use_dict --tag_dict $DICT --feature $FEAT>/dev/null 2>&1 &
    done
  fi
elif [ $1 = "separate" ]; then
  for FEAT in "POS" "PRC3" "PRC2" "PRC1" "PRC0" "PER" "ASP"\
  "VOX" "MOD" "GEN" "NUM" "STT" "CAS" "ENC0"
  do
    if [ $2 = "no_dict" ]; then
      nohup python src/separate.py --dynet-seed $SEED --dynet-mem $MEMORY\
      $CEMB_SIZE $WEMBED_SIZE $DEMBED_SIZE $HIDDEN_SIZE $N_LAYER $MLP_SIZE\
      --feature $FEAT --train $TRAIN --dev $DEV --epoch $EPOCH\
      --optimizer $OPTIMIZER>/dev/null 2>&1 &
    elif [ $2 = "dict_all" ]; then
      nohup python src/separate.py --dynet-seed $SEED --dynet-mem $MEMORY\
      $CEMB_SIZE $WEMBED_SIZE $DEMBED_SIZE $HIDDEN_SIZE $N_LAYER $MLP_SIZE\
      --feature $FEAT --train $TRAIN --dev $DEV --epoch $EPOCH\
      --optimizer $OPTIMIZER --use_dict --tag_dict $DICT --dict_all>/dev/null 2>&1 &
    fi
  done
fi
