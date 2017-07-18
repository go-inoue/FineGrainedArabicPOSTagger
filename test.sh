#!/bin/sh

export MKL_NUM_THREADS=4

SEED=670352551
MEMORY=1024
CEMB_SIZE=10
WEMBED_SIZE=100
DEMBED_SIZE=10
HIDDEN_SIZE=250
N_LAYER=1
MLP_SIZE=100

TRAIN="./data/atb-train_14mf+bw.conll"
DEV="./data/atb-dev_14mf+bw.conll"
TEST="./data/atb-test_14mf+bw.conll"
DICT="./data/feat_dict.pickle"

if [ $1 = "joint" ]; then
  if [ $2 = "no_dict" ]; then
    MODEL_DIR="./log/joint_no_dict*"
    python src/joint.py --dynet-seed 670352551 --dynet-mem 2048\
    $CEMB_SIZE $WEMBED_SIZE $DEMBED_SIZE $HIDDEN_SIZE $N_LAYER $MLP_SIZE\
    --mode test --train $TRAIN --dev $DEV --test $TEST --model_dir $MODEL_DIR\
    --save_conll
  elif [ $2 = "dict" ]; then
    MODEL_DIR="./log/joint_dict*"
    python src/joint.py --dynet-seed 670352551 --dynet-mem 2048\
    $CEMB_SIZE $WEMBED_SIZE $DEMBED_SIZE $HIDDEN_SIZE $N_LAYER $MLP_SIZE\
    --mode test --train $TRAIN --dev $DEV --test $TEST --model_dir $MODEL_DIR\
    --save_conll --use_dict --tag_dict $DICT
  elif [ $2 = "dict_one" ]; then
    for FEAT in "POS" "PRC3" "PRC2" "PRC1" "PRC0" "PER" "ASP"\
    "VOX" "MOD" "GEN" "NUM" "STT" "CAS" "ENC0"
    do
      MODEL_DIR="./log/joint_dict_$FEAT*"
      python src/joint.py --dynet-seed 670352551 --dynet-mem 2048\
      $CEMB_SIZE $WEMBED_SIZE $DEMBED_SIZE $HIDDEN_SIZE $N_LAYER $MLP_SIZE\
      --mode test --train $TRAIN --dev $DEV --test $TEST --model_dir $MODEL_DIR\
      --save_conll --use_dict --tag_dict $DICT --feature $FEAT
    done
  fi
elif [ $1 = "separate" ]; then
  for FEAT in "POS" "PRC3" "PRC2" "PRC1" "PRC0" "PER" "ASP" \
  "VOX" "MOD" "GEN" "NUM" "STT" "CAS" "ENC0"
  do
    if [ $2 = "no_dict" ]; then
      MODEL_DIR="./log/separate/$FEAT*"
      python src/separate.py --dynet-seed 670352551 --dynet-mem 2048 \
      $CEMB_SIZE $WEMBED_SIZE $DEMBED_SIZE $HIDDEN_SIZE $N_LAYER $MLP_SIZE \
      --mode test --train $TRAIN --dev $DEV --test $TEST \
      --model_dir $MODEL_DIR --feature $FEAT --save_conll
    elif [ $2 = "dict" ]; then
      MODEL_DIR="./log/separate_dict/$FEAT*"
      python src/separate.py --dynet-seed 670352551 --dynet-mem 2048 \
      $CEMB_SIZE $WEMBED_SIZE $DEMBED_SIZE $HIDDEN_SIZE $N_LAYER $MLP_SIZE \
      --mode test --train $TRAIN --dev $DEV --test $TEST \
      --model_dir $MODEL_DIR --feature $FEAT --save_conll --use_dict \
      --tag_dict $DICT
    fi
  done
fi
