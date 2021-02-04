#!/bin/sh
DATETIME=`date +%Y%m%d_%H%M%S`
SAVE_DIR="./outputs/run/${DATETIME}"

echo $SAVE_DIR
mkdir -p $SAVE_DIR

echo "Preprocessing"
python preprocessing.py save_dir=$SAVE_DIR current_time=$DATETIME $*

echo "Learning"
python learning.py save_dir=$SAVE_DIR current_time=$DATETIME $*

echo "Predicting"
python predicting.py save_dir=$SAVE_DIR current_time=$DATETIME $*

echo "END"
echo $SAVE_DIR
