#!/bin/bash
LOG="./$(date +"%Y%m%d").log"
. "player_settings.conf"
str1=0
str2=0
str3=0

cat_settings
echo -n "Please check settings, OK?[y/n]:"
read FLAG
case $FLAG in
  "n")
    echo 'Please edit "player_settings.conf", finish.'
    exit
    ;;
esac
echo 'Please select the step number
1) Feature extraction
2) Clustering by PLAYER
3) Visualization
'
echo -n "Step :"
read STEP

echo 'Please enter a few parameters required to run.'
case $STEP in
  1)
    echo 'start 1) Feature extraction'
    poetry run python src/player/1_gram_matrix.py $SAVE_DIR $PARQUET_EMBEDDING $LABEL -1 corpusId
    ;;
  2)
    echo 'start 2) Clustering by PLAYER'
    echo -n INPUT_STR 1: 
    read str1
    echo -n INPUT_STR 2: 
    read str2
    params=($(poetry run python src/player/2_bayesian_optm.py $SAVE_DIR $LABEL  ))
    Wv=${params[0]}
    Wg=${params[1]}
    K=${params[2]}
    poetry run python src/player/3_divergence.py $SAVE_DIR $Wv $Wg $K $LABEL
    ;;
  3)
    echo 'start 3) Visualization'
    echo -n INPUT_STR 3:
    read str3
    poetry run python src/visual/interactive_3d.py
    ;;
esac

echo 'end.'
