#!/bin/sh
LOG="./$(date +"%Y%m%d").log"
PARQUET_EMBEDDING="datas/akaike/akaike_aic_dataset.parquet"
RANDOM_SIZE=100
LABEL="akaike_random_"$RANDOM_SIZE
SAVE_DIR="results/exp_2_20241116/"
MIN_CLUSTER_N=60
str1=0
str2=0
str3=0

echo 'Please select the step number
1) Feature extraction
2) Clustering by PLAYER
3) Visualization
'
echo -n "Step :"
read STEP

echo 'params'
case $STEP in
  1)
    echo -n INPUT_STR 1: 
    read str1
    #$ poetry run python src/player/_1_gram_matrix.py results/pickup/1/ datas/pickup/fetch_corpusId-embedding_20241024163626.parquet sixtopic -1 CorpusId
    ;;
  2)
    echo -n INPUT_STR 2: 
    read str2
    #2_bayesian_optm.py
    #3_divergence.py
    ;;
  3)
    echo -n INPUT_STR 3:
    read str3
    # src/visual/interactive_3d.py
    ;;
esac
echo $str1 $str2 $str3

