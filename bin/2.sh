#!/bin/sh
LOG="./$(date +"%Y%m%d").log"
PARQUET_DETAIL="datas/akaike/akaike_aic_merged.parquet"
RANDOM_SIZE=100
LABEL="akaike_random_"$RANDOM_SIZE
SAVE_DIR="results/exp20241116/"
CLUSTER_N=93

#echo "Start 3 ----"
#poetry run python src/player/3_divergence.py $SAVE_DIR"3/" $SAVE_DIR"2/gaussian_n_"$CLUSTER_N"/"$LABEL"_gram_embedding_means_full.parquet" $SAVE_DIR"2/gaussian_n_"$CLUSTER_N"/" $LABEL
#echo "----"

echo "Start 4 ----"
for i in `seq 0 5`
do
  poetry run python src/player/4_interactive_3d.py $SAVE_DIR"4/" $SAVE_DIR"3/"$LABEL"_wasserstein_distance.csv" $LABEL
  if [ $i -lt 5 ]; then
    echo "wait; 60 sec."
    sleep 60
  else
    :
  fi
done
echo "----"

echo "Start 5 ----"
  poetry run python src/player/5_coloring_csv.py -save $SAVE_DIR"4/" -a $PARQUET_DETAIL -s $SAVE_DIR"2/"$LABEL"_gram_embedding_GaussianMixture_cluster_n_"$CLUSTER_N"_clusters.csv" -l $LABEL -id "corpusId"
echo "----"

