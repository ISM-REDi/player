#!/bin/sh
LOG="./$(date +"%Y%m%d").log"
PARQUET_EMBEDDING="datas/akaike/akaike_aic_dataset.parquet"
RANDOM_SIZE=100
LABEL="akaike_random_"$RANDOM_SIZE
SAVE_DIR="results/exp_2_20241116/"
MIN_CLUSTER_N=60

echo "Start 1 ----"
poetry run python src/player/1_gram_matrix.py $SAVE_DIR"1/" $PARQUET_EMBEDDING $LABEL $RANDOM_SIZE corpusId
echo "----"

echo "Start 2 ----"
for i in `seq $MIN_CLUSTER_N $RANDOM_SIZE`
do
  poetry run python src/player/2_clustering_gmm.py $SAVE_DIR"2/" $SAVE_DIR"1/"$LABEL"_gram_embedding.parquet" $i
done
echo "----"

echo "Start 3 ----"
for i in `seq $MIN_CLUSTER_N $RANDOM_SIZE`
do
  poetry run python src/player/eval_player.py $PARQUET_EMBEDDING $SAVE_DIR"2/"$LABEL"_gram_embedding_GaussianMixture_cluster_n_"$i"_clusters.csv" $i
done
echo "----"

# echo "Start 3 ----"
# poetry run python src/player/3_divergence.py $SAVE_DIR"3/" $SAVE_DIR2"/gaussian_n_"$CLUSTER_N"/"$LABEL"_gram_embedding_means_full.parquet" $SAVE_DIR"2/gaussian_n_"$CLUSTER_N"/" $LABEL
# echo "----"

# echo "Start 4 ----"
# poetry run python src/player/4_interactive_3d.py $SAVE_DIR"4/" $SAVE_DIR3"/"$LABEL"_wasserstein_distance.csv" $LABEL
# echo "----"

# echo "Start 5 ----"
# poetry run python src/player/5_coloring_csv.py -save $SAVE_DIR"4/" -a $PARQUET_EMBEDDING -s $SAVE_DIR"2/"$LABEL"_gram_embedding_GaussianMixture_cluster_n_"$CLUSTER_N"_clusters.csv" -l $LABEL
# echo "----"

# echo "Start 6 ----"
# poetry run python src/player/6_interactive_3d_coloring.py $SAVE_DIR4/$LABEL_t-SNE_3d_points_202409301636.parquet $SAVE_DIR4/$LABEL_coloring_set.csv tag
# echo "----"

