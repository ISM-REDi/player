#!/bin/sh

RANDOM_SIZE=100
LABEL="akaike_random_"$RANDOM_SIZE
SAVE_DIR="results/exp20241116/4/"
#cluster_no,journal,s2FieldsOfStudy,year
COLOR_COL=$1

echo "Start 6 ----"
for POINTS_PARQUET in `\find $SAVE_DIR -type f -name $LABEL"_t-SNE_3d_points_*.parquet"`; do
  poetry run python src/player/6_interactive_3d_coloring.py $POINTS_PARQUET $SAVE_DIR$LABEL"_coloring_set.csv" $COLOR_COL
done
echo "----"
