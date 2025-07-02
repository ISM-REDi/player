#!/bin/sh

poetry run python src/player/1_gram_matrix.py $SAVE_DIR1 $PARQUET_EMBEDDING $LABEL $RANDOM_SAMPLING_LEN $INDEXCOL

poetry run python src/player/2_bayesian_optm $SAVE_DIR2 $SAVE_DIR1 $LABEL $PARQUET_EMBEDDING

poetry run python src/player/3_divergence.py $SAVE_DIR3 $LABEL
