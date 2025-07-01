import optuna
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

import itertools
import datetime
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils/'))
from player import *
from evaluate import *

import subprocess

score_arr = []
covariance_type = 'full'#'diag'

def clusterLevenshteinDistance(cluster_df):
  cluster_distance_total = 0
  for v in itertools.combinations(cluster_df.index.values, 2):
    i = list(v)
    first = cluster_df.loc[i[0], 'author_ids'].tolist()
    second = cluster_df.loc[i[1], 'author_ids'].tolist()
    tmp_score = authorsLevenshteinDistance(first, second)
    # print("Levenshtein Distance = {}".format(tmp_score))
    cluster_distance_total += tmp_score
  return cluster_distance_total

def clusterJaccardDistance(cluster_df):
  cluster_distance_total = 0
  for v in itertools.combinations(cluster_df.index.values, 2):
    i = list(v)
    first = cluster_df.loc[i[0], 'author_ids'].tolist()
    second = cluster_df.loc[i[1], 'author_ids'].tolist()
    tmp_score = caseJaccardDistance(first, second)
    # print("Jaccard Distance = {}".format(tmp_score))
    cluster_distance_total += tmp_score
  return cluster_distance_total

def calc(cluster_label, detail_file, df_keys, keycolname, experimentFlag):
  df_data = pd.read_parquet(detail_file)
  df_keys["cluster_no"] = cluster_label
  df = pd.merge(df_keys, df_data, how="left")
  df.dropna(subset=keycolname, inplace=True)
  cluster_count_df = df.groupby("cluster_no").count()
  target_clusters = cluster_count_df[cluster_count_df[keycolname] > 1].index.values
  # print("targets; ", target_clusters)
  sum_distance = 0
  cluster_distance_total = 0
  for cluster_no in target_clusters:
    if experimentFlag:
      cluster_distance_total = clusterJaccardDistance(df[df["cluster_no"] == cluster_no])
      # print("cluster {} Jaccard Distance total = {}".format(cluster_no, cluster_distance_total))
    else:
      cluster_distance_total = clusterLevenshteinDistance(df[df["cluster_no"] == cluster_no])
    # print("cluster {} Levenshtein Distance total = {}".format(cluster_no, cluster_distance_total))
    sum_distance += cluster_distance_total
  # Average
  return sum_distance / len(target_clusters)

def create_objective(gram_a, gram_b, detail_file, df_keys, keycolname):
  def objective(trial):
    weight_dist = trial.suggest_float('Wv', 0, 1, step=0.1)
    weight_diff = trial.suggest_int('Wg', 1, 10)
    X_e = df_to_ndarray(gram_a)
    X_a = df_to_ndarray(gram_b)
    K_linear = X_e
    K_diffusion = X_a
    # 合成カーネル行列
    K_combined = weight_dist * K_linear + weight_diff * K_diffusion
    m = trial.suggest_int('K', 2, 99)# TODO
    
    # Gaussian mixtures
    clustering_gmm = GaussianMixture(n_components=m, covariance_type=covariance_type, random_state=19)
    label = clustering_gmm.fit_predict(K_combined)
    # print('AIC = {}'.format(clustering_gmm.aic(X)))
    # print('Log-likelihood of X = {}'.format(clustering_gmm.score(X)))
    
    # クラスタリング結果の良さを評価する
    # ver. log likelyfood
    # log_likelyfood = clustering_gmm.score(K_combined)
    # return log_likelyfood

    # ver. unit_score
    # ave_levenstine_distance = calc(label, detail_file, df_keys, keycolname, False)
    # player_score = ave_levenstine_distance
    # score_arr.append(float(format(player_score, '.3f')))
    # return player_score

    # ver. BIC
    # bic = clustering_gmm.bic(K_combined)
    # return bic
  
    # ver. Silhouette Score（高いほど良い）
    # silhouette = silhouette_score(K_combined, label)
    # return silhouette

    # ver. Davies–Bouldin Index
    # db_index = davies_bouldin_score(K_combined, label)
    # return db_index
  
    # ver. Jaccard Dist
    ave_jaccard_distance = calc(label, detail_file, df_keys, keycolname, True)
    return ave_jaccard_distance
  return objective

def main():
  start = datetime.datetime.now()

  # Settings
  param_n_trials = 100# 30
  read_filename = './datas/akaike/akaike_aic_dataset.parquet'
  result_label = 'comp_eval'
  save_dir = 'results/exp_aic/'
  # TEST
  # save_filename = './results/exp_aic/bayesian_optimization_aic_TEST.parquet'
  save_filename = './results/exp_aic/bayesian_optimization_aic.parquet'
  keycolname = 'corpusId'
  eval_func_name = 'Jaccard'# 'BIC', 'unit_score', 'Silhouette', 'Davies–Bouldin', 'Jaccard'
  file_path_e = save_dir+result_label+'_gram_coauthor.parquet'
  file_path_a = save_dir+result_label+'_gram_embedding.parquet'
  file_path_keys = save_dir+result_label+'_author_matrix_keys.parquet'

  # Create two gram matrices from the AIC dataset using a random sample.
  result = subprocess.run([
    'poetry', 'run', 'python', 'src/player/1_gram_matrix.py',
    save_dir,
    read_filename,
    result_label,
    '100',
    keycolname])
  print(result)
  
  # Bayesian optimization.
  df_e = pd.read_parquet(file_path_e)
  df_a = pd.read_parquet(file_path_a)
  df_keys = pd.read_parquet(file_path_keys)

  study = optuna.create_study()# If use Silhouette score, set direction="maximize".
  study.optimize(create_objective(df_e, df_a, read_filename, df_keys, keycolname), n_trials=param_n_trials)
  study.best_params
  record = {
    'timestamp': start,
    'path_to_result': save_filename,
    'n_trials': param_n_trials,
    'scores': study.best_params,
    'eval_func': eval_func_name
  }
  record_df = pd.DataFrame(pd.Series(record)).T
  try:
    df_save = pd.read_parquet(save_filename)
    df_save = pd.concat([df_save, record_df], ignore_index=True)
  except:
    df_save = record_df
  
  df_save.to_parquet(save_filename)

  print('end.')

  return 0

if __name__ == '__main__':
  main()