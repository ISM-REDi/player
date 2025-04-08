import optuna
import pandas as pd
from sklearn.mixture import GaussianMixture
import itertools
import datetime
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils/'))
from player import player
from evaluate import evaluate

file_path_e = "./results/pickup/1/sixtopic_gram_coauthor.parquet"
file_path_a = "./results/pickup/1/sixtopic_gram_embedding.parquet"
file_path_keys = "./results/pickup/1/sixtopic_author_matrix_keys.parquet"
detail_file = "./datas/pickup/fetch_corpusId-embedding_20241024163626.parquet"

df_e = pd.read_parquet(file_path_e)
df_a = pd.read_parquet(file_path_a)
df_keys = pd.read_parquet(file_path_keys)
covariance_type = 'full'#'diag'

score_arr = []

def clusterLevenshteinDistance(cluster_df):
  cluster_distance_total = 0
  for v in itertools.combinations(cluster_df.index.values, 2):
    i = list(v)
    first = cluster_df.loc[i[0], 'author_ids'].tolist()
    second = cluster_df.loc[i[1], 'author_ids'].tolist()
    tmp_score = evaluate.authorsLevenshteinDistance(first, second)
    # print("Levenshtein Distance = {}".format(tmp_score))
    cluster_distance_total += tmp_score
  return cluster_distance_total

def calc(cluster_label):
  df_data = pd.read_parquet(detail_file)
  df_keys["cluster_no"] = cluster_label
  df = pd.merge(df_keys, df_data, how="left")
  df.dropna(subset="CorpusId", inplace=True)
  cluster_count_df = df.groupby("cluster_no").count()
  target_clusters = cluster_count_df[cluster_count_df["CorpusId"] > 1].index.values
  # print("targets; ", target_clusters)
  sum_distance = 0
  for cluster_no in target_clusters:
    cluster_distance_total = clusterLevenshteinDistance(df[df["cluster_no"] == cluster_no])
    # print("cluster {} Levenshtein Distance total = {}".format(cluster_no, cluster_distance_total))
    sum_distance += cluster_distance_total
  # Average
  return sum_distance / len(target_clusters)

def objective(trial):
  weight_dist = trial.suggest_float('Wv', 0, 1, step=0.1)
  weight_diff = trial.suggest_int('Wg', 1, 10)
  X_e = player.df_to_ndarray(df_e)
  X_a = player.df_to_ndarray(df_a)
  K_linear = X_e
  K_diffusion = X_a

  # 合成カーネル行列
  K_combined = weight_dist * K_linear + weight_diff * K_diffusion

  m = trial.suggest_int('K', 1, 16 - 1)# TODO

  # Gaussian mixtures
  clustering_gmm = GaussianMixture(n_components=m, covariance_type=covariance_type, random_state=19)
  label = clustering_gmm.fit_predict(K_combined)
  # print('AIC = {}'.format(clustering_gmm.aic(X)))
  # print('Log-likelihood of X = {}'.format(clustering_gmm.score(X)))

  # クラスタリング結果の良さを評価する
  # log_likelyfood = clustering_gmm.score(K_combined)
  ave_levenstine_distance = calc(label)
  player_score = ave_levenstine_distance

  score_arr.append(float(format(player_score, '.3f')))
  return player_score

if __name__ == '__main__':
  start = datetime.datetime.now()
  param_n_trials = 30
  save_filename = './results/player_bayesian_optimization_sixtopic.parquet'
  study = optuna.create_study()#direction="maximize"
  study.optimize(objective, n_trials=param_n_trials)

  study.best_params
  record = {
    'timestamp': start,
    'path_embedding_matrix': file_path_e,
    'path_coauthor_matrix': file_path_a,
    'n_trials': param_n_trials,
    'scores': score_arr,
  }
  record_df = pd.DataFrame(pd.Series(record)).T
  try:
    df_save = pd.read_parquet(save_filename)
    df_save = pd.concat([df_save, record_df], ignore_index=True)
  except:
    df_save = record_df
  
  df_save.to_parquet(save_filename)

  print('end.')
  study.best_params['Wv'], study.best_params['Wg'], study.best_params['K']

