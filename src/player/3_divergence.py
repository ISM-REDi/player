import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils/'))
from player import player
import numpy as np
import pandas as pd
import time
import datetime as dt
import ot
from sklearn.mixture import GaussianMixture

file_path_e = "./results/pickup/1/sixtopic_gram_coauthor.parquet"
file_path_a = "./results/pickup/1/sixtopic_gram_embedding.parquet"
file_path_keys = "./results/pickup/1/sixtopic_author_matrix_keys.parquet"

def wasserstein_distances(source_mean, source_cov, n, means_full, covariance_full):
  distances = []
  for i in range(0, n):
    target_mean = means_full[i]
    target_cov = covariance_full[i]
    bwd = ot.gaussian.bures_wasserstein_distance(source_mean, target_mean, source_cov, target_cov, log=False)
    distances.append(bwd)
  return np.array(distances)

def save_cluster(clusterlabel, indexlabel, save_dir, filelabel, n, methodlabel):
  D = np.stack([indexlabel.transpose(), clusterlabel.transpose()], axis=1)
  np.savetxt(save_dir+filelabel+'_%s_cluster_n_%d_clusters.csv' % (methodlabel, n), D, delimiter=',', fmt='%s,%s')

def save_results(gmm, cov_type, save_dir, filelabel, size, n):
  col_names = [str(x) for x in list(range(0, size))]
  df_weights = pd.DataFrame(gmm.weights_, columns=['weight'])
  df_weights.to_parquet(save_dir+filelabel+'_weights_%s.parquet' % (cov_type))
  df_means = pd.DataFrame(gmm.means_, columns=col_names)
  df_means.to_parquet(save_dir+filelabel+'_means_%s.parquet' % (cov_type))
  covariances = gmm.covariances_
  for i in range(0, n):
    df_covariances = pd.DataFrame(covariances[i], columns=col_names)
    df_covariances.to_parquet(save_dir+filelabel+'_covariances_%s_%d.parquet' % (cov_type, i))

def main():
  start = time.time()
  save_dir = sys.argv[1]
  wv = float(sys.argv[2])
  wg = float(sys.argv[3])
  k = int(sys.argv[4])
  # means_file
  # covariance_dir
  datalabel = sys.argv[5]
  now = dt.datetime.now()
  
  # Read files.
  df_e = pd.read_parquet(file_path_e)
  df_a = pd.read_parquet(file_path_a)
  df_keys = pd.read_parquet(file_path_keys)
  covariance_type = 'full'#'diag'
  inner_save_dir = 'gaussian_n_'+str(k)+'/'
  gaussian_results_save_dir = save_dir+inner_save_dir

  if (not os.path.exists(gaussian_results_save_dir)):
    os.mkdir(gaussian_results_save_dir)

  X_e = player.df_to_ndarray(df_e)
  X_a = player.df_to_ndarray(df_a)
  K_linear = X_e
  K_diffusion = X_a
  # 合成カーネル行列
  K_combined = wv * K_linear + wg * K_diffusion
  # Gaussian mixtures
  clustering_gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=19)
  label = clustering_gmm.fit_predict(K_combined)

  save_cluster(label, df_keys["CorpusId"].to_numpy(), save_dir, datalabel, k, 'GaussianMixture')
  save_results(clustering_gmm, covariance_type, gaussian_results_save_dir, datalabel, len(df_keys), k)

  all_covariances = clustering_gmm.covariances_
  all_means = clustering_gmm.means_

  i = 0
  ndarr = np.empty(0)

  for means in all_means:
    cov = all_covariances[i]
    wd_arr = wasserstein_distances(means, cov, k, all_means, all_covariances)
    if len(ndarr) == 0:
      ndarr = wd_arr
    else:
      ndarr = np.vstack([ndarr, wd_arr])
    i += 1
  np.savetxt(save_dir+datalabel+'_wasserstein_distance.csv', ndarr, delimiter=',', fmt='%f')

  elapsed_time = time.time() - start
  print('elapsed_time [sec]: ' + str(elapsed_time))
  print('elapsed_time [hms]: ' + str(dt.timedelta(seconds=elapsed_time)))

  return 0

if __name__ == '__main__':
    main()