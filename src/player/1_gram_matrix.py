import sys
import os
import numpy as np
import pandas as pd
import time
import datetime as dt
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.ticker as ticker
import math
import seaborn as sns
import ot
from scipy import stats

#$ poetry run python src/player/_1_gram_matrix.py results/pickup/1/ datas/pickup/fetch_corpusId-embedding_20241024163626.parquet sixtopic -1 CorpusId
def generate_coauthor_matrix(df_paper, save_dir, filelabel, idrow_name):
  size, dim = df_paper.shape
  coauthor_matrix = np.zeros((size, size))
  ser_keys = df_paper[idrow_name]
  id_dict = dict(zip(df_paper[idrow_name], df_paper.index))
  for idx, corpus_id in ser_keys.items():
    try:
      edges = df_paper[df_paper[idrow_name] == corpus_id]['edges'].to_numpy()[0]
      for e in edges:
        try:
          e_0_index = id_dict[corpus_id]
          e_1_index = id_dict[e]
          coauthor_matrix[e_0_index, e_1_index] = 1
        except:
          continue
    except:
      all_author = df_paper['author_ids'].apply(lambda x: '+'.join(x))
      all_author.index = df_paper[idrow_name].values
      authors = df_paper[df_paper[idrow_name] == corpus_id]['author_ids'].values[0]
      for auth in authors:
        tmp = all_author[all_author == auth]
        if len(tmp.index) > 0:
          e_0_index = id_dict[corpus_id]
          e_1_index = id_dict[tmp.index[0]]
          coauthor_matrix[e_0_index, e_1_index] = 1

  for i, c_id in ser_keys.items():
    coauthor_matrix[i, i] = 0
    minus_d = - coauthor_matrix[i, :].sum()
    if minus_d != 0:
      coauthor_matrix[i, i] = minus_d

  col_names = [str(x) for x in list(range(0, len(df_paper)))]
  df_co_matrix = pd.DataFrame(coauthor_matrix, columns=col_names)
  df_co_matrix.to_parquet(save_dir+filelabel+'_author_matrix.parquet')
  df_co_matrix_keys = pd.DataFrame(df_paper[idrow_name].T, columns=[idrow_name])
  df_co_matrix_keys.to_parquet(save_dir+filelabel+'_author_matrix_keys.parquet')
  print(df_co_matrix.describe().T.describe())

  return coauthor_matrix

def to_ndarray(df, idrow_name, counter_dict):
  embedding_arr = np.empty(0)
  keys_arr = []
  for idx, row in df.iterrows():
    count = counter_dict[row["parentId"]]
    if count >= 5:
      continue
    else:
      counter_dict[row["parentId"]] = count + 1
    # x = row['embedding'][0]
    x = row['embedding']["vector"]
    # x = row.to_numpy()
    keys_arr.append(row[idrow_name])
    tmp = x
    if idx == 0 or len(embedding_arr) == 0:
      embedding_arr = tmp
    else:
      embedding_arr = np.vstack([embedding_arr, tmp])
  return embedding_arr, keys_arr

def to_ndarray_org(df, idrow_name):
  embedding_arr = np.empty(0)
  keys_arr = []
  for idx, row in df.iterrows():
    # x = row['embedding'][0]
    x = row['embedding']["vector"]
    # x = row.to_numpy()
    keys_arr.append(row[idrow_name])
    tmp = x
    if idx == 0 or len(embedding_arr) == 0:
      embedding_arr = tmp
    else:
      embedding_arr = np.vstack([embedding_arr, tmp])
  return embedding_arr, keys_arr

def gram_matrix_coauthor(coauthor_matrix):
  # kernel for coauthor network.
  # 対角化
  eig = np.linalg.eig(coauthor_matrix)
  # 対角行列 D
  D = np.diag(eig[0])
  # 固有ベクトル P
  P = eig[1]
  # P inverse
  iP = np.linalg.inv(P)
  # diffusion kernel
  l = 0.5
  K = iP * np.exp(l * D) * P
  K = K.astype(np.float64)
  print('---- lambda ----', l)
  print('---- K ----', K)
  return K

def gram_matrix_embedding(arr_1, filelabel):
  # kernel for embedding vector.
  # 1. Euclidean Distance
  dist_arr = ot.dist(arr_1, metric='euclidean')

  # 2. Manhattan Distance
  # dist_arr = ot.dist(arr_1, metric='cityblock')
  
  # centering matrix
  n = dist_arr.shape[0]
  J_c = 1./n*(np.eye(n) - 1 + (n-1)*np.eye(n))

  # perform double centering
  gram_embedding = -0.5*(J_c.dot(dist_arr)).dot(J_c)

  return gram_embedding

def save_parquet(dataset, col_names, save_dir, filelabel, datalabel):
  df_gram = pd.DataFrame(dataset, columns=col_names)
  df_gram.to_parquet(save_dir+filelabel+'_gram_'+datalabel+'.parquet')

def main():
  start = time.time()
  save_dir = sys.argv[1]
  input_file = sys.argv[2]
  filelabel = sys.argv[3]
  sampling_size = int(sys.argv[4])
  idrow_name = sys.argv[5]

  df = pd.read_parquet(input_file)
  df.fillna('', inplace = True)
  df.drop_duplicates(subset=[idrow_name], inplace=True)
  print(df.info())
  size, dim = df.shape
  if sampling_size > 0:
    target_df = df.sample(sampling_size)
    target_df.reset_index(inplace=True)
  else:
    target_df = df.copy()
  # Coauthor Matrix
  embedding_arr, keys_arr = to_ndarray_org(target_df, idrow_name)
  #df_matrix.index = df_matrix_keys['corpusId']
  #df_matrix.columns = df_matrix_keys['corpusId']
  # print(df_matrix.shape)
  keys_ser = target_df[idrow_name]
  keys_list = list(keys_arr)

  # sample_coauthor_matrix(target_df, sampling_size)
  coauthor_matrix = generate_coauthor_matrix(target_df, save_dir, filelabel, idrow_name)
  gram_a = gram_matrix_coauthor(coauthor_matrix)
  save_parquet(gram_a, keys_list, save_dir, filelabel, 'coauthor')

  gram_b = gram_matrix_embedding(embedding_arr, filelabel)
  save_parquet(gram_b, keys_list, save_dir, filelabel, 'embedding')

  elapsed_time = time.time() - start
  print('elapsed_time [sec]: ' + str(elapsed_time))
  print('elapsed_time [hms]: ' + str(dt.timedelta(seconds=elapsed_time)))
  
  return 0

if __name__ == '__main__':
    main()