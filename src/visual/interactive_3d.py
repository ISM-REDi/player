import sys
import numpy as np
import pandas as pd
import time
import datetime as dt
# import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE, MDS
import plotly.express as px


def test(idx, tmp, gram_arr):
  if len(tmp) > 0:
    if idx == 0 or len(gram_arr) == 0:
      gram_arr = tmp
    else:
      gram_arr = np.vstack([gram_arr, tmp])
  return gram_arr

def main():
  # $ poetry run python src/4_interactive_3d.py result/20231208_67-2/ result/20231208_67-2/refs-67-2-embeding_gram_mixed_means_full_wasserstein_distance.csv false
  start = time.time()
  save_dir = sys.argv[1]
  distances_file = sys.argv[2]
  label = sys.argv[3]
  input_datatype = sys.argv[4]
  #emb_flag = bool(sys.argv[4])

  now = dt.datetime.now()
  timestamp = now.strftime('%Y%m%d%H%M%S')
  gram_arr = np.empty(0)
  
  # Read files.
  try:
    df = pd.read_csv(distances_file, header=None)
    df.fillna(0, inplace=True)
    # to ndarray
    for idx, row in df.iterrows():
      x = row.to_numpy()
      tmp = x
      gram_arr = test(idx, tmp, gram_arr)
    size, dim = df.shape
  except:
    df = pd.read_parquet(distances_file)
    df = df['embedding'].apply(lambda x: x['vector'])
    for idx, row in df.items():
      gram_arr = test(idx, row, gram_arr)
    size = len(df)
  
  # print(df.info())
  # print(df.head())

  
  # Dimensionality reduction
  X_gram = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(gram_arr)
  # X_gram = MDS(n_components=2, normalized_stress='auto').fit_transform(gram_arr)
  cluster_list = [ 'ID%03d' % no for no in range(0, size)]
  df_plot = pd.DataFrame({'ID': cluster_list})
  df_plot['x'] = X_gram.T[0]
  df_plot['y'] = X_gram.T[1]
  df_plot['z'] = X_gram.T[2]
  fig = px.scatter_3d(df_plot, x='x', y='y', z='z',
    color='ID',
    text ='ID',
    opacity=0.7)

  # tight layout
  fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
  fig.write_html(save_dir+label+'_'+input_datatype+'_scatter_3d_plot_with_plotly_'+timestamp+'.html')# TODO rename
  df_plot.to_parquet(save_dir+label+'_'+input_datatype+'_t-SNE_3d_points_'+timestamp+'.parquet')

  elapsed_time = time.time() - start
  print('elapsed_time [sec]: ' + str(elapsed_time))
  print('elapsed_time [hms]: ' + str(dt.timedelta(seconds=elapsed_time)))

  return 0

if __name__ == '__main__':
    main()