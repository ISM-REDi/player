import sys,os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from scipy.spatial import Delaunay, delaunay_plot_2d, Voronoi, voronoi_plot_2d

def test(idx, tmp, dist_arr):
  if len(tmp) > 0:
    if idx == 0 or len(dist_arr) == 0:
      dist_arr = tmp
    else:
      dist_arr = np.vstack([dist_arr, tmp])
  return dist_arr

def main():
  distance_csv = sys.argv[1]
  #label = sys.argv[2]
  #input_datatype = sys.argv[3]
  
  dist_arr = np.empty(0)
  
  # Read files.
  df = pd.read_csv(distance_csv, header=None)
  df.fillna(0, inplace=True)
  # to ndarray
  for idx, row in df.iterrows():
    x = row.to_numpy()
    tmp = x
    dist_arr = test(idx, tmp, dist_arr)
  size, dim = df.shape
  
  # Dimensionality reduction
  X = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(dist_arr)

  X.T[0]#x
  X.T[1]#y

  vor = Voronoi(X)
  tri = Delaunay(X)
  fig = voronoi_plot_2d(vor)
  fig.savefig('sixtopic_voronoi_player.png')

  fig = delaunay_plot_2d(tri)
  fig.savefig('sixtopic_delaunay_player.png')


if __name__ == '__main__':
  main()

