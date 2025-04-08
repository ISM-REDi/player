import sys
# import numpy as np
import pandas as pd
import time
import datetime as dt
# import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE, MDS
import plotly.express as px

def coloring_group(colorling_df, max, color_prop, df, cluster_type_col):
  for i in range(0, max):
    targets = df[df[cluster_type_col] == i]
    value = lambda x: x.pop() if len(x) == 1 else '&'.join(x)
    try:
      if color_prop == "s2FieldsOfStudy":
        field_list = targets[color_prop].values[0]
        colorling_df.loc[i,'field'] = field_list[0]
        colorling_df.loc[i, color_prop] = value(set(field_list))
      else:
        print(targets[color_prop])
        colorling_df.loc[i, color_prop] = value(set(targets[color_prop].values.tolist()))
    except:
      print('skip cluster; ', i)
  return colorling_df

def main():

  start = time.time()
  point_file = sys.argv[1]
  coloring_file = sys.argv[2]
  color_col = sys.argv[3]
  cluster_type_col = sys.argv[4]
  
  # coloring infomations.
  coloring_info = pd.read_csv(coloring_file, header=0)
  coloring_info.sort_values(by=cluster_type_col, inplace=True)
  coloring_df = coloring_info.drop_duplicates(subset=cluster_type_col)
  coloring_df.reset_index(inplace=True)
  coloring_df = coloring_group(coloring_df, len(coloring_df), color_col, coloring_info,cluster_type_col)
  print(coloring_df)
  # points
  df = pd.read_parquet(point_file)
  df_plot = df.copy()
  df_plot['color_label'] = coloring_df[color_col]
  # df_plot['cluster_no'] = coloring_df['cluster_no']
  df_plot[cluster_type_col] = coloring_df[cluster_type_col].apply(lambda x : f'Cluster{x:03}')
  # df_plot = pd.merge([coloring_df, df])
  df_plot['color_label'] = df_plot['color_label'].apply(lambda x : str(x))#color_col[0:2]+'_'+
  # df_plot['color'] = coloring_df.loc[:, 'color']
  # sort_order = ['情報研','遺伝研','統数研','極地研']
  # sort_order = ['mirai201603','mirai201604','mirai201703','mirai201801', 'mirai201802', 'mirai201806']
  sort_order = ['kernel','transformer_a','transformer_b','covid-19', 'smartcity', 'fakenews', 'green-economy']
  # sort_order =  ["Environmental Science"
  #   ,"Biology"
  #   ,"Medicine"
  #   ,"Mathematics"
  #   ,"Computer Science"
  #   ,"Economics"
  #   ,"Psychology"
  #   ,"Engineering"
  #   ,"Agricultural and Food Sciences"
  #   ,"Business"
  #   ,"Physics"
  #   ,"Education"
  #   ,"Geology"
  #   ,"Chemistry"
  #   ,"Materials Science"
  #   ,"Political Science"
  #   ,"Art"]
  # df_plot['color_label'] = pd.Categorical(df_plot['color_label'], categories=sort_order)
  # df_plot.sort_values(by=['color_label'], inplace=True)
  df_plot.sort_values(by=cluster_type_col, inplace=True)
  df_plot = df_plot[df_plot['color_label'] != 'nan']
  print(df_plot)
  # df_plot['shape_label'] = coloring_df.loc[:, 'shape_label']
  # df_plot['thema'] = coloring_df.loc[:, 'thema']

  fig = px.scatter_3d(df_plot, x='x', y='y', z='z',
    color=cluster_type_col,
    text ='color_label',
    # symbol='shape_label',
    hover_data=[cluster_type_col],
    opacity=0.7)

  # tight layout
  filename = point_file.split('.')[0].replace('t-SNE_3d_points', 'scatter_3d_plot_with_plotly')
  fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
  fig.write_html(filename+'_'+color_col+'.html')

  elapsed_time = time.time() - start
  print('elapsed_time [sec]: ' + str(elapsed_time))
  print('elapsed_time [hms]: ' + str(dt.timedelta(seconds=elapsed_time)))

  return 0

if __name__ == '__main__':
    main()