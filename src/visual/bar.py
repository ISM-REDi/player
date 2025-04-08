import argparse
import pandas as pd
import numpy as np
import time
import datetime as dt
import requests
import random
import collections
import matplotlib.pyplot as plt

def main(argv=None):
  start = time.time()

  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-save',
    '--savedir',
    dest='savedir',
    required=True,
    help='save directory')
  parser.add_argument(
    '-d',
    '--data',
    dest='data',
    required=True,
    help='dataset file')
  parser.add_argument(
    '-l',
    '--label',
    dest='label',
    required=True,
    help='dataset label')

  known_args, rest_args = parser.parse_known_args(argv)
  now = dt.datetime.now()
  timestamp = now.strftime('%Y%m%d%H%M%S')

  df = pd.read_parquet(known_args.data)
  print(df.head(2))

  # 年別棒グラフ
  years = df['year'].value_counts()
  sorted_years = sorted(years.items(), key=lambda x:x[0])
  fig, ax = plt.subplots()
  keys, values = zip(*sorted_years)
  bar_container = ax.bar(keys, values)
  ax.bar_label(bar_container)
  plt.xticks(keys, rotation=45)
  plt.savefig(known_args.savedir+known_args.label+'_boxplot_count_by_year.png')
  plt.close()

  # 分野別棒グラフ
  field = df['s2FieldsOfStudy'].items()
  res = np.array([])
  for idx, fi in field:
    if idx == 0:
      res = fi
    elif fi is not None:
      res = np.append(res, fi)
  c = collections.Counter(res.flatten())
  sorted_c = sorted(c.items(), key=lambda x:x[1])
  fig, ax = plt.subplots()
  keys, values = zip(*sorted_c)
  bar_container = ax.barh(keys, values)
  ax.bar_label(bar_container)
  plt.savefig(known_args.savedir+known_args.label+'_boxplot_count_by_field.png', bbox_inches='tight')
  plt.close()

  # フィールド組合せ別棒グラフ
  s2Fields = df['s2FieldsOfStudy'].value_counts()
  # sorted_s2Fields = sorted(s2Fields.items(), key=lambda x:x[1])
  s2Fields_gt_1 = s2Fields[s2Fields > 1]# TODO 
  sorted_s2Fields = sorted(s2Fields_gt_1.items(), key=lambda x:x[1])
  fig, ax = plt.subplots()# figsize=(10, 20)
  keys, values = zip(*sorted_s2Fields)
  keys = [ ' and '.join(x) for x in keys ]
  bar_container = ax.barh(keys, values)
  ax.bar_label(bar_container)
  # plt.savefig(known_args.savedir+known_args.label+'_boxplot_count_by_multifield.png', bbox_inches='tight')
  plt.savefig(known_args.savedir+known_args.label+'_boxplot_count_by_multifield_gt_1.png', bbox_inches='tight')
  plt.close()

  elapsed_time = time.time() - start
  print('elapsed_time [sec]: ' + str(elapsed_time))
  print('elapsed_time [hms]: ' + str(dt.timedelta(seconds=elapsed_time)))

  return 0

if __name__ == '__main__':
  main()
