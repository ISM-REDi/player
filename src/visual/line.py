import argparse
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils/'))
from player import *

def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-i',
    '--input',
    dest='input',
    required=True,
    help='input file')
  parser.add_argument(
    '-c',
    '--condition',
    dest='cond',
    required=True,
    help='condition')
  known_args, _ = parser.parse_known_args(argv)

  df = pd.read_parquet(known_args.input)
  # output_df = pd.DataFrame(index=[])

  fig = plt.figure(figsize=(25, 10))
  ax = fig.add_subplot(1,1,1)
  x = series_to_ndarray(df['scores'])
  y_mean = np.mean(x, axis=0)
  ax.plot(y_mean, linestyle='solid')
  yerr_sd = np.array(x.std(axis=0))
  yerr_se = np.array(yerr_sd / 10)
  ax.errorbar(range(0,100), y_mean, yerr=yerr_se, capsize=3, fmt='o', ecolor='k', ms=7, mfc='None', mec='k')
  ax.set_xlabel('trials', size=20)
  ax.set_ylabel('player score', size=20)
  ax.set_ylim(0, 500)
  #ax.set_xlim(-1, 100)
  plt.tick_params(labelsize=15)
  ax.grid(True)
  fig.savefig('./results/rand_100_trials_optim-100_10_times_ave_ylim500.png')

  return 0

if __name__ == '__main__':
  main()