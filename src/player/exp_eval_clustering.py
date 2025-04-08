import sys,os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report

def main():

  header = ["player","no-kernel","kmeans","spectral","TRUE"]
  corpus_ids = [1625830,9989562,38553870,249151871,259936734,231573431,204838007,219560318,213186544,106892734,110580067,32730475,4410672,157458390,153968507,234862347]
  clusters = [[0,0,2,3,0],
  [0,0,0,0,0],
  [0,0,0,4,0],
  [1,1,1,1,1],
  [1,1,1,0,1],
  [2,2,1,1,2],
  [2,2,1,2,2],
  [3,3,3,0,3],
  [3,3,3,6,3],
  [4,4,4,4,4],
  [4,4,4,6,4],
  [5,5,5,5,5],
  [5,5,5,2,5],
  [6,6,6,6,6],
  [6,6,6,6,6],
  [6,6,6,6,6]]

  df = pd.DataFrame(clusters, columns=header)
  df.index = corpus_ids
  # print(df)
  
  print("Report Cluistering : PLAYER")
  print(classification_report(df['TRUE'], df['player']))
  print("Report Cluistering : No Kernel")
  print(classification_report(df['TRUE'], df['no-kernel']))
  print("Report Cluistering : K Means")
  print(classification_report(df['TRUE'], df['kmeans']))
  print("Report Cluistering : Spectral Clustering")
  print(classification_report(df['TRUE'], df['spectral']))

if __name__ == '__main__':
  main()

