import sys, os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from collections import Counter

def main():
  result_csv = sys.argv[1]
  label = sys.argv[2]
  
  df = pd.read_csv(result_csv, header=None, names=['corpusId', 'cluster_no'])
  
  sixtopic_corpus_ids = [1625830,9989562,38553870,249151871,259936734,231573431,204838007,219560318,213186544,106892734,110580067,32730475,4410672,157458390,153968507,234862347]
  sixtopic_group_true = [10,10,10,11,11,12,12,13,13,14,14,15,15,16,16,16]

  df_true = pd.DataFrame(
    {
    'corpusId': sixtopic_corpus_ids,
    'topic_group_no': sixtopic_group_true
  }, index=sixtopic_corpus_ids
  )

  df_eval = pd.merge(df, df_true, on='corpusId')
  tmp_df = df_eval.groupby('topic_group_no')['cluster_no'].agg(list).reset_index()
  print('Clustering results: ', tmp_df)
  for row in tmp_df.iterrows():
    # replace cluster number to group number.
    counter = Counter(row[1]['cluster_no'])#
    mode_no = counter.most_common(1)[0][0]
    df_eval.loc[df_eval['cluster_no'] == mode_no, 'cluster_no'] = row[1]['topic_group_no']
  
  # Assign the missing cluster number to the missing group number for replacement.
  unique_in_left = list(set(df_eval['topic_group_no'].to_list()) - set(df_eval['cluster_no'].to_list()))
  if len(unique_in_left) == 1:
    missing_cluster = df_eval[df_eval['cluster_no'] < 10]['cluster_no'].values[-1]
    df_eval.loc[df_eval['cluster_no'] == missing_cluster, 'cluster_no'] = unique_in_left[0]
  elif len(unique_in_left) > 0:
    print('error.')
  else :
    print('skip, nothing to do.')

  print('Labeled results: ', df_eval.groupby('topic_group_no')['cluster_no'].agg(list).reset_index())

  report = classification_report(df_eval['topic_group_no'], df_eval['cluster_no'], output_dict=True)
  print("The best classification_report '", label, "'")
  print(f"Accuracy       : {report['accuracy']:.2f}", )
  print(f"Macro F1 score : {report['macro avg']['f1-score']:.2f}")
  print(f"Weighted F1    : {report['weighted avg']['f1-score']:.2f}")

if __name__ == '__main__':
  main()

