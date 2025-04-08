import pandas as pd
import time
import datetime as dt
import argparse

def coloring_group(coloring_props, colorling_df, i, row, df):
  targets = df[df['cluster_no'] == row['cluster_no']]
  value = lambda x: x.pop() if len(x) == 1 else '+'.join(x)
  try:
    for prop in coloring_props:
      if prop == "s2FieldsOfStudy":
        field_list = targets[prop].values[0]
        colorling_df.loc[i,'field'] = field_list[0]
        colorling_df.loc[i,prop] = value(set(field_list))
      else:
        colorling_df.loc[i,prop] = value(set(targets[prop].values.tolist()))
  except:
    print('skip cluster ', i)
  return colorling_df

def main():
  parser = argparse.ArgumentParser(description='クラスタの色分けCSV')
  parser.add_argument(
    '-save',
    '--savedir',
    required=True,
    help='保存先ディレクトリ'
    )
  parser.add_argument(
    '-s',
    '--source',
    required=True,
    help='クラスタリング結果'
    )
  parser.add_argument(
    '-a',
    '--append',
    required=False,
    help='書誌詳細'
    )
  parser.add_argument(
    '-i',
    '--input',
    required=False,
    help='入力CorpusIdファイル'
    )
  parser.add_argument(
    '-l',
    '--label',
    # default='',
    help='ファイルに付与するラベル'
    )
  parser.add_argument(
    '-id',
    '--id',
    # default='',
    help='ID列名'
    )
  args = parser.parse_args()

  start = time.time()

  coloring_filelabel = args.label+'_coloring_set.csv'
  result_filelabel = args.label+'_results'
  #coloring_props = ['tag', 'parentId', 's2FieldsOfStudy']
  #coloring_props = ['tag', 's2FieldsOfStudy']
  coloring_props = ['journal', 's2FieldsOfStudy']

  df_source = pd.read_csv(args.source, index_col=False, names=[args.id, "cluster_no"])
  df_append = pd.read_parquet(args.append)
  df_append["CorpusId"] = df_append["CorpusId"].astype("int64")
  # df_append["corpusId"] = df_append["corpusId"].astype("int64")
  # df_input = pd.read_csv(args.input)
  
  print(df_source.info())
  print('----')
  print(df_append.info())
  print('----')
  # print(df_input.info())

  colorling_ser = df_source['cluster_no']
  colorling_ser.drop_duplicates(inplace=True)
  colorling_ser.reindex()
  colorling_df = pd.DataFrame(data=colorling_ser)

  df = pd.merge(df_source, df_append, on=args.id)
  # df = pd.merge(df, df_input, left_on="parentId", right_on=args.id)

  print(df.head())

  for prop in coloring_props:
    colorling_df[prop] = ''
  # TODO FIX
  colorling_df['field'] = ''

  for i, row in colorling_df.iterrows():
    # print(row)
    colorling_df = coloring_group(coloring_props, colorling_df, i, row, df)

  colorling_df.sort_values(by=['cluster_no'], inplace=True)
  colorling_df.to_csv(args.savedir+coloring_filelabel, index=False)
  df.to_csv(args.savedir+result_filelabel+'.csv', index=False)
  df.to_parquet(args.savedir+result_filelabel+'.parquet')

  elapsed_time = time.time() - start
  print('elapsed_time [sec]: ' + str(elapsed_time))
  print('elapsed_time [hms]: ' + str(dt.timedelta(seconds=elapsed_time)))
  
  return 0

if __name__ == '__main__':
    main()