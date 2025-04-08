import pandas as pd
import argparse

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
    help='左'
    )
  parser.add_argument(
    '-a',
    '--append',
    required=False,
    help='右'
    )
  args = parser.parse_args()

  result_filelabel = 'join_results.parquet'

  df_source = pd.read_parquet(args.source)
  df_append = pd.read_parquet(args.append)
  df_append["corpusId"] = df_append["corpusId"].astype("int64")
  extracted_df = df_append.loc[:, ["corpusId", "authors", "author_group_id", "author_ids", "author_names"]]
  print(df_source.info())
  print('----')
  print(extracted_df.info())
  print('----')


  df = pd.merge(df_source, extracted_df, on="corpusId")
  # df = pd.merge(df, df_input, left_on="parentId", right_on=args.id)

  print(df.head())
  authors = []
  for idx, row in df.iterrows():
     authors.append(row["author_ids"].tolist())
  flatten = [x for row in authors for x in row]
  dropduplicate = list(set(flatten))
  # print(len(flatten))
  print("", len(dropduplicate))
  df.to_parquet(args.savedir+result_filelabel)
 
  return 0

if __name__ == '__main__':
    main()