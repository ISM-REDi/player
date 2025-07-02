import argparse
import pandas as pd
import pyarrow.parquet as pq
import time
import requests
import utils.s2ag as s2ag

# Sample
# poetry run python src/s2ag/fetch_recommendations.py --input datas/inputcorpusId.parquet

l = []

def fetch_reccomends(corpusId):
  apitext = f'https://api.semanticscholar.org/recommendations/v1/papers/forpaper/CorpusId:'+str(corpusId)+'?fields=corpusId&limit=10'#&from=all-cs
  print(apitext)
  response = requests.get(apitext, headers=s2ag.headers, timeout=(3.0, 7.0)).json()
  #list of embedings by one corpusId
  reccomends = response.get('recommendedPapers')
  return reccomends


def fetch(corpusId):
  try:
    reccomends = fetch_reccomends(corpusId)
    for d in reccomends:
      if d.get('corpusId') == None: continue
      reccomendPaper = d.get('corpusId')
      record = s2ag.fetch(reccomendPaper, corpusId)
      if not record is None:
        record["parentId"] = corpusId
        l.append(record)
      time.sleep(3)
  except requests.exceptions.RequestException as e:
    print("requests.get error:",e)
    return pd.DataFrame(index=[])

def run(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--input',
    dest='input',
    required=True,
    help='input file')
  known_args, _ = parser.parse_known_args(argv)
  df = pd.read_parquet(known_args.input)
  output_df = pd.DataFrame(index=[])

  for row in df['corpusId']:
    fetch(row)
    time.sleep(3)
  output_df =  pd.DataFrame(l)

  print(output_df)

  output_df.to_parquet(s2ag.output_filename(known_args.input, "-recommend.parquet"))

if __name__ == '__main__':
  run()