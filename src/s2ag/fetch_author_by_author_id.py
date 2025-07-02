import numpy as np
import pandas as pd
import requests
#import time
from datetime import datetime as dt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from config import config

# Sample
# poetry run python src/s2ag/fetch_s2ag_from_author_id.py datas/aic/ Akaike 2737945 AIC

data_dir = sys.argv[1]
author_name = sys.argv[2]
author_id = sys.argv[3]
tag = sys.argv[4]

def fetch(output_df, timestamp, offset):
  fields = '&fields=title,externalIds,journal,year,publicationTypes,s2FieldsOfStudy'
  apitext = f'https://api.semanticscholar.org/graph/v1/author/{author_id}/papers?{fields}&offset={offset}'
  print(apitext)
  response = requests.get(apitext, headers=config.headers, timeout=(3.0, 7.0)).json()
  for d in response.get('data'):
    data = {
      'author_label': [author_name],
      'tag': [tag],
      'paperId': [d.get('paperId')],
      'title': [d.get('title')],
      'corpusId': [d.get('externalIds').get('CorpusId')],
      'journal': [d.get('journal')],
      'year': [d.get('year')],
      'publicationTypes': [d.get('publicationTypes')],
      's2FieldsOfStudy': [d.get('s2FieldsOfStudy')],
      }
    output_df = pd.concat([output_df, pd.DataFrame(data)], ignore_index=True)
  output_df.to_parquet((data_dir+'fetch_author_%s_%s_'+timestamp+'.parquet') % (author_name, author_id))
  try:
    next_offset = int(response.get('next'))
    fetch(output_df, timestamp, next_offset)
  except:
    next_offset = - 1
    # print('the response doesnot have next,', response)
    print('last offset number = ', offset)

def main():
  now = dt.now()
  timestamp = now.strftime('%Y%m%d%H%M%S')
  output_df = pd.DataFrame(index=[])
  
  fetch(output_df, timestamp, 0)
  output_df.to_csv((data_dir+'fetch_author_%s_%s_'+timestamp+'.csv') % (author_name, author_id))

  print("End.")


if __name__ == '__main__':
    main()
