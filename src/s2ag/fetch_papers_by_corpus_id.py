import pandas as pd
import requests
import time
from datetime import datetime as dt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from config import config

# Sample
# poetry run python src/s2ag/fetch_papers_by_corpus_id.py datas/ datas/aic_corpusId.csv AIC

data_dir = sys.argv[1]
corpusid_file = sys.argv[2]
data_label = sys.argv[3]

fields_full = '&fields=title,journal,year,publicationTypes,s2FieldsOfStudy,authors,embedding'
fields_only_emb = '&fields=embedding'

def fieldValues(d):
  tmp_field = []
  tmp_s2_field = []
  try :
    tmp_s2_fields = d.get('s2FieldsOfStudy', '')
    for field in tmp_s2_fields:
      try:
        s = field.get('source', '')
        f = field.get('category', '')
        if s == 's2-fos-model':
          tmp_s2_field.append(f)
        else:
          tmp_field.append(f)
      except:
        continue
  except:
    tmp_field = ''
    tmp_s2_field = ''
  return tmp_field, tmp_s2_field

def authorValues(d):
  tmp_author_id = []
  tmp_name = []
  try :
    tmp_authors = d.get('authors', '')
    for author in tmp_authors:
      try:
        aid = author.get('authorId', '')
        name = author.get('name', '')
        tmp_author_id.append(aid)
        tmp_name.append(name)
      except:
        continue
  except:
    tmp_author_id = ''
    tmp_name = ''
  return tmp_author_id, tmp_name

def fetch(corpusId, full_flag):
  fields = ''
  if full_flag:
    fields = fields_full
  else:
    fields = fields_only_emb

  apitext = f'https://api.semanticscholar.org/graph/v1/paper/CorpusId:'+str(corpusId)+'?'+fields
  print(apitext)
  d = requests.get(apitext, headers=config.headers, timeout=(3.0, 7.0)).json()
  tmp_field, tmp_s2_field = fieldValues(d)
  tmp_author_id, tmp_name = authorValues(d)

  record = {}
  if full_flag:
    record = {
      'paperId': d.get('paperId'),
      'title': d.get('title'),
      'corpusId': corpusId,
      'journal': d.get('journal'),
      'year': d.get('year'),
      'publicationTypes': d.get('publicationTypes'),
      's2FieldsOfStudy': tmp_s2_field,
      'authors': tmp_name,
      'author_ids': tmp_author_id,
      'embedding': d.get('embedding'),
    }
  else:
    record = {
      'paperId': d.get('paperId'),
      'corpusId': corpusId,
      'embedding': d.get('embedding'),
    }
  return record

def output_filename(input, ext):
  root_ext_pair = os.path.splitext(input)
  return root_ext_pair[0] + ext

def main():
  now = dt.now()
  timestamp = now.strftime('%Y%m%d%H%M%S')
  df = pd.read_csv(corpusid_file, header=0)
  l = []
  
  for i, row in df.iterrows():
    record = fetch(row["corpusId"], True)
    record["tag"] = row["tag"]
    l.append(record)
    time.sleep(5)
  output_df =  pd.DataFrame(l)
  output_df.to_parquet(data_dir+'fetch_corpusId-embedding_'+data_label+'_'+timestamp+'.parquet')
  print("End.")

if __name__ == '__main__':
    main()
