import argparse
import pandas as pd
import time
import requests
import utils.s2ag as s2ag
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from config import config

l = []

fields_full = '&fields=title,journal,year,publicationTypes,s2FieldsOfStudy,authors,embedding'

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

def fetch(corpusId, isCite, offset, skip_count):
  try:
    if isCite:
      datas, next_offset = fetch_citations(corpusId, offset)
      fieldName = 'citingPaper'
    else:
      datas, next_offset = s2ag.fetch_references(corpusId, offset)
      fieldName = 'citedPaper'
  except requests.exceptions.RequestException as e:
    print("requests.get error at citations or references:",e)
    print("skip offset ", offset)
    if skip_count > 20:
      print('limit over.')
      return pd.DataFrame(index=[])
    # retry
    fetch(corpusId, isCite, offset+1, skip_count+1)
    return pd.DataFrame(index=[])
  except:
    print('error')
    return pd.DataFrame(index=[])

  for d in datas:
    try:
      if d.get(fieldName) == None: continue
      paper = d.get(fieldName).get('corpusId')
      record = allFetch(paper, corpusId)
      time.sleep(10)
      if not record is None:
        l.append(record)
    except requests.exceptions.RequestException as e:
      print("requests.get error at embeddings:",e)
      continue
  if next_offset > 0 and (next_offset + 100) < 10000:
    fetch(corpusId, isCite, next_offset, skip_count)

def allFetch(corpusId, parentId):
  fields = ''
  fields = fields_full

  apitext = f'https://api.semanticscholar.org/graph/v1/paper/CorpusId:'+str(corpusId)+'?'+fields
  print(apitext)
  d = requests.get(apitext, headers=config.headers, timeout=(3.0, 7.0)).json()
  tmp_field, tmp_s2_field = fieldValues(d)
  tmp_author_id, tmp_name = authorValues(d)

  record = {
      'paperId': d.get('paperId'),
      'title': d.get('title'),
      'CorpusId': corpusId,
      'parentId': parentId,
      'journal': d.get('journal'),
      'year': d.get('year'),
      'publicationTypes': d.get('publicationTypes'),
      's2FieldsOfStudy': tmp_s2_field,
      'authors': tmp_name,
      'author_ids': tmp_author_id,
      'embedding': d.get('embedding'),
  }
  return record

def fetch_citations(corpusId, offset):
  apitext = f'https://api.semanticscholar.org/graph/v1/paper/CorpusId:'+str(corpusId)+'/citations?fields=corpusId&offset='+str(offset)
  print(apitext)
  response = requests.get(apitext, headers=config.headers, timeout=(3.0, 7.0)).json()
  #list of embedings by one corpusId
  citations = response.get('data')
  try:
    next_offset = int(response.get('next'))
  except:
    next_offset = - 1
    print('the response doesnot have next,', response)
    print('last offset number = ', offset)
  return citations, next_offset

def run(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--input',
    dest='input',
    required=True,
    help='input file')
  parser.add_argument(
    '--cite',
    dest='cite',
    required=True,
    help='cite or ref')
  known_args, _ = parser.parse_known_args(argv)
  # read courpusid list
  if known_args.input.endswith("csv"):
    print("csv")
    df = pd.read_csv(known_args.input)
  elif known_args.input.endswith("parquet"):
    print("parquet")
    df = pd.read_parquet(known_args.input)

  output_df = pd.DataFrame(index=[])

  isCite = (known_args.cite == 'cite')
  fileTerm = "-citations.parquet" if isCite else "-references.parquet"

  for row in df['CorpusId']:
    #list of embedings by one corpusId
    fetch(row, isCite, 0, 0)# !!! CHECK !!!
    time.sleep(10)
  
  output_df =  pd.DataFrame(l)
  print(output_df)

  output_df.to_parquet(s2ag.output_filename(known_args.input, fileTerm))

if __name__ == '__main__':
  run()
