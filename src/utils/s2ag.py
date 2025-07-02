import datetime
import requests
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from config import config

def search(apitext):
  response = requests.get(apitext, headers=config.headers, timeout=(3.0, 7.0)).json()
  if response.get('data') == None: return None
  data = response.get('data')
  next_count = response.get('next')
  return data, next_count

def fetch_reccomends(corpusId):
  apitext = f'https://api.semanticscholar.org/recommendations/v1/papers/forpaper/CorpusId:'+str(corpusId)+'?fields=corpusId'#&limit=10&from=all-cs
  print(apitext)
  response = requests.get(apitext, headers=config.headers, timeout=(3.0, 7.0)).json()
  #list of embedings by one corpusId
  reccomends = response.get('recommendedPapers')
  return reccomends

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

def fetch_references(corpusId, offset):
  apitext = f'https://api.semanticscholar.org/graph/v1/paper/CorpusId:'+str(corpusId)+'/references?fields=corpusId&offset='+str(offset)
  print(apitext)
  response = requests.get(apitext, headers=config.headers, timeout=(3.0, 7.0)).json()
  #list of embedings by one corpusId
  references = response.get('data')
  next_offset = int(response.get('next'))
  return references, next_offset

def fetch_embedding(corpusId, parentId):
  apitext = f'https://api.semanticscholar.org/graph/v1/paper/CorpusId:'+str(corpusId)+'?fields=embedding'
  print(apitext)
  response = requests.get(apitext, headers=config.headers, timeout=(3.0, 7.0)).json()
  if response.get('embedding') == None: return None
  embedding = response.get('embedding')
  dt_now = datetime.datetime.now()
  record = {
    'event_timestamp':dt_now,
    'CorpusId': corpusId,
    'embedding' : embedding,
    'parentId': parentId,
    'created' : dt_now
  }
  return record

def output_filename(input, ext):
  root_ext_pair = os.path.splitext(input)
  return root_ext_pair[0] + ext