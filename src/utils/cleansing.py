import argparse
import pandas as pd
import time
import datetime as dt

def compare_author(name_str, groupid_author_df):
  hit = groupid_author_df[groupid_author_df['match_str_name'] == name_str]
  if len(hit) > 0:
    return hit['author_group_id'].item()
  else:
    return ''

def weight_list(df, corpus_ids, corpus_id):
  result_list = []
  weight_dc = lambda corpus_id, weight: {
    'corpusId': corpus_id,
    'weight': weight,
  }
  source_author_ids = df[df['corpusId'] == corpus_id]['author_ids'].item()
  source_authors_len = len(source_author_ids)
  for calc_corpus_id in corpus_ids:
    target_ser = df[df['corpusId'] == calc_corpus_id]
    # print(target_ser['author_ids'].item)
    and_author_ids = set(source_author_ids) & set(target_ser['author_ids'].item())
    and_len = len(and_author_ids)
    # print("and_len = ", and_len)
    weight = 0.0
    if and_len > 0:
      weight = and_len / source_authors_len
    result_list.append(weight_dc(target_ser['corpusId'].item(), weight))
  return result_list

def corpusid_author_df(authors_df):
  authors_df.fillna('', inplace = True)

  df = pd.DataFrame()
  for idx, row in authors_df.iterrows():
    t = row['event_timestamp']
    i = row['corpusId']
    authros = row['authors']
    for a in authros:
      df = pd.concat([df, pd.Series({
        'event_timestamp': t,
        'corpusId': i,
        'authorId': a['authorId'],
        'name': a['name'],
      }).to_frame().T], ignore_index=True)
  return df

def solve_author_group_id(aid, names, groupid_author_df):
  aid_str = '+'.join(aid)
  name_str = '+'.join(names)
  if not len(aid) < len(names):
    match_aid = compare_author(name_str, groupid_author_df)
  else:
    cutted = groupid_author_df[groupid_author_df['match_str_id'] == aid_str]
    match_aid = compare_author(name_str, cutted)
  return match_aid

def grouping(cid_author_df):
  group_index = 1

  make_cg_dict = lambda cid, gid: {
    'corpusId': cid,
    'author_group_id': gid
  }

  make_ga_dict = lambda ids, names: {
    'author_group_id': 'group-'+format(group_index, '04'),
    'author_ids': ids,
    'author_names': names,
    'match_str_id': '+'.join(ids),
    'match_str_name': '+'.join(names),
  }

  cid_groupid_df = pd.DataFrame(index=[])
  groupid_author_df = pd.DataFrame(index=[])

  cids = cid_author_df['corpusId'].copy()
  cids = cids.drop_duplicates()
  cids = cids.reindex()

  for idx, corpus_id in cids.items():
    authors = cid_author_df[cid_author_df['corpusId'] == corpus_id]
    ids = authors['authorId'].tolist()
    if None in ids: ids = [x for x in ids if x is not None]
    ids = sorted(ids)
    names = sorted(authors['name'].tolist())

    cg_dict = {}
    ga_dict = {}
    if idx == 0:
      ga_dict = make_ga_dict(ids, names)
      group_index += 1
      cg_dict = make_cg_dict(corpus_id, ga_dict['author_group_id'])
    else:
      author_group_id = solve_author_group_id(ids, names, groupid_author_df)
      if len(author_group_id) > 0:
        cg_dict = make_cg_dict(corpus_id, author_group_id)
      else:
        ga_dict = make_ga_dict(ids, names)
        group_index += 1
        cg_dict = make_cg_dict(corpus_id, ga_dict['author_group_id'])

    if len(cg_dict) > 0:
      cid_groupid_df = pd.concat([cid_groupid_df, pd.Series(cg_dict).to_frame().T], ignore_index=True)
    if len(ga_dict) > 0:
      groupid_author_df = pd.concat([groupid_author_df, pd.Series(ga_dict).to_frame().T], ignore_index=True)

  groupid_author_df.drop(columns=['match_str_id', 'match_str_name'], inplace=True)
  return cid_groupid_df, groupid_author_df

def add_edges(df, cid_author_df):
  df['edges'] = [[] for x in range(0, len(df))]
  df['edge_count'] = 0
  df['weights'] = [[] for x in range(0, len(df))]
  for idx, row in df.iterrows():
    authorIds = row['author_ids']
    edgelist = []
    for author_id in authorIds:
      tmp = cid_author_df[cid_author_df['authorId'] == author_id]['corpusId'].to_list()
      if tmp != None:
        edgelist = tmp
      else:
        edgelist = edgelist.append(tmp)
    edgelist = list(set(edgelist))
    try:
      edgelist.remove(row['corpusId'])
      df.at[idx, 'edges'] = edgelist
      df.at[idx, 'edge_count'] = len(edgelist)
      weightlist = weight_list(df, edgelist, row['corpusId'])
      df.at[idx, 'weights'] = weightlist
    except:
      print('skip')
  return df

def save_parquet(df, save_dir, filelabel, datatag):
  df.to_parquet(save_dir+filelabel+'_'+datatag+'.parquet')

# poetry run python src/1_cleansing.py --save datas/akaike/ -aut datas/akaike/aic_corpus_id-citations_400-author.parquet -l akaike_aic_400
def main(argv=None):
  start = time.time()

  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-save',
    '--savedir',
    dest='savedir',
    required=True,
    help='save directory')
  parser.add_argument(
    '-aut',
    '--author',
    dest='author',
    required=True,
    help='author file')
  parser.add_argument(
    '-l',
    '--label',
    dest='label',
    required=True,
    help='dataset label')

  known_args, rest_args = parser.parse_known_args(argv)

  authors_df = pd.read_parquet(known_args.author)
  authors_df.dropna(subset='authors', inplace=True)
  print(authors_df.head(2))

  cid_author_df = corpusid_author_df(authors_df)
  save_parquet(cid_author_df, known_args.savedir, known_args.label, 'corpusId_author')

  cid_groupid_df, groupid_author_df = grouping(cid_author_df)
  save_parquet(cid_groupid_df, known_args.savedir, known_args.label, 'corpusId_groupId')
  save_parquet(groupid_author_df, known_args.savedir, known_args.label, 'groupId_authors')

  df = pd.merge(authors_df, cid_groupid_df, how='left', on='corpusId')
  df = pd.merge(df, groupid_author_df, how='left', on='author_group_id')

  # add edges column.
  df = add_edges(df, cid_author_df)
  save_parquet(df, known_args.savedir, known_args.label, 'dataset')

  elapsed_time = time.time() - start
  print('elapsed_time [sec]: ' + str(elapsed_time))
  print('elapsed_time [hms]: ' + str(dt.timedelta(seconds=elapsed_time)))

  return 0

if __name__ == '__main__':
  main()
