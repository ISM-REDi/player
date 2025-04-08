import string
import sys
import pandas as pd
import itertools

def levenshteinDistance(A, B):
  N, M = len(A), len(B)
  # Create an array of size NxM
  dp = [[0 for i in range(M + 1)] for j in range(N + 1)]

  # Base Case: When N = 0
  for j in range(M + 1):
    dp[0][j] = j
  # Base Case: When M = 0
  for i in range(N + 1):
    dp[i][0] = i
    # Transitions
  for i in range(1, N + 1):
    for j in range(1, M + 1):
      if A[i - 1] == B[j - 1]:
        dp[i][j] = dp[i-1][j-1]
      else:
        dp[i][j] = 1 + min(
          dp[i-1][j], # Insertion
          dp[i][j-1], # Deletion
          dp[i-1][j-1] # Replacement
          )
  return dp[N][M]

def generalizedLevenshteinDistance(A, B):
  # TODO
  return 0

def makeAuthorString(article_authors, author_ascii_dict):
  author_str = ""
  for a in article_authors:
    author_str += author_ascii_dict[a]

  return author_str

def authorsLevenshteinDistance(list_A, list_B):
  idstr = string.ascii_uppercase
  i = 0
  authors = {}
  for d in set(list_A + list_B):
    authors[d] = idstr[i]
    i += 1

  author_str_a = makeAuthorString(list_A, authors)
  author_str_b = makeAuthorString(list_B, authors)

  ld = levenshteinDistance(author_str_a, author_str_b)
  #  gld = generalizedLevenshteinDistance([], [])
  return ld

def clusterLevenshteinDistance(cluster_df):
  cluster_distance_total = 0
  for v in itertools.combinations(cluster_df.index.values, 2):
    i = list(v)
    first = cluster_df.loc[i[0], 'author_ids'].tolist()#author_ids authors
    second = cluster_df.loc[i[1], 'author_ids'].tolist()#author_ids 
    tmp_score = authorsLevenshteinDistance(first, second)
    # print("Levenshtein Distance = {}".format(tmp_score))
    cluster_distance_total += tmp_score
  return cluster_distance_total

def main():
  # $ poetry run python src/player/clustering_gmm.py datas/akaike2/ datas/akaike2/akaike_gram_embedding.parquet 100
  detail_file = sys.argv[1]
  cluster_file = sys.argv[2]
  n = sys.argv[3]

  # sample
  # docA_authors = ["author1", "author2", "author3", "author4"]
  # docB_authors = ["author1", "author2", "author3", "author4", "author5"] # docA とほとんど一緒、一人だけ増えた。
  # docC_authors = ["author1", "author6", "author7", "autho8", "author9"] # docA と一人だけ一緒、docBと長さは同じ。
  # docD_authors = ["author10"] # どれとも異なる著者。

  # print("Levenshtein Distance between {} and {} = {}".format("A", "B", authorsLevenshteinDistance(docA_authors, docB_authors)))
  # print("Levenshtein Distance between {} and {} = {}".format("A", "C", authorsLevenshteinDistance(docA_authors, docC_authors)))
  # print("Levenshtein Distance between {} and {} = {}".format("A", "D", authorsLevenshteinDistance(docA_authors, docD_authors)))
  # print("Levenshtein Distance between {} and {} = {}".format("B", "C", authorsLevenshteinDistance(docB_authors, docC_authors)))
  # print("Levenshtein Distance between {} and {} = {}".format("B", "D", authorsLevenshteinDistance(docB_authors, docD_authors)))
  # print("Levenshtein Distance between {} and {} = {}".format("C", "D", authorsLevenshteinDistance(docC_authors, docD_authors)))
  # # /sample
  
  # calculate Levenshtein Distance of all PLAYER.
  # args; クラスタリング結果, 著者情報をもつ書誌データ
  # 1. クラスタ内に2以上の書誌が含まれるクラスタだけ抽出する。
  # 2. その書誌の著者情報を取得する。
  # 3. クラスタごとのLevenshtein Distanceを計算し、1.のサイズでAverageをとる。

  df_detail = pd.read_parquet(detail_file)
  df_cluster = pd.read_csv(cluster_file, index_col=False, names=["corpusId", "cluster_no"], dtype={"corpusId":int, "cluster_no":int})
  df_detail["corpusId"] = df_detail["corpusId"].astype("int64")
  
  df = pd.merge(df_detail, df_cluster, on="corpusId")
  df.dropna(subset="corpusId", inplace=True)
  cluster_count_df = df.groupby("cluster_no").count()
  target_clusters = cluster_count_df[cluster_count_df["corpusId"]> 1].index.values
  print("targets; ", target_clusters)
  sum_distance = 0
  for cluster_no in target_clusters:
    cluster_distance_total = clusterLevenshteinDistance(df[df["cluster_no"] == cluster_no])
    # print("cluster {} Levenshtein Distance total = {}".format(cluster_no, cluster_distance_total))
    sum_distance += cluster_distance_total
  
  
  # Average
  average_levenshtein = sum_distance / len(target_clusters)
  print("Levenshtein Score; {} (n = {})".format(average_levenshtein, n))

  with open("results/exp_3_20241116/levenshtein_scores.csv", mode='a') as f:
    f.write(str(n)+","+str(average_levenshtein)+"\n")

if __name__ == '__main__':
  main()

