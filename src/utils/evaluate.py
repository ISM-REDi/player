import string

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

def makeAuthorString(article_authors, author_ascii_dict):
  author_str = ""
  for a in article_authors:
    author_str += author_ascii_dict[a]
  return author_str

def authorsLevenshteinDistance(list_A, list_B):
  idstr = string.ascii_letters + string.digits + string.punctuation
  i = 0
  authors = {}
  try:
    for d in set(list_A + list_B):
      authors[d] = idstr[i]
      i += 1
  except:
    print(i)

  author_str_a = makeAuthorString(list_A, authors)
  author_str_b = makeAuthorString(list_B, authors)

  ld = levenshteinDistance(author_str_a, author_str_b)

  return ld

def ngram_set(s, n=2):
  s = s.lower()
  return set([s[i:i+n] for i in range(len(s)-n+1)])

def jaccard_distance(s1, s2, n=2):
  set1 = ngram_set(s1, n)
  set2 = ngram_set(s2, n)
  if not set1 and not set2:
    return 0.0  # 両方空の場合は距離0
  intersection = set1 & set2
  union = set1 | set2
  return 1 - len(intersection) / len(union)

def jaccard_distance_authors(authors1, authors2):
    set1 = set(a.lower().strip() for a in authors1)
    set2 = set(a.lower().strip() for a in authors2)
    
    if not set1 and not set2:
        return 0.0  # 両方空なら距離ゼロとする
    
    intersection = set1 & set2
    union = set1 | set2
    
    return 1 - len(intersection) / len(union)

def caseJaccardDistance(list_A, list_B):
  return jaccard_distance_authors(list_A, list_B)
