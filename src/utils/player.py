import numpy as np

def df_to_ndarray(df):
  ndarr = np.empty(0)
  for idx, row in df.iterrows():
    x = row.to_numpy().T
    tmp = x
    if len(ndarr) == 0:
      ndarr = tmp
    else:
      ndarr = np.vstack([ndarr, tmp])
  return ndarr

def series_to_ndarray(ser):
  ndarr = np.empty(0)
  for idx, val in ser.items():
    # tmp = val[0]
    tmp = val["vector"]
    if len(ndarr) == 0:
      ndarr = tmp
    else:
      ndarr = np.vstack([ndarr, tmp])
  print("data shape; ", ndarr.shape)
  return ndarr