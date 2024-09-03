import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from index_selectors.importance_facility_location import ImportanceFacilityLocation
from libKMCUDA import kmeans_cuda



input_path = '/data/jingdong/datacomp/recap/metadata'
output_path = '/data/jingdong/datacomp/recap/output'

keys = ['recap_txt']
score_key = 'clip_l14_similarity_score'

ratio = 0.3
alpha = 1
devices=[2,3,4]

arrs = []
dfs = []
scores = []

for file in tqdm(os.listdir(input_path)):
    full_path = os.path.join(input_path, file)
    if file.endswith('.npz'):
        arr = np.load(full_path)[keys[0]]
        arrs.append(arr)

    elif file.endswith('.parquet'):
        df = pd.read_parquet(full_path)
        scores.append(df[score_key].to_numpy())
        dfs.append(df)

features = np.concatenate(arrs, axis=0).astype(np.float32)
importance_scores = np.concatenate(scores, axis=0).astype(np.float32)
metadata = pd.concat(dfs)

print(features.shape)
print(importance_scores.shape)
print(len(metadata))

print('start fitting')
# kmeans_cuda(features, 10000, tolerance=0.01, init="k-means++",
#                 yinyang_t=0.1, metric="L2", average_distance=False,
#                 seed=0, device=3, verbosity=0)

optimizer = ImportanceFacilityLocation(devices=devices)

index, _, _ = optimizer.fit_greedi(features, importance_scores, int(len(features) * ratio), alpha=alpha)
selected_subset = metadata.iloc[index.cpu().tolist()]

selected_subset.to_parquet(os.path.join(output_path, f'IVL_alpha_{alpha}_{"-".join(keys)}_ratio_{ratio}.parquet'))
