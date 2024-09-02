import torch
import numpy as np
import pdb

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


class ImportanceFacilityLocation:
    def __init__(self, devices, dtype=torch.float32):
        self.devices = devices
        assert len(devices) > 0
        self.dtype = dtype

    def fit_greedi(self, features, importance_scores, k, epsilon=0.01, alpha=0.5):
        m = len(self.devices)
        total = len(features)
        n = total // m

        _k = min(int(np.ceil(n/total*k*1.1)), n)
        _last_k = min(int(np.ceil(len(features[(m-1)*n:])/total*k*1.1)), len(features[(m-1)*n:]))

        offsets = [i*n for i in range(m)]

        args = [
            (features[i*n : (i+1)*n], importance_scores[i*n : (i+1)*n], self.devices[i], _k) 
            for i in range(m-1)
        ]
        args.append((features[(m-1)*n:], importance_scores[(m-1)*n:], self.devices[m-1], _last_k))
        
        partial_fit_stochastic_submodular = partial(
            self.fit_stochastic_submodular, 
            epsilon=epsilon,
            alpha=alpha
        )

        with Pool(processes=m) as pool:
            results = pool.starmap(partial_fit_stochastic_submodular, args)
        
            collected_index = torch.cat([self.to_device(results[i][0], self.devices[0]) + offsets[i] for i in range(len(results))])
            collected_features = torch.cat([self.to_device(results[i][1], self.devices[0]) for i in range(len(results))], dim=0)
            collected_scores = torch.cat([self.to_device(results[i][2], self.devices[0]) for i in range(len(results))])

            assert len(collected_index.shape) == 1
            assert len(collected_scores.shape) == 1
            assert len(collected_features.shape) == 2

            pdb.set_trace()
            second_stage_index, f, s = self.fit_stochastic_submodular(
                collected_features, 
                collected_scores, 
                self.devices[0], 
                k, 
                epsilon, 
                alpha
            )
            
            assert (collected_scores[second_stage_index] == s).all()

            return collected_index[second_stage_index], f, s


    def fit_stochastic_submodular(self, features, importance_scores, device, k, epsilon=0.01, alpha=0.5):
        assert epsilon < 1
        assert k <= len(features)
        
        features = self.to_device(features, device)
        assert len(features.shape) == 2
        padding = self.to_device(torch.ones(1, features.shape[1]), device)
        features = torch.concat([padding, features])

        n = features.shape[0]

        importance_scores = self.to_device(importance_scores, device).squeeze()
        padding = self.to_device(torch.ones(1), device)
        importance_scores = torch.concat([padding, importance_scores])
        assert n == importance_scores.shape[0]

        sample_size = int(np.ceil(n / k * np.log(1/epsilon)))
        
        selected_index = []
        max_similarity = torch.zeros(n, 1, dtype=self.dtype, device=device)
        index = torch.arange(n, dtype=torch.int64, device=device)
        print(index.shape)
        for _ in tqdm(range(k)):
            not_selected = index.nonzero()
            # pdb.set_trace()
            shuffle = torch.randperm(len(not_selected), dtype=torch.int64, device=device)
            subset = not_selected[shuffle][:sample_size]
            subset = subset.squeeze()
            # print(features.shape)
            # print(features[subset].shape)
            similarity = features @ features[subset].T
            d = torch.maximum(max_similarity, similarity).sum(dim=0)

            scores = alpha * d + (1-alpha) * importance_scores[subset]  
            subset_idx = scores.argmax()      
            # print(subset_idx)    
            selected_idx = subset[subset_idx].item()
            # print(selected_idx)
            
            index[selected_idx] = 0
            selected_index.append(selected_idx)
            max_similarity = torch.maximum(max_similarity, similarity[:, subset_idx].view((-1,1)))

        selected_index = torch.tensor(selected_index, dtype=torch.int64, device=device)
        selected_features = features[selected_index]
        selected_importance_scores = importance_scores[selected_index]
        
        del features
        del importance_scores
        torch.cuda.empty_cache()

        return selected_index-1, selected_features, selected_importance_scores


    def to_device(self, data, device):
        if isinstance(data, list):
            return torch.tensor(data, dtype=self.dtype).to(device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data, dtype=self.dtype).to(device)
        elif isinstance(data, torch.Tensor):
            if data.get_device() == device and data.dtype == self.dtype:
                return data
            return data.to(dtype=self.dtype, device=device)
        else:
            raise NotImplementedError


optimizer = ImportanceFacilityLocation(devices=[1,2,3,4])



importance_scores = torch.rand(12600000, 1)
features = torch.rand(12600000, 768)
print('start fitting')
optimizer.fit_greedi(features, importance_scores, 3500000)


        