import os, pdb
import math
import faiss
import time
import random
import pickle
import fsspec
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
from typing import Any, List, Tuple
import pandas as pd
from functools import partial
import argparse
from concurrent.futures import ThreadPoolExecutor

def load_metadata(
        metadata_dir_path: str, num_workers: int, columns: List[str] = None
) -> pd.DataFrame:
    """load metadata for many parquets

    Args:
        metadata_dir_path (str): directory where metadata is stored
        num_workers (int): number of cpu workers, each of which processes a parquet
        columns (List[str], optional): list of columns to retain from the parquet. Defaults to None.

    Returns:
        pd.DataFrame: loaded parquet columns
    """
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    parquet_paths = [str(x) for x in fs.ls(url) if ".parquet" in x]
    worker = partial(pd.read_parquet, columns=columns, filesystem=fs)

    return worker_threadpool(worker, pd.concat, parquet_paths, num_workers)


def get_threshold(
        metadata_dir_path: str, key: str, fraction: float, num_workers: int
) -> float:
    """compute a threshold given a collection of metadata, a key, and a target fraction of the pool to keep

    Args:
        metadata_dir_path (str): directory where metadata is stored
        key (str): column we are interested in the parquet column store
        fraction (float): top k fraction, represented as a decimal.
        num_workers (int): number of cpu workers, each of which processes a parquet.

    Returns:
        float: threshold value
    """
    print("loading all metadata for threshold computation")
    df = load_metadata(metadata_dir_path, num_workers=num_workers, columns=[key])
    n = int(len(df) * fraction)
    threshold = -np.sort(-df[key].values)[n]

    return threshold


def load_key_helper(fs_url: Tuple[Any, str], key: str) -> np.ndarray:
    """helper to read a parquet and load the uids

    Args:
        fs_url (Tuple[Any, str]): pair of fsspec file system and parquet url

    Returns:
        np.ndarray: array of uids
    """
    fs, url = fs_url
    df = pd.read_parquet(url, columns=[key], filesystem=fs)
    return np.array(df[key].values)


def load_features(metadata_dir_path: str, fraction: float, num_workers: int, feature_type: str = 'image', embedding_key=None) -> np.ndarray:
    """load all uids in a metadata containing directory

    Args:
        metadata_dir_path (str): directory where metadata is stored
        num_workers (int): number of cpu workers, each of which processes a parquet

    Returns:
        np.ndarray: array of uids
    """
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    parquet_paths = [(fs, str(x)) for x in fs.ls(url) if ".parquet" in x]

    worker = partial(
        load_features_helper,
        fraction=fraction,
        feature_type=feature_type,
        embedding_key=embedding_key
    )

    return worker_threadpool(
        worker, np.concatenate, parquet_paths, num_workers
    )


def load_features_helper(fs_url: Tuple[Any, str], fraction: float = 1.0, feature_type: str = 'image', embedding_key=None) -> np.ndarray:
    """helper to read a parquet and load the uids

    Args:
        fs_url (Tuple[Any, str]): pair of fsspec file system and parquet url

    Returns:
        np.ndarray: array of uids
    """
    fs, url = fs_url
    df = pd.read_parquet(url, columns=["uid"], filesystem=fs)

    with fs.open(url.replace('.parquet', '.npz')) as f:
        if embedding_key is not None:
            features = np.load(f)
            # candidate_embedding = np.load(f)[embedding_key]
            candidate_embedding = np.concatenate([features[key] for key in embedding_key], axis=-1)
        elif feature_type == 'image':
            candidate_embedding = np.load(f)["l14_img"]
            # graph = GraphDensitySampler(X=candidate_embedding, y=None, seed=0,
            #                             importance_scores=df[key], n_neighbor=5)
        elif feature_type in ['image|text']:
            features = np.load(f)
            image_embeddings = features["l14_img"]
            text_embeddings = features["l14_txt"]
            candidate_embedding = np.concatenate((image_embeddings, text_embeddings), axis=-1)
        elif feature_type in ['text|image']:
            features = np.load(f)
            image_embeddings = features["l14_img"]
            text_embeddings = features["l14_txt"]
            candidate_embedding = np.concatenate((text_embeddings, image_embeddings), axis=-1)
        else:
            candidate_embedding = np.load(f)["l14_txt"]

    if fraction < 1.0:
        selection_size = int(df.size * fraction)
        selected_idxs = random.sample(range(df.size), k=selection_size)
        return np.array(
            [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in df.iloc[selected_idxs]["uid"].values],
            np.dtype("u8,u8"),
        ), candidate_embedding[selected_idxs, :]
    else:
        return np.array(
            [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in df["uid"].values],
            np.dtype("u8,u8"),
        ), candidate_embedding


def worker_threadpool(
        worker_fn: Any, concat_fn: Any, paths: List[str], n_workers: int
) -> np.ndarray:
    """get filtered uids

    Args:
        worker_fn (Any): function to map over the pool
        concat_fn (Any): function to use to collate the results
        paths (List[str]): metadata paths to process
        n_workers (int): number of cpu workers

    Returns:
        np.ndarray: filtered uids
    """
    print("creating thread pool for processing")
    with Pool(n_workers) as pool:
        values = []
        for v in tqdm(
                pool.imap_unordered(worker_fn, paths),
                total=len(paths),
        ):
            values.append(v)

    if type(values[0]) not in [list, tuple]:
        return concat_fn(values)
    else:
        return [concat_fn([v[i] for v in values]) for i in range(0, len(values[0]))]


def create_faiss_index(train_embeddings, d=768, gpu_config=None):
    # took 5 minutes
    # took 12 minuts for image+text

    m = 16  # number of centroid IDs in final compressed vectors
    bits = 8  # number of bits in each centroid
    nlist = 100

    quantizer = faiss.IndexFlatL2(d)  # we keep the same L2 distance flat index
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
    if gpu_config is not None:
        index = faiss.index_cpu_to_gpu(gpu_config['res'], gpu_config['device'], index)

    # train_embeddings = np.load(embeddings_file)
    print(train_embeddings.shape)
    # train_embeddings = np.float32(train_embeddings)
    start_time = time.time()
    index.train(train_embeddings)
    print("--- took %s seconds to train ---" % (time.time() - start_time))
    # faiss.write_index(faiss.index_gpu_to_cpu(index), out_index_file)

    return index


def add_to_index(metadata_dir_path: str, index, feature_type: str, embedding_key=None):
    # index = faiss.read_index(index_path)
    # index = faiss.index_cpu_to_gpu(gpu_config['res'], gpu_config['device'], index)
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    parquet_paths = [(fs, str(x)) for x in fs.ls(url) if ".parquet" in x]

    uids = []
    overall_start_time = time.time()
    for p in tqdm(parquet_paths):
        start_time = time.time()
        u, features = load_features_helper(p, feature_type=feature_type, embedding_key=embedding_key)
        # features = np.float32(features)
        index.add(features)
        print("--- took %s seconds to train ---" % (time.time() - start_time))
        uids.append(u)
    print("--- took %s seconds to add all embeddings ---" % (time.time() - overall_start_time))

    # np.save(index_path.replace('.index', '_index_uids.npy'), uids)
    # faiss.write_index(faiss.index_gpu_to_cpu(index), index_path.replace('.index', '_added.index'))

    return uids, index


def initialize_graph_helper(fs_url: Tuple[Any, str], index, index_scores, feature_type: str = 'image', top_k: int = 5,
                            offset_map={}, embedding_key=None) -> np.ndarray:
    """helper to read a parquet and load the uids

    Args:
        fs_url (Tuple[Any, str]): pair of fsspec file system and parquet url

    Returns:
        np.ndarray: array of uids
    """
    # index = faiss.read_index(index_path)
    fs, url = fs_url
    absolute_offset = offset_map[url]
    df = pd.read_parquet(url, columns=["uid", "clip_l14_similarity_score"], filesystem=fs)
    assert df["clip_l14_similarity_score"][100] == index_scores[absolute_offset + 100], "order does not match"
    with fs.open(url.replace('.parquet', '.npz')) as f:
        if embedding_key is not None:
            # candidate_embedding = np.load(f)[embedding_key]
            features = np.load(f)
            candidate_embedding = np.concatenate([features[key] for key in embedding_key], axis=-1)
        elif feature_type == 'image':
            candidate_embedding = np.load(f)["l14_img"]
            # graph = GraphDensitySampler(X=candidate_embedding, y=None, seed=0,
            #                             importance_scores=df[key], n_neighbor=5)
        elif feature_type in ['image|text', 'text|image']:
            features = np.load(f)
            image_embeddings = features["l14_img"]
            text_embeddings = features["l14_txt"]
            candidate_embedding = np.concatenate((image_embeddings, text_embeddings), axis=-1)
        else:
            candidate_embedding = np.load(f)["l14_txt"]

    gamma = 1.
    step = 50000
    epsilon = 0.0000001
    key = "clip_l14_similarity_score"
    offset = 0.115  # ??
    index_scores = index_scores + offset

    total_samples = candidate_embedding.shape[0]
    total_splits = math.ceil(total_samples / step)
    graph_scores = []
    neighbors = []
    distances = []
    clip_scores = []
    for i in tqdm(range(0, total_splits), desc='Iterating over %s splits' % total_splits):
        # batch_embeddings = candidate_embedding[step * i:step * (i + 1)].astype('float32')
        D_raw, I_raw = index.search(np.float32(candidate_embedding[step * i:step * (i + 1)]), k=top_k + 1)
        D_raw = D_raw - np.tile(np.expand_dims(D_raw[:, 0].transpose(), 1), (1, top_k + 1))
        D, I = [], []

        D.append(D_raw[:, 1:top_k + 1])
        I.append(I_raw[:, 1:top_k + 1])

        D = np.concatenate(D, axis=0)
        I = np.concatenate(I, axis=0)

        # if i == 0:
        #     print("Test: ", 0, absolute_offset, I_raw[:10], '-->', I[:10], D_raw[:10], '-->', D[:10])

        distances.append(D)
        dist = np.exp(-D * gamma) * np.maximum(index_scores[I], np.ones_like(I) * epsilon)
        # if i == 0:
        #     print(I[:10], index_scores[I][:10], D[:10], dist[:10])
        if len(dist.shape) == 1:
            dist = np.expand_dims(dist, axis=-1)
        scores = np.array(df.iloc[step * i:step * (i + 1)][key])
        scores = scores + offset
        clip_scores.append(np.copy(scores))
        # sorted_idxs = np.argsort(scores)[::-1]
        multiplier = np.sum(dist, axis=-1)
        # if i == 0:
        #     print("Test: ", scores[:10], multiplier[:10], end=" ")
        scores = multiplier + np.maximum(scores, np.ones_like(scores) * epsilon)
        # if i == 0:
        #     print("-->", scores[:10])
        # scores = multiplier*np.maximum(scores, np.ones_like(scores)*epsilon)
        # for idx in sorted_idxs[:10]:
        #     print("%s = %s * %s" % (scores[idx], multiplier[idx], clip_scores[-1][idx]), D[idx], index_scores[I][idx])
        # sorted_idxs = sorted_idxs[::-1]
        # for idx in sorted_idxs[:10]:
        #     print("%s = %s * %s" % (scores[idx], multiplier[idx], clip_scores[-1][idx]), D[idx], index_scores[I][idx])
        graph_scores.append(scores)
        neighbors.append(I)

    return np.array(
        [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in df["uid"].values],
        np.dtype("u8,u8"),
    ), np.concatenate(neighbors), np.concatenate(graph_scores), np.concatenate(clip_scores), np.concatenate(distances)


def initialize_graph(metadata_dir_path: str, index, index_scores, offset_map, num_workers: int, feature_type: str = 'image',
                     top_k=5, embedding_key=None):
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    parquet_paths = [(fs, str(x)) for x in fs.ls(url) if ".parquet" in x]
    # worker = partial(
    #     initialize_graph_helper,
    #     index=index,
    #     index_scores=index_scores,
    #     feature_type=feature_type,
    #     top_k=top_k,
    #     offset_map=offset_map,
    #     embedding_key=embedding_key
    # )

    values = []
    for path in tqdm(parquet_paths):
        v = initialize_graph_helper(
            path,
            index=index,
            index_scores=index_scores,
            feature_type=feature_type,
            top_k=top_k,
            offset_map=offset_map,
            embedding_key=embedding_key
        )
        values.append(v)

    if type(values[0]) not in [list, tuple]:
        return np.concatenate(values)
    else:
        return [np.concatenate([v[i] for v in values]) for i in range(0, len(values[0]))]

    # return worker_threadpool(
    #     worker, np.concatenate, parquet_paths, num_workers
    # )


def iterative_selection(graph_scores_path: str, neighbors_path: str, clip_scores_path: str,
                        distances_path: str, fraction: float, gamma: float = 1.0, graph_score=False, threshold=None):
    # sort
    graph_scores = np.load(graph_scores_path)
    neighbors = np.load(neighbors_path)
    distances = np.load(distances_path)
    k = int(fraction * graph_scores.shape[0])
    clip_scores = np.load(clip_scores_path)
    print(clip_scores.shape, graph_scores.shape)

    # frequency
    bins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    clip_score_min = np.min(clip_scores)
    clip_score_max = np.max(clip_scores)
    print("CLIP score minimum and maximum: ", clip_score_min, clip_score_max)
    # clip_scores = clip_scores - clip_score_min
    for x in clip_scores:
        bins[int(x * 10)] += 1
    total = sum(bins)

    print('--------- Frequency of CLIP scores -----------')
    print(bins, total)
    print('--------- Normed Frequency of CLIP scores -----------')
    bins = [round(float(b) / total, 2) for b in bins]
    print(bins)
    # offset = 0.115
    # clip_scores = clip_scores + offset
    sorted_clip_idxs = np.argsort(clip_scores)[::-1]

    # if not os.path.exists(sorted_index_path):
    #     start_time = time.time()
    print("Min and max of clip scores are %s and %s" % (np.min(clip_scores), np.max(clip_scores)))
    print("Min and max of updated node scores are %s and %s" % (np.min(graph_scores), np.max(graph_scores)))
    sorted_idxs = np.argsort(graph_scores)[::-1]
    #     print("-------- Took %s seconds to sort" % (time.time() - start_time))
    #     np.save(sorted_index_path, sorted_idxs)
    # else:
    #     sorted_idxs = np.load(sorted_index_path)

    for i in range(0, 10):
        print("After graph updates, most important samples as per graph: %s --> %s" % (
        clip_scores[sorted_idxs[i]], graph_scores[sorted_idxs[i]]))
    for i in range(0, 10):
        print("After graph updates, most important samples as per CLIP: %s --> %s" % (
        clip_scores[sorted_clip_idxs[i]], graph_scores[sorted_clip_idxs[i]]))

    if graph_score:
        selected_idxs = sorted_idxs[:k]
    else:
        # get a sample, reduce values, increase counter
        start_time = time.time()
        counter = 0
        selected = np.zeros(graph_scores.shape[-1])
        print(selected.shape)
        selected_idxs = []
        standby = []
        while len(selected_idxs) < k:

            if len(standby) >= 1 and np.max([graph_scores[standby]]) >= graph_scores[sorted_idxs[counter]]:
                idx = np.argmax(graph_scores[standby])
                selected_idxs.append(standby[idx])
                # for n_k, n in enumerate(neighbors[standby[idx]]):
                #     graph_scores[n] = graph_scores[n] - graph_scores[standby[idx]]*np.exp(distances[standb[idx]])
                graph_scores[neighbors[standby[idx]]] = graph_scores[neighbors[standby[idx]]] - graph_scores[
                    standby[idx]] * np.exp(-distances[standby[idx]] * gamma)
                selected[standby[idx]] = 1
                del standby[idx]
                continue
            else:
                if selected[sorted_idxs[counter]] == 0:

                    # check for more than last selected score and next score and put on standby
                    if graph_scores[sorted_idxs[counter]] < graph_scores[sorted_idxs[counter + 1]]:
                        standby.append(sorted_idxs[counter])
                    else:
                        selected_idxs.append(sorted_idxs[counter])
                        # for n in neighbors[sorted_idxs[counter]]:
                        #     graph_scores[n] = graph_scores[n] - graph_scores[sorted_idxs[counter]]
                        graph_scores[neighbors[sorted_idxs[counter]]] = graph_scores[neighbors[sorted_idxs[counter]]] - \
                                                                        graph_scores[sorted_idxs[counter]] * np.exp(
                            -distances[sorted_idxs[counter]] * gamma)
                        selected[sorted_idxs[counter]] = 1

                counter += 1

            if counter % 50000 == 0:
                print("Selected %s samples" % len(selected_idxs))
                print("Length of standy cache: ", len(standby))
        print("-------- Took %s seconds to iterate" % (time.time() - start_time))

    # frequency
    bins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for x in selected_idxs:
        bins[int(clip_scores[x] * 10)] += 1
    total = sum(bins)
    # bins = [round(float(b)/total, 2) for b in bins]
    print('--------- Frequency of selected CLIP scores -----------')
    print(bins)

    if threshold is not None:
        threshold = threshold + 0.115  # to avoid negative CLIP score
        selected_idxs = [idx for idx in selected_idxs if clip_scores[idx] >= threshold]
        print("After threshold selection")
        # frequency
        bins = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for x in selected_idxs:
            bins[int(clip_scores[x] * 10)] += 1
        total = sum(bins)
        # bins = [round(float(b)/total, 2) for b in bins]
        print('--------- Frequency of selected CLIP scores -----------')
        print(bins)

    print("Selected %s samples" % len(selected_idxs))
    return np.array(selected_idxs)


def filter_metadata(metadata_dir, selected_index_path, output_path):    
    arr = np.load(selected_index_path)
    index_set = set([f'{item["f0"]:016x}' + f'{item["f1"]:016x}' for item in arr])

    dfs = []
    for file in tqdm(os.listdir(metadata_dir)):
        if file.endswith('.parquet'):
            full_path = os.path.join(metadata_dir, file)
            
            df = pd.read_parquet(full_path)
            df = df[df['uid'].isin(index_set)]

            dfs.append(df)

    filtered_df = pd.concat(dfs, ignore_index=True)
    
    print(len(filtered_df))
    filtered_df.to_parquet(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=str, required=True, help="Path to directory containing DataComp metadata")
    parser.add_argument("--out-dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of workers for processing the dataset in parallel")
    parser.add_argument("--n-neighbors", type=int, default=1, help="Number of nearest neighbors in D2 Pruning")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight for reverse message passing")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the dataset to retain")
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--feature-type", type=str, default='image|text',
                        help="Features to use for computing sample distance", choices=['image', 'text', 'image|text'])
    # parser.add_argument("--embedding-key", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument('--embedding-key', nargs='+', default=None)
    args = parser.parse_args()

    # Step 0: Get all clip scores and offsets for faster retrieval
    fs, url = fsspec.core.url_to_fs(args.metadata_dir)
    parquet_paths = [(fs, str(x)) for x in fs.ls(url) if ".parquet" in x]
    scores = []
    offset = 0
    offset_map = {}
    for p in tqdm(parquet_paths):
        s = load_key_helper(p, key="clip_l14_similarity_score")
        n_samples = s.shape[0]
        offset_map[p[-1]] = offset
        offset = offset + n_samples
        scores.append(s)
    scores = np.concatenate(scores)
    # np.save(os.path.join(args.out_dir, 'clip_scores.npy'), np.concatenate(scores))
    # pickle.dump(offset_map, open(os.path.join(args.out_dir, 'offsets.pkl'), 'wb'))

    # For D2 Pruning on the entire DataComp dataset
    # Step 1: Pool the features of 256k random samples together for training faiss.
    # uids_file = os.path.join(args.out_dir, 'no_filter_256k_uids_%s.npy' % args.feature_type)
    # feature_file = os.path.join(args.out_dir, 'no_filter_256k_%s.npy' % args.feature_type)
    _, features = load_features(args.metadata_dir, 0.02, args.num_workers, args.feature_type, embedding_key=args.embedding_key)
    # np.save(uids_file, uids)
    # np.save(feature_file, features)

    gpu_config = None
    if args.device is not None:
        gpu_config = {'res': faiss.StandardGpuResources(), 'device': args.device}
    # Step 2: Add embeddings to index
    # index_init_file = os.path.join(args.out_dir, 'no_filter_256k_%s.index' % args.feature_type)
    index = create_faiss_index(features, args.feature_dim if args.feature_type in ['image', 'text'] else args.feature_dim*2, gpu_config=gpu_config)
    added_uids, added_index = add_to_index(args.metadata_dir, index, args.feature_type, embedding_key=args.embedding_key)

    # Step 4: Initialize graph
    start_time = time.time()
    uids, neighbors, graph_scores, clip_scores, distances = initialize_graph(args.metadata_dir,
                                                                       added_index,
                                                                       scores,
                                                                       offset_map,
                                                                       args.num_workers, args.feature_type, args.n_neighbors,
                                                                       embedding_key=args.embedding_key)
    np.save(os.path.join(args.out_dir, 'graph_%snn_%s_uids.npy' % (args.n_neighbors, args.feature_type)), uids)
    np.save(os.path.join(args.out_dir, 'graph_%snn_%s_neighbors.npy' % (args.n_neighbors, args.feature_type)), neighbors)
    np.save(os.path.join(args.out_dir, 'graph_%snn_%s_graph_scores.npy' % (args.n_neighbors, args.feature_type)), graph_scores)
    np.save(os.path.join(args.out_dir, 'graph_%snn_%s_clip_scores.npy' % (args.n_neighbors, args.feature_type)), clip_scores)
    np.save(os.path.join(args.out_dir, 'graph_%snn_%s_distances.npy' % (args.n_neighbors, args.feature_type)), distances)
    print("-------- Took %s seconds to initialize graph " % (time.time() - start_time))

    # Step 5: Iterative selection
    selected_idxs = iterative_selection(os.path.join(args.out_dir, 'graph_%snn_%s_graph_scores.npy' % (args.n_neighbors, args.feature_type)),
                                        os.path.join(args.out_dir, 'graph_%snn_%s_neighbors.npy' % (args.n_neighbors, args.feature_type)),
                                        os.path.join(args.out_dir, 'graph_%snn_%s_clip_scores.npy' % (args.n_neighbors, args.feature_type)),
                                        os.path.join(args.out_dir, 'graph_%snn_%s_distances.npy' % (args.n_neighbors, args.feature_type)), args.fraction,
                                        gamma=args.gamma)
    uids = np.load(os.path.join(args.out_dir, 'graph_%snn_%s_uids.npy' % (args.n_neighbors, args.feature_type)))[selected_idxs]
    uids.sort()
    key = args.feature_type
    if args.embedding_key is not None:
        key = args.embedding_key

    selected_uid_path = os.path.join(args.out_dir, 'graph_%snn_g=%s_uids_%s_percent_%s.npy' % (args.n_neighbors, args.gamma, int(args.fraction * 100), key))
    np.save(selected_uid_path, uids)

    selected_metadata_path = os.path.join(
        args.out_dir, 
        'filtered_metadata_graph_%snn_g=%s_uids_%s_percent_%s.parquet' % (args.n_neighbors, args.gamma, int(args.fraction * 100), key))
    filter_metadata(args.metadata_dir, selected_uid_path, selected_metadata_path)
