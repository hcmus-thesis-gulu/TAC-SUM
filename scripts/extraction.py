import os
import time
import argparse
import numpy as np

from model.propogator import Clusterer
from model.selector import Selector
from model.utils import calculate_num_clusters


def localize_context(embeddings, method, n_clusters, window_size,
                     min_seg_length, distance, embedding_dim, intermediate_components=50, final_reducer='tsne'):
    clusterer = Clusterer(method, distance, n_clusters, embedding_dim, intermediate_components, final_reducer=final_reducer)
    selector = Selector(window_size, min_seg_length)
    labels, reduced_embeddings = clusterer.cluster(embeddings)
    
    return (labels, selector.select(labels),
            clusterer.num_clusters, reduced_embeddings)


def localize_videos(embedding_folder, context_folder, method,
                    max_len, window_size, min_seg_length,
                    distance, embedding_dim, modulation, intermediate_components, final_reducer='tsne'):
    for embedding_name in os.listdir(embedding_folder):
        if embedding_name.endswith('_embeddings.npy'):
            filename = embedding_name[:-len('_embeddings.npy')]
            print(f"Processing the context of video {filename}")
            
            embedding_file = os.path.join(embedding_folder, embedding_name)
            embeddings = np.load(embedding_file)
            print(f"The extracted context has {embeddings.shape[0]} embeddings")
            
            # sample_file = os.path.join(embedding_folder, f'{filename}_samples.npy')
            # samples = np.load(sample_file)
            segment_file = filename + '_segments.npy'
            labels_file = filename + '_labels.npy'
            reduced_file = filename + '_reduced.npy'
            
            segment_path = os.path.join(context_folder, segment_file)
            labels_path = os.path.join(context_folder, labels_file)
            
            # Reduced embeddings
            reduced_path = os.path.join(embedding_folder, reduced_file)
            
            print(f'Clustering frames of {filename}')
            if os.path.exists(segment_path):
                continue
            
            num_clusters = calculate_num_clusters(num_frames=embeddings.shape[0],
                                                  max_length=max_len,
                                                  modulation=modulation
                                                  )
            print(f"Initial number of clusters is {num_clusters} with modulation {modulation}")
            local_context = localize_context(embeddings=embeddings,
                                             method=method,
                                             n_clusters=num_clusters,
                                             window_size=window_size,
                                             min_seg_length=min_seg_length,
                                             distance=distance,
                                             embedding_dim=embedding_dim,
                                             intermediate_components=intermediate_components,
                                             final_reducer=final_reducer
                                             )
            
            labels, segments, n_clusters, reduced_embs = local_context
            print(f'Number of clusters: {n_clusters}')
            print(f'Number of segments: {len(segments)}')
            
            np.save(segment_path, segments)
            np.save(labels_path, labels)
            np.save(reduced_path, reduced_embs)


def main():
    parser = argparse.ArgumentParser(description='Convert Global Context of Videos into Local Semantics.')
    
    parser.add_argument('--embedding-folder', type=str, required=True,
                        help='path to folder containing feature files')
    parser.add_argument('--context-folder', type=str, required=True,
                        help='path to output folder for clustering')
    
    parser.add_argument('--method', type=str, default='ours',
                        choices=['kmeans', 'dbscan', 'gaussian',
                                 'agglo', 'ours'],
                        help='clustering method')
    # parser.add_argument('--num-clusters', type=int, default=0,
    #                     help='Number of clusters with 0 being automatic detection')
    parser.add_argument('--max-len', type=int, default=60,
                        help='Maximum length of output summarization in seconds')
    parser.add_argument('--distance', type=str, default='euclidean',
                        choices=['jensenshannon', 'euclidean', 'cosine'],
                        help='distance metric for clustering')
    parser.add_argument('--embedding-dim', type=int, default=-1,
                        help='dimension of embeddings')
    
    parser.add_argument('--window-size', type=int, default=10,
                        help='window size for smoothing')
    parser.add_argument('--min-seg-length', type=int, default=10,
                        help='minimum segment length')
    parser.add_argument('--modulation', type=float, default=1e-3,
                        help='modulation factor for number of clusters')
    parser.add_argument('--intermediate-components', type=int, default=-1,
                        help='intermediate components')
    parser.add_argument('--final-reducer', type=str, default='tsne',
                        choices=['pca', 'tsne'])
    
    args = parser.parse_args()

    localize_videos(embedding_folder=args.embedding_folder,
                    context_folder=args.context_folder,
                    method=args.method,
                    # num_clusters=args.num_clusters,
                    max_len=args.max_len,
                    window_size=args.window_size,
                    min_seg_length=args.min_seg_length,
                    distance=args.distance,
                    embedding_dim=args.embedding_dim,
                    modulation=args.modulation,
                    intermediate_components=args.intermediate_components,
                    final_reducer=args.final_reducer,
                    )


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
