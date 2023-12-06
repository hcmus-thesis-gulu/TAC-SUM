import cv2 as cv
import numpy as np
from scipy import sparse
# Probability distribution distance
from scipy.spatial.distance import jensenshannon


def count_frames(video_path):
    # Extract features for each frame of the video
    video = cv.VideoCapture(video_path)
    # Get the video's frame rate, total frames
    fps = int(video.get(cv.CAP_PROP_FPS))
    
    count = 0
    while True:
        ret, _ = video.read()
        if not ret:
            break
        count += 1
    
    video.release()
    return fps, count
    

def mean_embeddings(embeddings):
    return np.mean(embeddings, axis=0)


def distance_metric(distance):
    if distance == 'jensenshannon':
        return jensenshannon
    elif distance == 'euclidean':
        return distance
    elif distance == 'cosine':
        return distance
    else:
        raise ValueError(f'Unknown distance metric: {distance}')


# Compute the cosine similarity between set of features and its mean
def similarity_score(embeddings, mean=None):
    if mean is None:
        mean = mean_embeddings(embeddings)
    
    return np.dot(embeddings, mean) / (np.linalg.norm(embeddings) *
                                       np.linalg.norm(mean)
                                       )


def construct_connectivity(data, labels):
    # create the connectivity matrix for the agglomerative clustering model
    row = []
    col = []
    subcluster_dict = {}
    embedding_shape = data.shape

    for i in range(embedding_shape[0]):
        if labels[i] not in subcluster_dict:
            subcluster_dict[labels[i]] = []
        subcluster_dict[labels[i]].append(i)
    for subcluster in subcluster_dict.values():
        for i, element in enumerate(subcluster):
            for j in range(i+1, len(subcluster)):
                row.append(element)
                col.append(subcluster[j])
                row.append(subcluster[j])
                col.append(element)
    
    connectivity = sparse.csr_matrix((np.ones(len(row)), (row, col)),
                                     shape=(embedding_shape[0],
                                            embedding_shape[0])
                                     )
    
    return connectivity


def calculate_num_clusters(num_frames, max_length, frame_rate=4, modulation=1e-3):
    max_clusters = max_length*frame_rate
    num_clusters = max_clusters*2.0/(1 + np.exp((-modulation) * num_frames)) - max_clusters
    return int(num_clusters)
