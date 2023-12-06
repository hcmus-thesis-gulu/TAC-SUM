import numpy as np
from preprocess.utils import mean_embeddings, similarity_score


# # For each segment, compute the mean features and
# # similarity of all features with the mean
# def extractSummary(embeddings, segments, method):
#     segment_scores = []
#     keyframe_indices = []
    
#     for _, start, end in segments:
#         # Get the associated features
#         segment_features = embeddings[start:end]
        
#         # Calculate the similarity with representative
#         if method == "mean":
#             mean = mean_embeddings(segment_features)
#         elif method == "middle":
#             mean = segment_features[len(segment_features) // 2]
        
#         score = similarity_score(segment_features, mean)
#         segment_scores.extend(score.tolist())
        
#         # Select the representative with the highest similarity
#         keyframe_index = np.argmax(score) + start
#         keyframe_indices.append(keyframe_index)
    
#     return np.asarray(segment_scores), np.asarray(keyframe_indices).sort()


# def computeSummary(embeddings, segments, length, method):
#     scores, key_indices = extractSummary(embeddings, segments, method)
    
#     if length < len(key_indices):
#         selection_step = len(key_indices) / length
#         selections = key_indices[::selection_step]
#         return selections
#     else:
#         unselected_scores = np.delete(scores, key_indices)
#         selection = np.argpartition(unselected_scores,
#                                     -length)[-length:]
        
#         # Final selections are the keyframes and the selected frames
#         selections = np.concatenate((key_indices, selection))
#         selections.sort()
        
#         return selections
