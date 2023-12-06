import h5py
import numpy as np
import scipy.io


data_path = 'data/eccv16_dataset_summe_google_pool5.h5'

# with h5py.File(data_path, "r") as hdf:
#     for name, vid in hdf.items():
#         user_summary = np.array(vid['user_summary'])
#         break


# groundtruth_file = 'data/GT/Air_Force_One.mat'
# groundtruth_data = scipy.io.loadmat(groundtruth_file)
     
# user_score = groundtruth_data.get('user_score')

# total_frames = user_score.shape[0]
# total_users = user_score.shape[1]

# groundtruth_score = groundtruth_data.get('gt_score')
# groundtruth_score = np.array(groundtruth_score)

# sample_score_file = 'output/clustering/Air_Force_One_scores.npy'
# sample_scores = np.load(sample_score_file)
# sample_index_file = 'output/features/Air_Force_One_samples.npy'
# sample_indexes = np.load(sample_index_file)
# prediction_scores = np.zeros(total_frames)
# prediction_scores[sample_indexes] = sample_scores

# keyframe_index_file = 'output/clustering/Air_Force_One_keyframes.npy'
# keyframe_indexes = np.load(keyframe_index_file)

# selected_keyframes = np.zeros((total_frames), dtype=int)
# selected_keyframes[keyframe_indexes] = 1

# print(prediction_scores.shape, selected_keyframes.shape)
# print(selected_keyframes.sum(), keyframe_indexes.shape)

# for i in range(total_frames):
    # print(selected_keyframes[i])


def evaluate_summary(predicted_summary, user_summary, eval_method):
    """ Compare the predicted summary with the user defined one(s).

    :param ndarray predicted_summary: The generated summary from our model.
    :param ndarray user_summary: The user defined ground truth summaries (or summary).
    :param str eval_method: The proposed evaluation method; either 'max' (SumMe) or 'avg' (TVSum).
    :return: The reduced fscore based on the eval_method
    """
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[0:len(predicted_summary)] = predicted_summary
    
    print(S.max(), user_summary.max(), len(predicted_summary))

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G
        print(user, np.sum(G))

        # Compute precision, recall, f-score
        precision = sum(overlapped)/sum(S)
        recall = sum(overlapped)/sum(G)
        if precision+recall == 0:
            f_scores.append(0)
        else:
            f_scores.append(2 * precision * recall * 100 / (precision + recall))

    if eval_method == 'max':
        return max(f_scores)
    else:
        return sum(f_scores)/len(f_scores)


# f_score = evaluate_summary(predicted_summary=selected_keyframes, 
#                            user_summary=user_summary, eval_method='max')
# print(f_score)