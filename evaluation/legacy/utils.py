# -*- coding: utf-8 -*-
# author: Bhavya Jain
# link: https://github.com/wulfebw/algorithms/blob/master/scripts/dynamic_programming/knapsack.py

import numpy as np
from scipy import interpolate


def knapsack(W, wt, val, n):
    """ Maximize the value that a knapsack of capacity W can hold. You can either put the item or discard it, there is
    no concept of putting some part of item in the knapsack.

    :param int W: Maximum capacity -in frames- of the knapsack.
    :param list[int] wt: The weights (lengths -in frames-) of each video shot.
    :param list[float] val: The values (importance scores) of each video shot.
    :param int n: The number of the shots.
    :return: A list containing the indices of the selected shots.
    """
    K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    selected = []
    w = W
    for i in range(n, 0, -1):
        if K[i][w] != K[i - 1][w]:
            selected.insert(0, i - 1)
            w -= wt[i - 1]

    return selected


def generate_summary(segmentation, scores, fill_mode,
                     select_percentile=0.15):
    """ Generate the automatic machine summary, based on the video shots; the frame importance scores; the number of
    frames in the original video and the position of the sub-sampled frames of the original video.

    :param np.ndarray segmentation: The boundaries and number of frames of the shots for the -original- testing video.
    :param np.ndarray all_scores: The indices of sub-sampled frames in the original video together with its predicted importances.
    :param str fill_mode: The method to use for filling the missing frames in the original video.
    :param float select_percentile: The percentage of the frames to select from the last shot.
    :return: A list containing the indices of the selected frames for the -original- testing video.
    """
    
    # Get shots' boundaries
    # The segmentation is [number_of_shots, 3]
    # The scores is [number_of_subsampled_frames, 2]
    
    shot_bound = segmentation[:, :-1]  # [number_of_shots, 2]
    frame_init_scores = scores[:, 1]
    shot_frames = segmentation[:, -1]
    positions = scores[:, 0]
    
    n_frames = shot_frames.sum()

    # Compute the importance scores for the initial frame sequence (not the sub-sampled one)
    scorer = interpolate.interp1d(positions, frame_init_scores,
                                  kind=fill_mode,
                                  fill_value=(frame_init_scores[0],
                                              frame_init_scores[-1])
                                  )
    
    frame_scores = scorer(np.arange(n_frames))
    
    # frame_scores = np.zeros(n_frames, dtype=np.float32)
    # if positions.dtype != int:
    #     positions = positions.astype(np.int32)
    # if positions[-1] != n_frames:
    #     positions = np.concatenate([positions, [n_frames]])
    # for i in range(len(positions) - 1):
    #     pos_left, pos_right = positions[i], positions[i + 1]
    #     if i == len(frame_init_scores):
    #         frame_scores[pos_left:pos_right] = 0
    #     else:
    #         frame_scores[pos_left:pos_right] = frame_init_scores[i]

    # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
    shot_imp_scores = []
    shot_lengths = []
    for shot in shot_bound:
        shot_lengths.append(shot[1] - shot[0] + 1)
        shot_imp_scores.append((frame_scores[shot[0]:shot[1] + 1].mean()).item())

    # Select the best shots using the knapsack implementation
    final_shot = shot_bound[-1]
    final_max_length = int((final_shot[1] + 1) * select_percentile)

    selected = knapsack(final_max_length, shot_lengths,
                        shot_imp_scores, len(shot_lengths))

    # Select all frames from each selected shot (by setting their value in the summary vector to 1)
    summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
    for shot in selected:
        summary[shot_bound[shot][0]:shot_bound[shot][1] + 1] = 1

    return summary


# if __name__ == "__main__":
#     """ Driver program to test above function
#     val = [4, 4, 2, 2, 2, 4]
#     wt =  [2, 2, 1, 1, 1, 2]
#     W = 7
#     n = len(val)
#     selected = knapSack(W, wt, val, n)
#     print(selected)
#     """
    
#     pass
