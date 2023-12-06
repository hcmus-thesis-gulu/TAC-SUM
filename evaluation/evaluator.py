import numpy as np
from classic.utils import generate_summary


# With shots based on KnapSack
def calculateSummary(scores, segments, length, video_length,
                     fill_mode, expand):
    length = int(length)
    key_length = length if length > 0 else int(video_length * expand)
        
    return generate_summary(segments=segments,
                            scores=scores,
                            fill_mode=fill_mode,
                            key_length=key_length)


def computeSummary(scores, keyframe_indices, length, video_length, expand):
    try:
        length = int(length)
        expansion = int(expand)
        
        if length > 0:
            kf_length = length // (2 * int(expansion) + 1)
        else:
            kf_length = len(keyframe_indices)
        
        # print(f"===Video Length: {video_length}===")
        # print(f"===Length: {length}===")
        # print(f"===KF Length: {kf_length}===")
        
        if kf_length < len(keyframe_indices):
            selection_step = (len(keyframe_indices) // kf_length) + 1
        else:
            selection_step = 1
        
        # print(f"===Selection Step: {selection_step}===")
        selection = keyframe_indices[::selection_step]
        # print("===KF SELECTION===")
        # print(selection)
        
        selected_mask = np.isin(scores[:, 0], selection)
        unselected_scores = scores[~selected_mask]
        
        remained_length = min(kf_length - len(selection),
                              len(unselected_scores))
        assert remained_length >= 0
        
        other_positions = np.argpartition(unselected_scores[:, 1],
                                          -remained_length)[-remained_length:]
        other_selection = unselected_scores[other_positions, 0]
    except Exception as error:
        print(error)
        print(f"length: {length}")
        print(f"expand: {expansion}")
        print(f"selection_step: {selection_step}")
        print(f"kf_length: {kf_length}")
        print(f"len(keyframe_indices): {len(keyframe_indices)}")
        print(f"len(selection): {len(selection)}")
        print(f"remained_length: {remained_length}")
    
    # print("===OTHER SELECTION===")
    # print(other_selection)
    
    selections = np.concatenate((selection, other_selection))
    kf_selections = np.sort(selections)
    # print("===KF SELECTIONS===")
    # print(kf_selections)
    
    summary = np.array([], dtype=np.int32)
    
    for kf_idx in kf_selections:
        min_idx = max(0, kf_idx - expansion)
        max_idx = min(video_length - 1, kf_idx + expansion)
        
        kf_summary = np.arange(min_idx, max_idx + 1)
        summary = np.union1d(summary, kf_summary)
    
    # print('===SUMMARY===')
    # print(summary)
    return summary


def evaluateSummary(scores, user_summary, keyframe_indices, segmentation,
                    coef, mode, fill_mode, expand):
    f_scores = []
    lengths = []
    summary_lengths = []
    video_length = len(user_summary)
    
    if mode == 'shot':
        segments = np.array([[segment['start'], segment['end'],
                              segment['num_frames']]
                             for segment in segmentation])
    
    for user in range(user_summary.shape[1]):
        user_selected = np.where(user_summary[:, user] > 0)[0]
        
        if mode == 'frame':
            length = len(user_selected)
            machine_selected = computeSummary(scores=scores,
                                              keyframe_indices=keyframe_indices,
                                              length=coef*length,
                                              video_length=video_length,
                                              expand=expand,
                                              )
            
            tp = len(np.intersect1d(machine_selected, user_selected))
            fp = len(np.setdiff1d(machine_selected, user_selected))
            fn = len(np.setdiff1d(user_selected, machine_selected))
        elif mode == 'fragment':
            user_fragments = np.unique(user_summary[:, user])
            length = len(user_fragments)
            machine_selected = computeSummary(scores=scores,
                                              keyframe_indices=keyframe_indices,
                                              length=coef*length,
                                              video_length=video_length,
                                              expand=expand,
                                              )
            
            intersected_indices = np.intersect1d(machine_selected,
                                                 user_selected)
            
            tp = len(np.unique(user_summary[intersected_indices, user]))
            fp = len(np.unique(user_summary[machine_selected, user])) - tp
            fn = len(np.unique(user_summary[user_selected, user])) - tp
        elif mode == 'shot':
            length = len(user_selected)
            
            machine_summary = calculateSummary(scores=scores,
                                               segments=segments,
                                               length=coef*length,
                                               video_length=video_length,
                                               fill_mode=fill_mode,
                                               expand=expand,
                                               )
            
            machine_selected = np.where(machine_summary > 0)[0]
            
            tp = len(np.intersect1d(machine_selected, user_selected))
            fp = len(np.setdiff1d(machine_selected, user_selected))
            fn = len(np.setdiff1d(user_selected, machine_selected))
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        f_scores.append(f_score)
        lengths.append(length)
        summary_lengths.append(len(machine_selected))
    
    # Maximum F-measure across all users
    f_score = max(f_scores)
    summary_length = lengths[np.argmax(f_scores)]
    
    return f_score, f_scores, lengths, summary_length, summary_lengths
