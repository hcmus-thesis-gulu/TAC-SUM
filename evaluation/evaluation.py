import os
import argparse
import time
import numpy as np
import json

from scipy.io import loadmat
import h5py
from evaluator import evaluateSummary


def evaluateSummaries(groundtruth_folder, summary_folder, result_folder,
                      coef, mode, expand):
    assert mode != 'shot', 'Shot-based evaluation is not supported yet.'
    f_measures = {}
    
    for groundtruth_name in os.listdir(groundtruth_folder):
        if groundtruth_name.endswith('.mat'):
            filename = os.path.splitext(groundtruth_name)[0]
            groundtruth_path = os.path.join(groundtruth_folder,
                                            groundtruth_name)
            groundtruth = loadmat(groundtruth_path)
            
            user_summary = groundtruth['user_score']
            scores_path = os.path.join(summary_folder,
                                       filename + '_scores.npy')
            keyframes_path = os.path.join(summary_folder,
                                          filename + '_keyframes.npy')
            
            if os.path.exists(scores_path):
                scores = np.load(scores_path)
                if mode == 'shot':
                    keyframe_indices = None
                else:
                    keyframe_indices = np.load(keyframes_path)
                
                eval_results = evaluateSummary(scores=scores,
                                               user_summary=user_summary,
                                               keyframe_indices=keyframe_indices,
                                               segmentation=None,
                                               coef=coef,
                                               mode=mode,
                                               fill_mode=None,
                                               expand=expand
                                               )
                
                f_score, f_scores, lengths, summary_length, summary_lengths = eval_results
                
                f_measures[filename] = {
                    'f_score': f_score,
                    'f_scores': f_scores,
                    'lengths': lengths,
                    'summary_length': summary_length,
                    'summary_lengths': summary_lengths,
                    'video_length': len(user_summary),
                    'summarized_rate': summary_length / len(user_summary)
                }
                
                print(f'F-measure of {filename} is {f_score} and summarized '
                      + f'rate is {summary_length / len(user_summary)}')
    
    if f_measures:
        f_measure = np.mean([f_measures[key]['f_score']
                             for key in f_measures])
        
        mean_sum_rate = np.mean([f_measures[key]['summarized_rate']
                                 for key in f_measures])
        std_sum_rate = np.std([f_measures[key]['summarized_rate']
                               for key in f_measures])
        
        results = {
            'f_measures': f_measures,
            'average_f_measure': f_measure,
            'dist_summarized_rate': {
                'mean': mean_sum_rate,
                'std': std_sum_rate,
            }
        }
        
        print(f'Average F-measure: {f_measure:.4f}')
        print(f"Average summarized rate: {mean_sum_rate:.4f} ± {std_sum_rate:.4f}")
        
        top_5 = sorted(f_measures.items(), key=lambda x: x[1]['f_score'],
                       reverse=True)[:5]        
        results['top_5'] = {
            top_5[i][0]: top_5[i][1]['f_score'] for i in range(5)
        }
        
        top_5_avg = np.mean([top_5[i][1]['f_score'] for i in range(5)])
        results['top_5_avg'] = top_5_avg
        
        print(f"Top 5 average F-measure: {top_5_avg:.4f}")
        
    if result_folder is not None:
        json_file = os.path.join(result_folder, 'results.json')
        with open(json_file, 'w', encoding='utf-8') as file:
            json.dump(results, file)
        
    return None, f_measure, top_5_avg


def testSummaries(groundtruth_folder, summary_folder, result_folder,
                  coef, mode, fill_mode, expand):
    assert mode != 'fragment', 'Fragment-based testing is not supported yet.'
    
    groundtruth_file = 'eccv16_dataset_summe_google_pool5.h5'
    groundtruth = h5py.File(groundtruth_folder + '/' + groundtruth_file, 'r')
    
    splits_file = 'summe_splits.json'
    splits_path = os.path.join(groundtruth_folder, splits_file)
    splits = json.load(open(splits_path, 'r', encoding='utf-8'))
    
    segmentations_file = 'segmentations.json'
    segmentations_path = os.path.join(groundtruth_folder, segmentations_file)
    segmentations = json.load(open(segmentations_path, 'r', encoding='utf-8'))
    
    test_keys = []
    for split in splits:
        test_key = split['test_keys']
        test_keys.extend(test_key)
        
    # Deduplicate
    test_keys = list(set(test_keys))
    f_measures = {}
    
    for test_key in test_keys:
        filename = np.array(groundtruth[test_key +
                                        '/video_name']).astype(str).item()
        user_summary = np.array(groundtruth[test_key + '/user_summary']).T
        
        scores_file = filename + '_scores.npy'
        keyframes_file = filename + '_keyframes.npy'
        
        scores_path = os.path.join(summary_folder, scores_file)
        keyframes_path = os.path.join(summary_folder, keyframes_file)
        
        segmentation_info = segmentations[test_key]
        segmentation = segmentation_info['segments']
        
        if os.path.exists(scores_path):
            scores = np.load(scores_path)
            if mode == 'shot':
                keyframe_indices = None
            else:
                keyframe_indices = np.load(keyframes_path)
            
            eval_results = evaluateSummary(scores=scores,
                                           user_summary=user_summary,
                                           keyframe_indices=keyframe_indices,
                                           segmentation=segmentation,
                                           coef=coef,
                                           mode=mode,
                                           fill_mode=fill_mode,
                                           expand=expand
                                           )
            
            f_score, f_scores, lengths, summary_length, summary_lengths = eval_results
            
            f_measures[test_key] = {
                'f_score': f_score,
                'f_scores': f_scores,
                'lengths': lengths,
                'summary_length': summary_length,
                'summary_lengths': summary_lengths,
                'video_length': len(user_summary),
                'summarized_rate': summary_length / len(user_summary)
            }
            
            print(f'F-measure of {filename} is {f_score} and summarized '
                  + f'rate is {summary_length / len(user_summary)}')
    
    if f_measures:
        f_measure = np.mean([f_measures[key]['f_score']
                             for key in f_measures])
        
        mean_sum_rate = np.mean([f_measures[key]['summarized_rate']
                                 for key in f_measures])
        std_sum_rate = np.std([f_measures[key]['summarized_rate']
                               for key in f_measures])
        
        results = {
            'f_measures': f_measures,
            'average_f_measure': f_measure,
            'dist_summarized_rate': {
                'mean': mean_sum_rate,
                'std': std_sum_rate,
            }
        }
        
        print(f'Average F-measure: {f_measure:.4f}')
        print(f"Average summarized rate: {mean_sum_rate:.4f} ± {std_sum_rate:.4f}")
        
        top_5 = sorted(f_measures.items(), key=lambda x: x[1]['f_score'],
                       reverse=True)[:5]        
        results['top_5'] = {
            top_5[i][0]: top_5[i][1]['f_score'] for i in range(5)
        }
        
        top_5_avg = np.mean([top_5[i][1]['f_score'] for i in range(5)])
        results['top_5_avg'] = top_5_avg
        
        print(f"Top 5 average F-measure: {top_5_avg:.4f}")
        
        max_f_measure = 0
        avg_f_measure = 0
        
        # Split-wise results
        for idx, split in enumerate(splits):
            test_keys = split['test_keys']
            f_measures_split = {key: f_measures[key]
                                for key in f_measures if key in test_keys}
            
            if f_measures_split:
                split_f_measure = np.mean([f_measures_split[key]['f_score']
                                           for key in f_measures_split])
                
                split_mean_sum_rate = np.mean([f_measures_split[key]['summarized_rate']
                                               for key in f_measures_split])
                split_std_sum_rate = np.std([f_measures_split[key]['summarized_rate']
                                             for key in f_measures_split])
                
                results[f"split_{idx}"] = {
                    'f_measures': f_measures_split,
                    'average_f_measure': split_f_measure,
                    'dist_summarized_rate': {
                        'mean': split_mean_sum_rate,
                        'std': split_std_sum_rate,
                    }
                }
                
                print(f"Split {idx} - Average F-measure: {split_f_measure:.4f}")
                print(f"Split {idx} - Average summarized rate: "
                      + f"{split_mean_sum_rate:.4f} ± {split_std_sum_rate:.4f}")
                
                max_f_measure = max(max_f_measure, split_f_measure)
                avg_f_measure += split_f_measure

        avg_f_measure /= len(splits)
        print(f'Maximum F-measure: {max_f_measure:.4f}')
        print(f'Average F-measure: {avg_f_measure:.4f}')
    
    if result_folder is not None:
        json_file = os.path.join(result_folder, 'results.json')
        with open(json_file, 'w', encoding='utf-8') as file:
            json.dump(results, file)
        
    return max_f_measure, avg_f_measure, top_5_avg


def evaluate():
    parser = argparse.ArgumentParser(description='Evaluate machine learning algorithm summaries.')
    
    parser.add_argument('--original', action='store_true',
                        help='Evaluate the original dataset'
                        )
    # parser.add_argument('--original-folder', type=str,
    #                     help='Path to the folder containing the original dataset'
    #                     )
    
    parser.add_argument('--groundtruth-folder', type=str, required=True,
                        help='Path to the folder containing groundtruth .mat files')
    parser.add_argument('--summary-folder', type=str, required=True,
                        help='Path to the folder containing output .npy files')
    parser.add_argument('--result-folder', type=str, default='result',
                        help='Path to the folder containing the result of evaluation')
    
    parser.add_argument('--mode', type=str, default='frame',
                        choices=['frame', 'fragment', 'shot'],
                        help='Evaluation mode: "frame" or "fragment"')
    parser.add_argument('--coef', type=float, default=2.0,
                        help='Coefficient for taking more frames'
                        + ', 0 to get all keyframes')
    parser.add_argument('--expand', type=float, default=0,
                        help='Int >= 1: Expand around keyframes; '
                        + '0 < Float < 1: Default percentage of selection')
    parser.add_argument('--fill-mode', type=str, default='linear',
                        choices=['linear', 'nearest', 'nearest-up'],
                        help='Mode for filling in missing frames: '
                        + '"linear", "nearest", or "nearest-up"')
    
    args = parser.parse_args()
    
    if args.original:
        evaluateSummaries(groundtruth_folder=args.groundtruth_folder,
                          summary_folder=args.summary_folder,
                          result_folder=args.result_folder,
                          coef=args.coef,
                          mode=args.mode,
                          expand=args.expand
                          )
    else:
        testSummaries(groundtruth_folder=args.groundtruth_folder,
                      summary_folder=args.summary_folder,
                      result_folder=args.result_folder,
                      coef=args.coef,
                      mode=args.mode,
                      fill_mode=args.fill_mode,
                      expand=args.expand
                      )


if __name__ == '__main__':
    start_time = time.time()
    evaluate()
    end_time = time.time()
    
    print(f'Time taken: {end_time - start_time:.2f} seconds')
