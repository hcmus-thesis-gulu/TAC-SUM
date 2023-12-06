import os
import time
import argparse
import numpy as np

from model.generator import Summarizer


def summarize_videos(embedding_folder, context_folder, summary_folder,
                     reduced_emb, scoring_mode, kf_mode, bias, key_length):
    summarizer = Summarizer(scoring_mode, kf_mode)
    
    for embedding_name in os.listdir(embedding_folder):
        file_end = '_reduced.npy' if reduced_emb else '_embeddings.npy'
        
        if embedding_name.endswith(file_end):
            filename = embedding_name[:-len(file_end)]
            print(f"Processing the context of video {filename}")
            
            embedding_name = os.path.join(embedding_folder, embedding_name)
            embeddings = np.load(embedding_name)
            print(f"The extracted context has {embeddings.shape[0]} embeddings")
            
            samples_file = filename + '_samples.npy'
            samples_path = os.path.join(embedding_folder, samples_file)
            samples = np.load(samples_path)
            
            segments_path = os.path.join(context_folder, filename + '_segments.npy')
            segments = np.load(segments_path)
            print(f"The extracted context has {segments.shape[0]} segments")
            
            scores_file = filename + '_scores.npy'
            keyframes_file = filename + '_keyframes.npy'
            
            scores_path = os.path.join(summary_folder, scores_file)
            keyframes_path = os.path.join(summary_folder, keyframes_file)
            
            print(f'Summarizing video {filename}')
            if os.path.exists(scores_path):
                continue
            
            scores = summarizer.score_segments(embeddings=embeddings,
                                               segments=segments,
                                               bias=bias)
            
            sampled_scores = [[sample, score]
                              for sample, score in zip(samples, scores)
                              ]
            
            # Sort by sample index
            sorted_scores = np.asarray(sorted(sampled_scores,
                                              key=lambda x: x[0]))
            np.save(scores_path, sorted_scores)
            
            if key_length >= 0:
                keyframe_indices = summarizer.select_keyframes(segments,
                                                               scores,
                                                               key_length)
                
                print(f'Selected {len(keyframe_indices)} keyframes')
                
                keyframe_idxs = np.asarray([samples[idx]
                                            for idx in keyframe_indices])
                np.save(keyframes_path, np.sort(keyframe_idxs))


def main():
    parser = argparse.ArgumentParser(description='Generate Summaries from Partitioned Selections of Videos.')
    
    parser.add_argument('--embedding-folder', type=str, required=True,
                        help='path to folder containing feature files')
    parser.add_argument('--context-folder', type=str, required=True,
                        help='path to output folder for clustering')
    parser.add_argument('--summary-folder', type=str, required=True,
                        help='path to output folder for summaries')
    
    parser.add_argument('--scoring-mode', type=str, default='mean',
                        choices=['mean', 'middle', 'uniform'],
                        help='Method of representing segments')
    parser.add_argument('--kf-mode', type=str, default='mean',
                        help='Method of representing segments')
    parser.add_argument('--reduced-emb', action='store_true',
                        help='Use reduced embeddings or not')
    parser.add_argument('--bias', type=float, default=0.5,
                        help='Bias for frames near the keyframes')
    
    # How many keyframes to select
    parser.add_argument('--key-length', type=int, default=-1,
                        help="Maximum number of keyframes to select, "
                        + "-1 to not select, 0 to auto select")
    
    args = parser.parse_args()

    summarize_videos(embedding_folder=args.embedding_folder,
                     context_folder=args.context_folder,
                     summary_folder=args.summary_folder,
                     reduced_emb=args.reduced_emb,
                     scoring_mode=args.scoring_mode,
                     kf_mode=args.kf_mode,
                     bias=args.bias,
                     key_length=args.key_length
                     )


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
