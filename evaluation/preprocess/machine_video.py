import os
import argparse
import numpy as np
import json

from preprocess.utils import broadcast_video


# Adopt shot-based video materialization also for machine-generated summary
def materialize_video(video_folder, summary_folder, demo_folder, video_name,
                      segmentation, max_length, sum_rate, fps,
                      extension):
    raw_video_path = os.path.join(video_folder, f'{video_name}.mp4')

    if segmentation is not None:
        scores_file = f'{video_name}_scores.npy'
        scores_path = os.path.join(summary_folder, scores_file)
        scores = np.load(scores_path)
        input = scores

        segments = np.array([[segment['start'], segment['end'],
                              segment['num_frames']]
                             for segment in segmentation])

        output_file = f'{video_name}_shots.{extension}'
        output_path = os.path.join(demo_folder, output_file)
    else:
        keyframes_file = f'{video_name}_keyframes.npy'
        keyframes_path = os.path.join(summary_folder, keyframes_file)
        keyframes = np.load(keyframes_path)
        input = keyframes

        segments = None

        output_file = f'{video_name}_keyframes.{extension}'
        output_path = os.path.join(demo_folder, output_file)

    broadcast_video(input_video_path=raw_video_path,
                    input=input,
                    output_video_path=output_path,
                    segments=segments,
                    max_length=max_length,
                    sum_rate=sum_rate,
                    fps=fps,
                    extension=extension
                    )


def materialize_videos(video_folder, summary_folder, demo_folder, shot,
                       max_length, sum_rate, fps=None, extension='webm'):
    video_files = os.listdir(video_folder)

    if shot is not None:
        print(f'Shot-based summary with shots from {shot}')
        segmentations_file = 'segmentations.json'
        segmentations_path = os.path.join(shot, segmentations_file)
        segmentations = json.load(open(segmentations_path, 'r',
                                       encoding='utf-8'))

        segments = {
            segmentation['video_name']: segmentation['segments']
            for segmentation in segmentations.values()
        }

    for video_file in video_files:
        if video_file.endswith('.mp4'):
            video_name = video_file.split('.')[0]
            print(f'Processing {video_name}')

            materialize_video(video_folder=video_folder,
                              summary_folder=summary_folder,
                              demo_folder=demo_folder,
                              video_name=video_name,
                              segmentation=segments[video_name] if shot else None,
                              max_length=max_length,
                              sum_rate=sum_rate,
                              fps=fps,
                              extension=extension
                              )


def main():
    parser = argparse.ArgumentParser(description='Visualize result')
    parser.add_argument('--video-folder', type=str, required=True,
                        help='Path to folder containing videos')
    parser.add_argument('--summary-folder', type=str, required=True,
                        help='path to output folder for clustering')
    parser.add_argument('--demo-folder', type=str, required=True,
                        help='path to folder saving demo videos')

    parser.add_argument('--shot', type=str, default=None,
                        help='Path to folder containing shots for '
                        + 'shot-based summary')
    parser.add_argument('--output-fps', type=int, help='video fps')
    parser.add_argument('--max-length', type=int, default=30,
                        help='maximum length of output video (in seconds)')
    parser.add_argument('--sum-rate', type=float, default=0.15,
                        help='rate of summary video (0 < rate < 1)')
    parser.add_argument('--extension', type=str, default='webm',
                        help='extension of output video')

    args = parser.parse_args()

    materialize_videos(video_folder=args.video_folder,
                       summary_folder=args.summary_folder,
                       demo_folder=args.demo_folder,
                       shot=args.shot,
                       max_length=args.max_length,
                       sum_rate=args.sum_rate,
                       fps=args.output_fps,
                       extension=args.extension
                       )


if __name__ == '__main__':
    main()
