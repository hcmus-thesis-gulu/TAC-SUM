import os
import argparse
import numpy as np
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

from model.reducer import Reducer


def visualize_cluster(video_folder, embedding_folder,
                      context_folder, video_name,
                      color_value, show_image=False):
    sample_file = os.path.join(embedding_folder, f'{video_name}_samples.npy')
    embedding_file = os.path.join(embedding_folder, f'{video_name}.npy')
    keyframe_file = os.path.join(context_folder, f'{video_name}_keyframes.npy')
    label_file = os.path.join(context_folder, f'{video_name}_labels.npy')
    reduced_file = os.path.join(context_folder, f'{video_name}_reduced.npy')
    
    video_file = os.path.join(video_folder, f'{video_name}.mp4')
    
    # try:
    sample_idxs = np.load(sample_file)
    keyframe_idxs = np.load(keyframe_file)
    labels = np.load(label_file)
    embeddings = np.load(embedding_file)
    reduced_embeddings = np.load(reduced_file)
    
    # Fit and transform the data
    reducer = Reducer(intermediate_components=None)
    _, reduced_embeddings = reducer.reduce(embeddings)
    
    # Plot the transformed data
    fig, ax = plt.subplots()
    ax.margins(tight=True)
    
    if color_value == 'index':
        color, label = sample_idxs, "Sample indices"
    else:
        color, label = labels, "Cluster labels"
    
    sc = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                    c=color, cmap='rainbow', alpha=0.6)
    ax.set_ylabel('1st t-SNE dim')
    ax.set_xlabel('2nd t-SNE dim')
    cbar = fig.colorbar(sc)
    cbar.set_label(label)
    
    if show_image:
        video = cv.VideoCapture(video_file)
        
        frame_idx = 0
        embedding_idx = 0
        pbar = tqdm(total=len(sample_idxs))
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            if frame_idx in sample_idxs:
                props = dict(edgecolor='red', linewidth=1)
                if frame_idx not in keyframe_idxs:
                    # frame = cv.cvtColor(frame,
                    #                     cv.COLOR_BGR2GRAY
                    #                     )
                    props = None
                
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(frame, zoom=0.02),
                                                    reduced_embeddings[embedding_idx],
                                                    bboxprops=props,
                                                    pad=0.1
                                                    )
                
                ax.add_artist(imagebox)
                embedding_idx += 1
                pbar.update(1)
            
            frame_idx += 1
        pbar.close()
        
    plt.show()
    # except Exception as e:
    #   print(e)


def main():
    parser = argparse.ArgumentParser(description='Visualize result')
    parser.add_argument('--video-folder', type=str, required=True,
                        help='Path to folder containing videos')
    parser.add_argument('--embedding-folder', type=str, required=True,
                        help='path to folder containing feature files')
    parser.add_argument('--context-folder', type=str, required=True,
                        help='path to output folder for clustering')
    parser.add_argument('--demo-folder', type=str, required=True,
                        help='path to folder saving demo videos')
    parser.add_argument('--video-name', type=str, help='video name')
    
    parser.add_argument('--show-image', action='store_true',
                        help='show image in cluster')
    parser.add_argument('--color-value', type=str, default='index',
                        choices=['index', 'label'],
                        help='color value')

    args = parser.parse_args()
    
    visualize_cluster(video_folder=args.video_folder,
                      embedding_folder=args.embedding_folder,
                      context_folder=args.context_folder,
                      video_name=args.video_name,
                      show_image=args.show_image,
                      color_value=args.color_value
                      )


if __name__ == '__main__':
    main()
