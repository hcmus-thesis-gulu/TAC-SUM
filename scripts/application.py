import os
import numpy as np
from tqdm import tqdm
import cv2 as cv

from torchvision.transforms import ToTensor
from PIL import Image

from model.embedder import Embedder
from model.propogator import Clusterer
from model.selector import Selector
from model.generator import Summarizer

from model.utils import count_frames, calculate_num_clusters


class VidSum():
    """VidSum Video Summarization class
    Perform the full process of summarizing the given video
    """
    def __init__(self):
        self.embedder = Embedder(model_type='clip',
                                 representation='cls',
                                 model_kind='base',
                                 patch=32,
                                 device='cuda')
        
        self.input_frame_rate = None
        
        self.method = None
        self.distance = None
        self.max_length = None
        self.modulation = None
        self.intermediate_components = None
        self.window_size = None
        self.min_seg_length = None
        
        self.reduced_emb = None
        self.scoring_mode = None
        self.kf_mode = None
        self.bias = None
        
        self.output_frame_rate = None
        self.max_length = None
        self.sum_rate = None
        self.extension = None
        
        self.codec_dict = {
            'mp4': 'mp4v',
            'avi': 'DIVX',
            'webm': 'VP09'
        }

    def set_params(self, input_frame_rate, method, distance, max_length,
                   modulation, intermediate_components, window_size,
                   min_seg_length, reduced_emb, scoring_mode, kf_mode,
                   bias, output_frame_rate, sum_rate, extension):
        self.input_frame_rate = int(input_frame_rate)
        
        self.method = method
        self.distance = distance
        self.max_length = int(max_length)
        self.modulation = float(10.0 ** float(modulation))
        self.intermediate_components = int(intermediate_components)
        self.window_size = int(window_size)
        self.min_seg_length = int(min_seg_length)
        
        self.reduced_emb = bool(reduced_emb)
        self.scoring_mode = scoring_mode
        self.kf_mode = kf_mode
        self.bias = float(bias)
        
        self.output_frame_rate = int(output_frame_rate)
        self.sum_rate = sum_rate
        self.extension = extension.lower()
    
    def generate_context(self, video_path):
        # Define transformations
        transform = ToTensor()
        
        # Get file path
        print(f'Extracting features at {video_path}')
        
        # Extract features for each frame of the video
        cap = cv.VideoCapture(video_path)
        # Get the video's frame rate, total frames
        fps, total_frames = count_frames(video_path)
        
        if self.input_frame_rate is None:
            self.input_frame_rate = fps
        
        # Calculate the total number of samples
        frame_step = fps // self.input_frame_rate
        total_samples = (total_frames + frame_step - 1) // frame_step
        
        # Create holders
        embeddings = []
        samples = []

        pbar = tqdm(total=total_samples)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step:
                frame_idx += 1
                continue
            
            # Convert frame to PyTorch tensor and extract features
            img = Image.fromarray(frame, mode="RGB")
            img = transform(img).unsqueeze(0)
            embedding = self.embedder.image_embedding(img)
            
            embeddings.append(embedding)
            samples.append(frame_idx)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        return np.vstack(embeddings), np.asarray(samples, dtype=np.int32)
    
    def localize_video(self, embeddings):
        num_clusters = calculate_num_clusters(num_frames=embeddings.shape[0],
                                              max_length=self.max_length,
                                              modulation=self.modulation)

        clusterer = Clusterer(method=self.method, distance=self.distance,
                              num_clusters=num_clusters,
                              embedding_dim=embeddings.shape[1],
                              intermediate_components=self.intermediate_components)
        
        selector = Selector(window_size=self.window_size,
                            min_seg_length=self.min_seg_length)
        
        labels, reduced_embeddings = clusterer.cluster(embeddings)
        
        segments = selector.select(labels)
        
        print(f'Number of segments: {len(segments)}')
        print(f'Number of clusters: {clusterer.num_clusters}')
        
        return segments, labels, reduced_embeddings
    
    def generate_summary(self, embeddings, samples, segments):
        summarizer = Summarizer(scoring_mode=self.scoring_mode,
                                kf_mode=self.kf_mode)
        
        scores = summarizer.score_segments(embeddings=embeddings,
                                           segments=segments,
                                           bias=self.bias)
        
        sampled_scores = [[sample, score]
                          for sample, score in zip(samples, scores)]
        
        # Sort by sample index
        sorted_scores = np.asarray(sorted(sampled_scores,
                                          key=lambda x: x[0]))
        
        keyframe_indices = summarizer.select_keyframes(segments,
                                                       scores,
                                                       0)
        
        print(f'Selected {len(keyframe_indices)} keyframes')
        
        keyframe_idxs = np.asarray([samples[idx]
                                    for idx in keyframe_indices])
            
        return sorted_scores, keyframe_idxs

    def generate_video(self, output_video_path, input_video_path,
                       frame_indices):
        raw_video = cv.VideoCapture(input_video_path)
        width = int(raw_video.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(raw_video.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        computed_fps, video_length = count_frames(input_video_path)
    
        if self.output_frame_rate is None:
            fps = computed_fps
        else:
            fps = self.output_frame_rate
        
        print(f'FPS: {fps}')
        
        # Maximum nunmber of frames in the summary
        frames_length = int(self.max_length * fps)
        
        # Estimated number of frames in the summary
        estimated_length = int(video_length * self.sum_rate)
        
        # Final number of frames of the summary
        summary_length = max(min(frames_length, estimated_length),
                             len(frame_indices))
        print(f'Frames in the summary: {summary_length}')
        
        # Length of the fragment around each keyframe
        fragment_length = summary_length // len(frame_indices)
        print(f'Length of the fragment: {fragment_length}')
        
        # Fragment width of the computed fragment length
        fragment_width = max(0, (fragment_length - 1) // 2)
        print(f'Width of fragments: {fragment_width}')
        
        output_codec = self.codec_dict[self.extension]
        fourcc = cv.VideoWriter_fourcc(*output_codec)
        video = cv.VideoWriter(output_video_path, fourcc,
                               float(fps), (width, height))
        
        cur_idx = 0
        pbar = tqdm(total=len(frame_indices))
        kf_idx = 0
        
        while True:
            ret, frame = raw_video.read()
            if not ret:
                break
            
            while kf_idx < len(frame_indices) and frame_indices[kf_idx] < cur_idx - fragment_width:
                kf_idx += 1
            if kf_idx < len(frame_indices) and abs(frame_indices[kf_idx] - cur_idx) <= fragment_width:
                video.write(frame)
            
            if cur_idx in frame_indices:
                pbar.update(1)
            
            cur_idx += 1
        
        raw_video.release()
        video.release()
        pbar.close()

    def store_result(self, video_path, output_folder, data):
        video_name = os.path.basename(video_path).split('.')[0]
        video_folder = os.path.join(output_folder, video_name)
        
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
            
        # Generate video
        output_video_name = f'{video_name}_summary.{self.extension}'
        output_video_path = os.path.join(video_folder,
                                         output_video_name)
        
        self.generate_video(output_video_path, video_path,
                            data['keyframe_idxs'])
        
        # Save scores
        scores_path = os.path.join(video_folder,
                                   video_name + '_scores.npy')
        np.save(scores_path, data['scores'])
        
        # Save segments
        segments_path = os.path.join(video_folder,
                                     video_name + '_segments.npy')
        np.save(segments_path, data['segments'])
        
        # Save labels
        labels_path = os.path.join(video_folder,
                                   video_name + '_labels.npy')
        np.save(labels_path, data['labels'])
        
        # Save reduced embeddings
        reduced_embeddings_path = os.path.join(video_folder,
                                               video_name + '_reduced_embeddings.npy')
        np.save(reduced_embeddings_path, data['reduced_embeddings'])
        
        return output_video_path
        
    def summarize(self, video_path, output_folder):
        data = {}
        
        # Generate context
        data['embeddings'], data['samples'] = self.generate_context(video_path)
        
        # Localize video
        local_information = self.localize_video(data['embeddings'])
        data['segments'], data['labels'], data['reduced_embeddings'] = local_information
        
        # Generate summary
        embedding = data['reduced_embeddings'] if self.reduced_emb else data['embeddings']
        summary = self.generate_summary(embedding,
                                        data['samples'],
                                        data['segments'])
        data['scores'], data['keyframe_idxs'] = summary
        
        # Generate video
        output_video_path = self.store_result(video_path, output_folder, data)
        
        return output_video_path
        