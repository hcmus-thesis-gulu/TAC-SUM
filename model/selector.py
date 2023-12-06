import numpy as np


class Selector:
    def __init__(self, window_size=5, min_seg_length=10):
        self.window_size = window_size
        self.min_seg_length = min_seg_length

    # Segment the video into shots based on the smoothed labels
    def select(self, labels):
        segments = []   # List of (label, start, end) tuples
        start = 0
        current_label = None
        window_size = min(self.window_size, len(labels))

        for i in range(len(labels)):
            # Smooth the labels by taking the majority label in a window
            # whose length is at least window_size
            if i < (window_size // 2):
                left = 0
                right = window_size
            elif i >= len(labels) - (window_size // 2):
                left = len(labels) - window_size
                right = len(labels)
            else:
                left = i - (window_size // 2)
                right = i + (window_size // 2) + 1
            
            window = labels[left:right]
            
            label = np.bincount(window).argmax()
            
            # Partition the video into segments based on the label
            if i == 0:
                current_label = label
            elif i == len(labels) - 1:
                segments.append((current_label, start, i+1))
            elif label != current_label:
                # Handle short segments
                if len(segments) > 0 and i - start < self.min_seg_length:
                    current_label = label
                    
                    # If go back to previous segment,
                    # the short one is relabeled with the previous label
                    if segments[-1][0] == label:
                        # segments[-1][2] = i
                        segments[-1] = (segments[-1][0], segments[-1][1], i)
                        start = i
                    
                    # If another segment encountered, divide the segment into two,
                    # and add the first half to the previous segment
                    # while keeping the second one
                    else:
                        middle = (start + i) // 2
                        # segments[-1][2] = middle
                        segments[-1] = (segments[-1][0], segments[-1][1],
                                        middle)
                        start = middle
                
                # Add the segment to the list of segments
                else:
                    segments.append((current_label, start, i))
                    start = i
                    current_label = label
        
        # Post process the segments to merge consecutive segments with the same label
        post_segments = []
        current_label = None
        for label, start, end in segments:
            if current_label is None:
                current_label = label
                post_segments.append((current_label, start, end))
            elif label == current_label:
                post_segments[-1] = (current_label, post_segments[-1][1], end)
            else:
                current_label = label
                post_segments.append((current_label, start, end))
        
        return np.asarray(post_segments)
