#!/bin/bash

# Check if video folder path and output folder path are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 video_folder_path output_folder_path video_name"
    exit 1
fi

# Create features and clustering folders inside output folder
mkdir -p "$2/demo"

# Run feature extraction script with arguments
python visualizer.py \
--video-folder $1 \
--embedding-folder "$2/embeddings" \
--clustering-folder "$2/contexts" \
--demo-folder "$2/demo" \
--video-name "$3" \
--visual-type cluster \
--output-fps 4 \
--intermediate-components 50 \
--color-value label
