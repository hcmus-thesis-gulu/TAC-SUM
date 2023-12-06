#!/bin/bash

# Check if video folder path and output folder path are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 video_folder_path output_folder_path"
    exit 1
fi

# Create embeddings and contexts folders inside output folder
mkdir -p "$2/embeddings" "$2/contexts"

# Run context extraction script with arguments
python scripts/context.py \
--video-folder "$1" \
--embedding-folder "$2/embeddings" \
--frame-rate 4 \
--representation cls

# Run semantic extraction script with arguments
python scripts/extraction.py \
--embedding-folder "$2/embeddings" \
--context-folder "$2/contexts" \
--method ours \
--num-clusters 6 \
--distance cosine \
--embedding-dim 3 \
--window-size 5 \
--min-seg-length 3 \
--modulation 1e-4
