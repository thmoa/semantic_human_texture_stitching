#!/usr/bin/env bash
echo "Step 1: make unwraps..."
mkdir -p data/sample/unwraps
python2 step1_make_unwraps.py data/sample/frame_data.pkl data/sample/frames data/sample/segmentations data/sample/unwraps
echo "Step 2: make priors..."
python2 step2_segm_vote_gmm.py data/sample/unwraps data/sample/segm.png data/sample/gmm.pkl
echo "Step 3: stitch texture..."
python2 step3_stitch_texture.py data/sample/unwraps data/sample/segm.png data/sample/gmm.pkl data/sample/sample_texture.jpg