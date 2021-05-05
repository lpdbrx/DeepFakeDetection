#!/usr/bin/env bash

echo ""
echo "-------------------------------------------------"
echo "| Index DFDC dataset                            |"
echo "-------------------------------------------------"
# put your dfdc source directory path and uncomment the following line
DFDC_SRC= "/Users/lpdbrx/Desktop/EPFL/MA2c/DeepFakeDetection.nosync/icpr2020dfdc-master/data/datasets/dfdc2/train_sample_videos"
python index_dfdc.py --source $DFDC_SRC

echo ""
echo "-------------------------------------------------"
echo "| Index FF dataset                              |"
echo "-------------------------------------------------"
# put your ffpp source directory path and uncomment the following line
FFPP_SRC= "/Users/lpdbrx/Desktop/EPFL/MA2c/DeepFakeDetection.nosync/icpr2020dfdc-master/data/datasets/ffpp2"
python index_ffpp.py --source $FFPP_SRC


echo ""
echo "-------------------------------------------------"
echo "| Extract faces from DFDC                        |"
echo "-------------------------------------------------"
# put your source and destination directories and uncomment the following lines
# DFDC_SRC=/your/dfdc/source/folder
# VIDEODF_SRC=/previously/computed/index/path
# FACES_DST=/faces/output/directory
# FACESDF_DST=/faces/df/output/directory
# CHECKPOINT_DST=/tmp/per/video/outputs

# python extract_faces.py
# --source /Users/lpdbrx/Desktop/EPFL/MA2c/DeepFakeDetection.nosync/icpr2020dfdc-master/data/datasets/dfdc2/train_sample_videos
# --videodf /Users/lpdbrx/Desktop/EPFL/MA2c/DeepFakeDetection.nosync/icpr2020dfdc-master/data/dfdc_videos.pkl
# --facesfolder /Users/lpdbrx/Desktop/EPFL/MA2c/DeepFakeDetection.nosync/icpr2020dfdc-master/output/dfdc/faces
# --facesdf /Users/lpdbrx/Desktop/EPFL/MA2c/DeepFakeDetection.nosync/icpr2020dfdc-master/output/dfdc/dfs
# --checkpoint /Users/lpdbrx/Desktop/EPFL/MA2c/DeepFakeDetection.nosync/icpr2020dfdc-master/output/dfdc/checkpoints

python extract_faces.py \
--source $DFDC_SRC \
--videodf $VIDEODF_SRC \
--facesfolder $FACES_DST \
--facesdf $FACESDF_DST \
--checkpoint $CHECKPOINT_DST

echo ""
echo "-------------------------------------------------"
echo "| Extract faces from FF                         |"
echo "-------------------------------------------------"
# put your source and destination directories and uncomment the following lines
# FFPP_SRC=/your/dfdc/source/folder
# VIDEODF_SRC=/previously/computed/index/path
# FACES_DST=/faces/output/directory
# FACESDF_DST=/faces/df/output/directory
# CHECKPOINT_DST=/tmp/per/video/outputs
python extract_faces.py \
--source $FFPP_SRC \
--videodf $VIDEODF_SRC \
--facesfolder $FACES_DST \
--facesdf $FACESDF_DST \
--checkpoint $CHECKPOINT_DST