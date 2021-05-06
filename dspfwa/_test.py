# """
# Dual Spatial Pyramid for exposing face warp artifacts in DeepFake videos (DSP-FWA)
# """
# import torch
# import torch.nn.functional as F
# import cv2, os, dlib
# import numpy as np
# from py_utils.face_utils import lib
# from py_utils.vid_utils import proc_vid as pv
# from py_utils.DL.pytorch_utils.models.classifier import SPPNet
# import glob
# from tqdm import tqdm
# import pandas as pd

# sample_num = 10
# # Employ dlib to extract face area and landmark points
# pwd = os.path.dirname(os.path.abspath('dspfwa'))
# front_face_detector = dlib.get_frontal_face_detector()
# lmark_predictor = dlib.shape_predictor(pwd + '/dlib_model/shape_predictor_68_face_landmarks.dat')

# ds = 'celebdf'
# ds_ = 'celebdf-35-5-10' #'ffpp-c23-720-140-140'

# def im_test(net, im):
#     face_info = lib.align(im[:, :, (2,1,0)], front_face_detector, lmark_predictor)
#     # Samples
#     if len(face_info) != 1:
#         prob = -1
#     else:
#         _, point = face_info[0]
#         rois = []
#         for i in range(sample_num):
#             roi, _ = lib.cut_head([im], point, i)
#             rois.append(cv2.resize(roi[0], (224, 224)))

#         # vis_ = np.concatenate(rois, 1)
#         # cv2.imwrite('vis.jpg', vis_)

#         bgr_mean = np.array([103.939, 116.779, 123.68])
#         bgr_mean = bgr_mean[np.newaxis, :, np.newaxis, np.newaxis]
#         bgr_mean = torch.from_numpy(bgr_mean).float().cuda()

#         rois = torch.from_numpy(np.array(rois)).float().cuda()
#         rois = rois.permute((0, 3, 1, 2))
#         prob = net(rois - bgr_mean)
#         prob = F.softmax(prob, dim=1)
#         prob = prob.data.cpu().numpy()
#         prob = 1 - np.mean(np.sort(prob[:, 0])[np.round(sample_num / 2).astype(int):])
#     return prob, face_info

# def draw_face_score(im, face_info, prob):
#     if len(face_info) == 0:
#         return im

#     _, points = np.array(face_info[0])
#     x1 = np.min(points[:, 0])
#     x2 = np.max(points[:, 0])
#     y1 = np.min(points[:, 1])
#     y2 = np.max(points[:, 1])

#     # Real: (0, 255, 0), Fake: (0, 0, 255)
#     color = (0, prob * 255, (1 - prob) * 255)
#     cv2.rectangle(im, (x1, y1), (x2, y2), color, 10)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(im, '{:.3f}'.format(prob), (x1, y1 - 10), font, 1, color, 3, cv2.LINE_AA)
#     return im

#     num_class = 2
# net = SPPNet(backbone=50, num_class=2)
# net = net.cuda()
# net.eval()
# model_path = os.path.join('ckpt/', 'SPP-res50.pth')
# if os.path.isfile(model_path):
#     print("=> loading checkpoint '{}'".format(model_path))
#     checkpoint = torch.load(model_path)#, map_location=torch.device('cpu'))
#     start_epoch = checkpoint['epoch']
#     net.load_state_dict(checkpoint['net'])
#     print("=> loaded checkpoint '{}' (epoch {})"
#           .format(model_path, start_epoch))
# else:
#     raise ValueError("=> no checkpoint found at '{}'".format(model_path))

# print(torch.load("../icpr2020dfdc/weights/binclass/net-EfficientNetB4_traindb-"+ds_+"_face-scale_size-224_seed-41_ENB4/bestval.pth", map_location=torch.device('cpu')).keys())
# test_folder = torch.load("../icpr2020dfdc/weights/binclass/net-EfficientNetB4_traindb-"+ds_+"_face-scale_size-224_seed-41_ENB4/bestval.pth", map_location=torch.device('cpu'))['test_videos_used'][0]
# df = pd.read_pickle("../icpr2020dfdc/output/"+ds+"/dfs/from_video_0_to_video_0.pkl")
# #df['folder']=np.arange(len(df))
# test_df = df[df.video.isin(test_folder)]
# all_truth = test_df.label*1.0
# all_preds = []
# for file in tqdm(test_df.index):
#     f_path = "../icpr2020dfdc/output/"+ds+"/faces/"+file
#     suffix = f_path.split('.')[-1]
#     if suffix.lower() in ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'nef', 'raf']:
#         im = cv2.imread(f_path)
#         if im is None:
#             prob = -1
#         else:
#             prob, face_info = im_test(net, im)
#         all_preds.append(prob)

#     elif suffix.lower() in ['mp4', 'avi', 'mov']:
#         # Parse video
#         imgs, frame_num, fps, width, height = pv.parse_vid(f_path)
#         probs = []
#         for fid, im in enumerate(imgs):
#             print('Frame: ' + str(fid))
#             prob, face_info = im_test(net, im)
#             probs.append(prob)

#         print(probs)

# all_preds = np.array(all_preds)
# f = open("runs/"+ds+"/test_truth_preds.txt", "a")
# f.write(str(all_truth.tolist()))
# f.write("\n")
# f.write(str(all_preds.tolist()))
# f.write("\n\n")
# f.close()

import sys
sys.path.append('../icpr2020dfdc')
from blazeface import FaceExtractor, BlazeFace
from architectures import fornet,weights

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from torchvision.transforms import ToPILImage, ToTensor

from isplutils import utils, split

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import ImageChops, Image
from typing import List, Dict, Tuple

from isplutils.data import FrameFaceIterableDataset, load_face

from py_utils.DL.pytorch_utils.models.classifier import SPPNet
from torchvision import transforms

print('Test begin')

DFDC, FFPP, CELEBDF = True, False, False

device = torch.device('cuda:{:d}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
face_policy="scale"
face_size=224
seed=41
suffix=None

if DFDC:
    ds, ds_ = "dfdc", ["dfdc-35-5-10"]
elif FFPP:
    ds, ds_ = "ffpp", ["ffpp-c23-720-140-140"]
elif CELEBDF:
    ds, ds_ = "celebdf", ["celebdf-35-5-10"]

tag_params = dict(net="SPPNet",traindb='-'.join(ds_),
                  face=face_policy,size=face_size,seed=seed)
print('Parameters')
print(tag_params)
tag = ''
tag += '_'.join(['-'.join([key, str(tag_params[key])]) for key in tag_params])
if suffix is not None:
    tag += '_' + suffix
print(tag)

# Variables
weight_file = "weights/binclass/"+tag+"/bestval.pth"
face_folder = "../output/"+ds+"/faces"

net = SPPNet(backbone=50, num_class=1)
net.load_state_dict(torch.load(weight_file, map_location=device)['net'])
net.eval().to(device)

# Getting weights of BlazeFace and EfficientNetB4
transf = utils.get_transformer(face_policy, face_size, 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]), 
                                train=False)
facedet = BlazeFace().to(device)
facedet.load_weights("../icpr2020dfdc/blazeface/blazeface.pth")
facedet.load_anchors("../icpr2020dfdc/blazeface/anchors.npy")
face_extractor = FaceExtractor(facedet=facedet)
test_folder = torch.load(weight_file, map_location=device)['test_videos_used']
test_folder = test_folder[0]

df = pd.read_pickle("../output/"+ds+"/dfs/from_video_0_to_video_0.pkl")
test_df = df[df.video.isin(test_folder)]

# TODO on computer
# Extract faces
# imgs_face = []
# for index in tqdm(range(len(test_df))):
#     path = face_folder+'/'+test_df.index[index]
#     img = Image.open(path)
#     img_faces = face_extractor.process_image(img=img)
#     if len(img_faces['faces']) > 0:
#         img_face = img_faces['faces'][0] # take the face with the highest confidence score found by BlazeFace
#     imgs_face.append(img_face)

imgs_face = np.load("_imgs_face.npy", allow_pickle=True)

divisor=100
all_preds = []
for i in tqdm(range(len(test_df)//divisor)):
    faces_t = torch.stack([transf(image=im)['image'] for im in imgs_face[i*divisor:(i+1)*divisor]])
    with torch.no_grad():
        faces_pred = torch.sigmoid(net(faces_t.to(device))).cpu().numpy().flatten()
    all_preds.append(faces_pred)

faces_t = torch.stack([transf(image=im)['image'] for im in imgs_face[len(test_df)//divisor*divisor:]])
with torch.no_grad():
    faces_pred = torch.sigmoid(net(faces_t.to(device))).cpu().numpy().flatten()
all_preds.append(faces_pred)
all_preds = np.array([i for tab in all_preds for i in tab])
all_truth = np.array(test_df.label)*1.0

f = open("runs/binclass/"+tag+"/test_preds.txt", "a")
f.write(str(all_truth.tolist()))
f.write("\n")
f.write(str(all_preds.tolist()))
f.write("\n\n")
f.close()

print("Completed")

