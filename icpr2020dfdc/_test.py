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

print('Test begin')

DFDC, FFPP, CELEBDF = False, True, False

device = torch.device('cuda:{:d}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
face_policy="scale"
face_size=224

if DFDC:
    ds, ds_ = "dfdc", ["dfdc-35-5-10"]
elif FFPP:
    ds, ds_ = "ffpp", ["ffpp-c23-720-140-140"]
elif CELEBDF:
    ds, ds_ = "celebdf", ["celebdf-35-5-10"]

tag = utils.make_train_tag(net_class=getattr(fornet, "EfficientNetB4"),
                           traindb=ds_,
                           face_policy=face_policy,
                           patch_size=face_size,
                           seed=41,
                           suffix="ENB4",
                           debug=False)

# Variables
weight_file = "weights/binclass/"+tag+"/bestval.pth"
face_folder = "output/"+ds+"/faces"

net = getattr(fornet,"EfficientNetB4")()
net.load_state_dict(torch.load(weight_file, map_location=device)['net'])
net.eval().to(device)

# Getting weights of BlazeFace and EfficientNetB4
transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)
facedet = BlazeFace().to(device)
facedet.load_weights("blazeface/blazeface.pth")
facedet.load_anchors("blazeface/anchors.npy")
face_extractor = FaceExtractor(facedet=facedet)
test_folder = torch.load(weight_file, map_location=device)['test_videos_used']
test_folder = test_folder[0]

df = pd.read_pickle("output/"+ds+"/dfs/from_video_0_to_video_0.pkl")
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