"""
Dual Spatial Pyramid for exposing face warp artifacts in DeepFake videos (DSP-FWA)
"""
import torch
import torch.nn.functional as F
import cv2, os, dlib
import numpy as np
from py_utils.face_utils import lib
from py_utils.vid_utils import proc_vid as pv
from py_utils.DL.pytorch_utils.models.classifier import SPPNet
import glob
from tqdm import tqdm
import pandas as pd

sample_num = 10
# Employ dlib to extract face area and landmark points
pwd = os.path.dirname(os.path.abspath('dspfwa'))
front_face_detector = dlib.get_frontal_face_detector()
lmark_predictor = dlib.shape_predictor(pwd + '/dlib_model/shape_predictor_68_face_landmarks.dat')

ds = 'celebdf'
ds_ = 'celebdf-35-5-10' #'ffpp-c23-720-140-140'

def im_test(net, im):
    face_info = lib.align(im[:, :, (2,1,0)], front_face_detector, lmark_predictor)
    # Samples
    if len(face_info) != 1:
        prob = -1
    else:
        _, point = face_info[0]
        rois = []
        for i in range(sample_num):
            roi, _ = lib.cut_head([im], point, i)
            rois.append(cv2.resize(roi[0], (224, 224)))

        # vis_ = np.concatenate(rois, 1)
        # cv2.imwrite('vis.jpg', vis_)

        bgr_mean = np.array([103.939, 116.779, 123.68])
        bgr_mean = bgr_mean[np.newaxis, :, np.newaxis, np.newaxis]
        bgr_mean = torch.from_numpy(bgr_mean).float().cuda()

        rois = torch.from_numpy(np.array(rois)).float().cuda()
        rois = rois.permute((0, 3, 1, 2))
        prob = net(rois - bgr_mean)
        prob = F.softmax(prob, dim=1)
        prob = prob.data.cpu().numpy()
        prob = 1 - np.mean(np.sort(prob[:, 0])[np.round(sample_num / 2).astype(int):])
    return prob, face_info

def draw_face_score(im, face_info, prob):
    if len(face_info) == 0:
        return im

    _, points = np.array(face_info[0])
    x1 = np.min(points[:, 0])
    x2 = np.max(points[:, 0])
    y1 = np.min(points[:, 1])
    y2 = np.max(points[:, 1])

    # Real: (0, 255, 0), Fake: (0, 0, 255)
    color = (0, prob * 255, (1 - prob) * 255)
    cv2.rectangle(im, (x1, y1), (x2, y2), color, 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, '{:.3f}'.format(prob), (x1, y1 - 10), font, 1, color, 3, cv2.LINE_AA)
    return im

    num_class = 2
net = SPPNet(backbone=50, num_class=2)
net = net.cuda()
net.eval()
model_path = os.path.join('ckpt/', 'SPP-res50.pth')
if os.path.isfile(model_path):
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)#, map_location=torch.device('cpu'))
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['net'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(model_path, start_epoch))
else:
    raise ValueError("=> no checkpoint found at '{}'".format(model_path))

print(torch.load("../icpr2020dfdc/weights/binclass/net-EfficientNetB4_traindb-"+ds_+"_face-scale_size-224_seed-41_ENB4/bestval.pth", map_location=torch.device('cpu')).keys())
test_folder = torch.load("../icpr2020dfdc/weights/binclass/net-EfficientNetB4_traindb-"+ds_+"_face-scale_size-224_seed-41_ENB4/bestval.pth", map_location=torch.device('cpu'))['test_videos_used'][0]
df = pd.read_pickle("../icpr2020dfdc/output/"+ds+"/dfs/from_video_0_to_video_0.pkl")
#df['folder']=np.arange(len(df))
test_df = df[df.video.isin(test_folder)]
all_truth = test_df.label*1.0
all_preds = []
for file in tqdm(test_df.index):
    f_path = "../icpr2020dfdc/output/"+ds+"/faces/"+file
    suffix = f_path.split('.')[-1]
    if suffix.lower() in ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'nef', 'raf']:
        im = cv2.imread(f_path)
        if im is None:
            prob = -1
        else:
            prob, face_info = im_test(net, im)
        all_preds.append(prob)

    elif suffix.lower() in ['mp4', 'avi', 'mov']:
        # Parse video
        imgs, frame_num, fps, width, height = pv.parse_vid(f_path)
        probs = []
        for fid, im in enumerate(imgs):
            print('Frame: ' + str(fid))
            prob, face_info = im_test(net, im)
            probs.append(prob)

        print(probs)

all_preds = np.array(all_preds)
f = open("runs/"+ds+"/test_truth_preds.txt", "a")
f.write(str(all_truth.tolist()))
f.write("\n")
f.write(str(all_preds.tolist()))
f.write("\n\n")
f.close()
