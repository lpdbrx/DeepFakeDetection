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
from tqdm import tqdm
import pandas as pd


import shutil, warnings, sys, glob

sys.path.append('../icpr2020dfdc')
from blazeface import FaceExtractor, BlazeFace
from isplutils.data import FrameFaceIterableDataset, load_face
from isplutils import utils, split

import albumentations as A
import torch.multiprocessing
from torchvision.transforms import ToPILImage, ToTensor

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
#!pip install --upgrade --force-reinstall --no-deps albumentations==0.4.6
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
#!pip install tensorboardX
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from PIL import ImageChops, Image
from typing import List, Dict, Tuple
from torchvision import transforms





## TODO: delete

sample_num = 10
# Employ dlib to extract face area and landmark points
pwd = os.path.dirname(os.path.abspath('dspfwa'))
front_face_detector = dlib.get_frontal_face_detector()
lmark_predictor = dlib.shape_predictor(pwd + '/dlib_model/shape_predictor_68_face_landmarks.dat')

ds = 'ffpp'
ds_ = 'ffpp-c23-720-140-140'

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
        bgr_mean = torch.from_numpy(bgr_mean).float()#.cuda()

        rois = torch.from_numpy(np.array(rois)).float()#.cuda()
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


########################################
# CHOOSE DF
########################################

DFDC, FFPP, CELEBDF = True, False, False
suffix = "ENB4"


########################################
# OVERWRITTEN FUNCTIONS
########################################

def tb_attention(tb: SummaryWriter,
                 tag: str,
                 iteration: int,
                 net: nn.Module,
                 device: torch.device,
                 patch_size_load: int,
                 face_crop_scale: str,
                 val_transformer: A.BasicTransform,
                 root: str,
                 record: pd.Series):
    # Crop face
    sample_t = load_face(record=record, root=root, size=patch_size_load, scale=face_crop_scale,
                         transformer=val_transformer)
    sample_t_clean = load_face(record=record, root=root, size=patch_size_load, scale=face_crop_scale,
                               transformer=ToTensorV2())
    if torch.cuda.is_available():
        sample_t = sample_t.cuda(device)
    # Transform
    # Feed to net
    with torch.no_grad():
        att: torch.Tensor = net.get_attention(sample_t.unsqueeze(0))[0].cpu()
    att_img: Image.Image = ToPILImage()(att)
    sample_img = ToPILImage()(sample_t_clean)
    att_img = att_img.resize(sample_img.size, resample=Image.NEAREST).convert('RGB')
    sample_att_img = ImageChops.multiply(sample_img, att_img)
    sample_att = ToTensor()(sample_att_img)
    tb.add_image(tag=tag, img_tensor=sample_att, global_step=iteration)


def batch_forward(net: nn.Module, device: torch.device, criterion, data: torch.Tensor, labels: torch.Tensor) -> (
        torch.Tensor, float, int):
    data = data.to(device)
    labels = labels.to(device)
    out = net(data)
    pred = torch.sigmoid(out).detach().cpu().numpy()
    loss = criterion(out, labels)
    return loss, pred


def validation_routine(net, device, val_loader, criterion, tb, iteration, tag: str, loader_len_norm: int = None):
    net.eval()
    loader_len_norm = loader_len_norm if loader_len_norm is not None else val_loader.batch_size
    val_num = 0
    val_loss = 0.
    pred_list = list()
    labels_list = list()
    for val_data in tqdm(val_loader, desc='Validation', leave=False, total=len(val_loader) // loader_len_norm):
        batch_data, batch_labels = val_data

        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward(net, device, criterion, batch_data,
                                                           batch_labels)
        pred_list.append(val_batch_pred.flatten())
        val_num += val_batch_num
        val_loss += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss /= val_num
    tb.add_scalar('{}/loss'.format(tag), val_loss, iteration)

    if isinstance(criterion, nn.BCEWithLogitsLoss):
        val_labels = np.concatenate(labels_list)
        val_pred = np.concatenate(pred_list)
        val_roc_auc = roc_auc_score(val_labels, val_pred)
        tb.add_scalar('{}/roc_auc'.format(tag), val_roc_auc, iteration)
        tb.add_pr_curve('{}/pr'.format(tag), val_labels, val_pred, iteration)

    return val_loss


def save_model(net: nn.Module, optimizer: optim.Optimizer,
               train_loss: float, val_loss: float,
               iteration: int, batch_size: int, epoch: int,
               path: str, train_videos_used: List, val_videos_used: List, test_videos_used: List):
    path = str(path)
    state = dict(net=net.state_dict(),
                 opt=optimizer.state_dict(),
                 train_loss=train_loss,
                 val_loss=val_loss,
                 iteration=iteration,
                 batch_size=batch_size,
                 epoch=epoch,
                 train_videos_used=train_videos_used, 
                 val_videos_used=val_videos_used, 
                 test_videos_used=test_videos_used)
    torch.save(state, path)



# modified
def load_df(dfdc_df_path: str, ffpp_df_path: str, celebdf_df_path: str, dfdc_faces_dir: str, ffpp_faces_dir: str, celebdf_faces_dir: str, dataset: str) -> (pd.DataFrame, str):
    if dataset.startswith('dfdc'):
        df = pd.read_pickle(dfdc_df_path)
        root = dfdc_faces_dir
    elif dataset.startswith('celebdf'):
        df = pd.read_pickle(celebdf_df_path)
        root = celebdf_faces_dir
    elif dataset.startswith('ffpp'):
        df = pd.read_pickle(ffpp_df_path)
        folder = np.arange(len(df))
        df['folder']=folder
        root = ffpp_faces_dir
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(dataset))
    return df, root

#modified
def get_split_df(df: pd.DataFrame, dataset: str, split: str) -> pd.DataFrame:
    if dataset == 'dfdc-35-5-10' or dataset == 'celebdf-35-5-10':
        st0 = np.random.get_state()
        np.random.seed(41)

        fake_videos_df = df[df['class']=='manipulated_sequences']
        videos = fake_videos_df.video.unique()
        rdm_videos_indices = np.random.choice(len(videos), len(videos), replace=False)
        train_videos = videos[rdm_videos_indices[:int(0.75*len(videos))]]
        valid_videos = videos[rdm_videos_indices[int(0.75*len(videos)):int(0.8*len(videos))]]
        test_videos = videos[rdm_videos_indices[int(0.8*len(videos)):]]

        real_videos_df = df[df['class']=='original_sequences']
        videos = real_videos_df.video.unique()    
        rdm_videos_indices = np.random.choice(len(videos), len(videos), replace=False)
        train_videos = np.append(train_videos, videos[rdm_videos_indices[:int(0.75*len(videos))]])
        valid_videos = np.append(valid_videos, videos[rdm_videos_indices[int(0.75*len(videos)):int(0.8*len(videos))]])
        test_videos = np.append(test_videos, videos[rdm_videos_indices[int(0.8*len(videos)):]])
        
        if split == 'train':
            videos_used = train_videos
            split_df = df[df.video.isin(videos_used)]
        elif split == 'val':
            videos_used = valid_videos
            split_df = df[df.video.isin(videos_used)]
        elif split == 'test':
            videos_used = test_videos
            split_df = df[df.video.isin(videos_used)]
        else:
            raise NotImplementedError('Unknown split: {}'.format(split))

    elif dataset.startswith('ffpp'):
        # Save random state
        st0 = np.random.get_state()
        # Set seed for this selection only
        np.random.seed(41)
        # Split on original videos
        crf = dataset.split('-')[1]

        random_youtube_videos = np.random.permutation(
            df[(df['source'] == 'youtube') & (df['quality'] == crf)]['video'].unique())
        train_orig = random_youtube_videos[:720]
        val_orig = random_youtube_videos[720:720 + 140]
        test_orig = random_youtube_videos[720 + 140:]
        if split == 'train':
            split_df = pd.concat((df[df['original'].isin(train_orig)], df[df['video'].isin(train_orig)]), axis=0)
            videos_used = split_df.video.unique()
        elif split == 'val':
            split_df = pd.concat((df[df['original'].isin(val_orig)], df[df['video'].isin(val_orig)]), axis=0)
            videos_used = split_df.video.unique()
        elif split == 'test':
            split_df = pd.concat((df[df['original'].isin(test_orig)], df[df['video'].isin(test_orig)]), axis=0)
            videos_used = split_df.video.unique()
        else:
            raise NotImplementedError('Unknown split: {}'.format(split))

        if dataset.endswith('fpv'):
            fpv = int(dataset.rsplit('-', 1)[1][:-3])
            idxs = []
            for video in split_df['video'].unique():
                idxs.append(np.random.choice(split_df[split_df['video'] == video].index, fpv, replace=False))
            idxs = np.concatenate(idxs)
            split_df = split_df.loc[idxs]
        # Restore random state
        np.random.set_state(st0)
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(dataset))
    return split_df, videos_used


def make_splits(dfdc_df: str, ffpp_df: str, celebdf_df: str, dfdc_dir: str, ffpp_dir: str, celebdf_dir: str, dbs: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple[pd.DataFrame, str]]]:
    """
    Make split and return Dataframe and root
    :param
    dfdc_df: str, path to the DataFrame containing info on the faces extracted from the DFDC dataset with extract_faces.py
    ffpp_df: str, path to the DataFrame containing info on the faces extracted from the FF++ dataset with extract_faces.py
    dfdc_dir: str, path to the directory containing the faces extracted from the DFDC dataset with extract_faces.py
    ffpp_dir: str, path to the directory containing the faces extracted from the FF++ dataset with extract_faces.py
    dbs: {split_name:[split_dataset1,split_dataset2,...]}
                Example:
                {'train':['dfdc-35-5-15',],'val':['dfdc-35-5-15',]}
    :return: split_dict: dictonary containing {split_name: ['train', 'val'], splitdb: List(pandas.DataFrame, str)}
                Example:
                {'train, 'dfdc-35-5-15': (dfdc_train_df, 'path/to/dir/of/DFDC/faces')}
    """
    split_dict = {}
    full_dfs = {}

    for split_name, split_dbs in dbs.items():
        split_dict[split_name] = dict()
        for split_db in split_dbs:
            if split_db not in full_dfs:
                full_dfs[split_db] = load_df(dfdc_df, ffpp_df, celebdf_df, dfdc_dir, ffpp_dir, celebdf_dir, split_db)
            full_df, root = full_dfs[split_db]
            split_df, folder = get_split_df(df=full_df, dataset=split_db, split=split_name)
            split_dict[split_name][split_db] = (split_df, root, folder)

    return split_dict
















net_class = "SPPNet"
if CELEBDF:
    train_datasets, val_datasets, test_datasets = ["celebdf-35-5-10"], ["celebdf-35-5-10"], ["celebdf-35-5-10"]
if FFPP:
    train_datasets, val_datasets, test_datasets = ["ffpp-c23-720-140-140"], ["ffpp-c23-720-140-140"], ["ffpp-c23-720-140-140"]
if DFDC:
    train_datasets, val_datasets, test_datasets = ["dfdc-35-5-10"], ["dfdc-35-5-10"], ["dfdc-35-5-10"]

dfdc_df_path = "../icpr2020dfdc/output/dfdc/dfs/from_video_0_to_video_0.pkl"
ffpp_df_path = "../icpr2020dfdc/output/ffpp/dfs/from_video_0_to_video_0.pkl"
celebdf_df_path = "../icpr2020dfdc/output/celebdf/dfs/from_video_0_to_video_0.pkl"
dfdc_faces_dir = "../icpr2020dfdc/output/dfdc/faces"
ffpp_faces_dir = "../icpr2020dfdc/output/ffpp/faces"
celebdf_faces_dir = "../icpr2020dfdc/output/celebdf/faces"
face_policy = "scale"
face_size = 224

batch_size = 32
initial_lr = 1e-5
validation_interval = 500
patience = 10
max_num_iterations = 20000
initial_model = None
train_from_scratch = True

max_train_samples = -1
max_val_samples = 6000

log_interval = 100
num_workers = 6
device = torch.device('cuda:{:d}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
seed = 41


debug = False
suffix = ""

enable_attention = True

weights_folder = "weights/binclass/"
logs_folder = 'runs/binclass/'




########################################
# MODEL
########################################

# Random initialization
np.random.seed(seed)
torch.random.manual_seed(seed)

# Load net
net: nn.Module = SPPNet(backbone=50, num_class=1).to(device)

# Loss and optimizers
criterion = nn.BCEWithLogitsLoss()

min_lr = initial_lr * 1e-5
optimizer = optim.Adam(net.parameters(), lr=initial_lr)

#allows dynamic learning rate reducing based on some validation measurements.
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode='min',
    factor=0.1,
    patience=patience,
    cooldown=2 * patience,
    min_lr=min_lr,
) 

# Training parameters and tag
tag_params = dict(net="SPPNet",traindb='-'.join(train_datasets),
                  face=face_policy,size=face_size,seed=seed)
print('Parameters')
print(tag_params)
tag = ''
tag += '_'.join(['-'.join([key, str(tag_params[key])]) for key in tag_params])
if suffix is not None:
    tag += '_' + suffix
print(tag)

# Model checkpoint paths
bestval_path = os.path.join(weights_folder, tag, 'bestval.pth')
last_path = os.path.join(weights_folder, tag, 'last.pth')
periodic_path = os.path.join(weights_folder, tag, 'it{:06d}.pth')

os.makedirs(os.path.join(weights_folder, tag), exist_ok=True)

# Load model
val_loss = min_val_loss = 10
epoch = iteration = 0


# Initialize Tensorboard
logdir = os.path.join(logs_folder, tag)
if iteration == 0:
    # If training from scratch or initialization remove history if exists
    shutil.rmtree(logdir, ignore_errors=True)


# TensorboardX instance
tb = SummaryWriter(logdir=logdir)
if iteration == 0:
    dummy = torch.randn((1, 3, face_size, face_size), device=device)
    dummy = dummy.to(device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tb.add_graph(net, [dummy, ], verbose=False)

transformer = utils.get_transformer(face_policy=face_policy, patch_size=face_size,
                                    net_normalizer=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), train=True)


# Datasets and data loaders
print('Loading data')
# Check if paths for DFDC and FF++ extracted faces and DataFrames are provided
for dataset in train_datasets:
    if dataset.split('-')[0] == 'dfdc' and (dfdc_df_path is None or dfdc_faces_dir is None):
        raise RuntimeError('Specify DataFrame and directory for DFDC faces for training!')
    elif dataset.split('-')[0] == 'ff' and (ffpp_df_path is None or ffpp_faces_dir is None):
        raise RuntimeError('Specify DataFrame and directory for FF++ faces for training!')
    elif dataset.split('-')[0] == 'celebdf' and (celebdf_df_path is None or celebdf_faces_dir is None):
        raise RuntimeError('Specify DataFrame and directory for CELEB-DF faces for training!')
for dataset in val_datasets:
    if dataset.split('-')[0] == 'dfdc' and (dfdc_df_path is None or dfdc_faces_dir is None):
        raise RuntimeError('Specify DataFrame and directory for DFDC faces for validation!')
    elif dataset.split('-')[0] == 'ff' and (ffpp_df_path is None or ffpp_faces_dir is None):
        raise RuntimeError('Specify DataFrame and directory for FF++ faces for validation!')
    elif dataset.split('-')[0] == 'celebdf' and (celebdf_df_path is None or celebdf_faces_dir is None):
        raise RuntimeError('Specify DataFrame and directory for CELEB-DF faces for training!')
for dataset in test_datasets:
    if dataset.split('-')[0] == 'dfdc' and (dfdc_df_path is None or dfdc_faces_dir is None):
        raise RuntimeError('Specify DataFrame and directory for DFDC faces for test!')
    elif dataset.split('-')[0] == 'ff' and (ffpp_df_path is None or ffpp_faces_dir is None):
        raise RuntimeError('Specify DataFrame and directory for FF++ faces for test!')
    elif dataset.split('-')[0] == 'celebdf' and (celebdf_df_path is None or celebdf_faces_dir is None):
        raise RuntimeError('Specify DataFrame and directory for CELEB-DF faces for test!')


# Load splits with the make_splits function
splits = make_splits(dfdc_df=dfdc_df_path, ffpp_df=ffpp_df_path, celebdf_df=celebdf_df_path,
                     dfdc_dir=dfdc_faces_dir, ffpp_dir=ffpp_faces_dir, celebdf_dir=celebdf_faces_dir,
                     dbs={'train': train_datasets, 'val': val_datasets, 'test': test_datasets})


train_dfs = [splits['train'][db][0] for db in splits['train']]
train_roots = [splits['train'][db][1] for db in splits['train']]
train_videos_used = [splits['train'][db][2] for db in splits['train']]
val_dfs = [splits['val'][db][0] for db in splits['val']]
val_roots = [splits['val'][db][1] for db in splits['val']]
val_videos_used = [splits['val'][db][2] for db in splits['val']]
test_dfs = [splits['test'][db][0] for db in splits['test']]
test_roots = [splits['test'][db][1] for db in splits['test']]
test_videos_used = [splits['test'][db][2] for db in splits['test']]

train_dataset = FrameFaceIterableDataset(roots=train_roots,
                                         dfs=train_dfs,
                                         scale=face_policy,
                                         num_samples=max_train_samples,
                                         transformer=transformer,
                                         size=face_size,
                                         )

val_dataset = FrameFaceIterableDataset(roots=val_roots,
                                       dfs=val_dfs,
                                       scale=face_policy,
                                       num_samples=max_val_samples,
                                       transformer=transformer,
                                       size=face_size,
                                       )

train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, )

val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, )



print('Training samples: {}'.format(len(train_dataset)))
print('Validation samples: {}'.format(len(val_dataset)))
print('Test samples: {}'.format(len(test_dfs)))


if len(train_dataset) == 0:
    print('No training samples. Halt.')

if len(val_dataset) == 0:
    print('No validation samples. Halt.')





stop = False
while not stop:

    # Training
    optimizer.zero_grad()

    train_loss = train_num = 0
    train_pred_list = []
    train_labels_list = []
    for train_batch in tqdm(train_loader, desc='Epoch {:03d} - Train loss {:03f} - Val loss {:03f}'.format(epoch, train_loss, val_loss), leave=False,
                            total=len(train_loader) // train_loader.batch_size):
        
        net.train()
        batch_data, batch_labels = train_batch

        train_batch_num = len(batch_labels)
        train_num += train_batch_num
        train_labels_list.append(batch_labels.numpy().flatten())

        train_batch_loss, train_batch_pred = batch_forward(net, device, criterion, batch_data, batch_labels)

        train_pred_list.append(train_batch_pred.flatten())
        if torch.isnan(train_batch_loss):
            raise ValueError('NaN loss')

        train_loss += train_batch_loss.item() * train_batch_num

        # Optimization
        train_batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
                
        # Logging
        if iteration > 0 and (iteration % log_interval == 0):
            train_loss /= train_num
            tb.add_scalar('train/loss', train_loss, iteration)
            tb.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration)
            tb.add_scalar('epoch', epoch, iteration)

            # Checkpoint
            save_model(net, optimizer, train_loss, val_loss, iteration, batch_size, epoch, last_path, train_videos_used, val_videos_used, test_videos_used)
            train_loss = train_num = 0

        # Validation
        if iteration > 0 and (iteration % validation_interval == 0):

            # Model checkpoint
            save_model(net, optimizer, train_loss, val_loss, iteration, batch_size, epoch,
                       periodic_path.format(iteration), train_videos_used, val_videos_used, test_videos_used)

            # Train cumulative stats
            train_labels = np.concatenate(train_labels_list)
            train_pred = np.concatenate(train_pred_list)
            train_labels_list = []
            train_pred_list = []
            
            f = open("runs/binclass/"+tag+"/validation_preds.txt", "a")
            f.write(str(train_labels.tolist()))
            f.write("\n")
            f.write(str(train_pred.tolist()))
            f.write("\n\n")
            f.close()

            train_roc_auc = roc_auc_score(train_labels, train_pred)
            tb.add_scalar('train/roc_auc', train_roc_auc, iteration)
            tb.add_pr_curve('train/pr', train_labels, train_pred, iteration)

            # Validation
            val_loss = validation_routine(net, device, val_loader, criterion, tb, iteration, 'val')
            tb.flush()

            # LR Scheduler
            lr_scheduler.step(val_loss)
            
            # Model checkpoint
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                save_model(net, optimizer, train_loss, val_loss, iteration, batch_size, epoch, bestval_path, train_videos_used, val_videos_used, test_videos_used)

            # Attention
            if enable_attention and hasattr(net, 'get_attention'):
                net.eval()
                # For each dataframe show the attention for a real,fake couple of frames
                for df, root, sample_idx, tag in [
                    (train_dfs[0], train_roots[0], train_dfs[0][train_dfs[0]['label'] == False].index[0],
                     'train/att/real'),
                    (train_dfs[0], train_roots[0], train_dfs[0][train_dfs[0]['label'] == True].index[0],
                     'train/att/fake'),
                ]:
                    record = df.loc[sample_idx]
                    tb_attention(tb, tag, iteration, net, device, face_size, face_policy,
                                 transformer, root, record)

            if optimizer.param_groups[0]['lr'] == min_lr:
                print('Reached minimum learning rate. Stopping.')
                stop = True
                break

        iteration += 1

        if iteration > max_num_iterations:
            print('Maximum number of iterations reached')
            stop = True
            break

        # End of iteration

    epoch += 1

# Needed to flush out last events
tb.close()

print('Training completed')









# num_class = 2
# net = SPPNet(backbone=50, num_class=2)
# net = net.cuda()
# net.eval()
# model_path = os.path.join('ckpt/', 'SPP-res50.pth')
# if os.path.isfile(model_path):
#     print("=> loading checkpoint '{}'".format(model_path))
#     checkpoint = torch.load(model_path)
#     start_epoch = checkpoint['epoch']
#     net.load_state_dict(checkpoint['net'])
#     print("=> loaded checkpoint '{}' (epoch {})"
#           .format(model_path, start_epoch))
# else:
#     raise ValueError("=> no checkpoint found at '{}'".format(model_path))

# print(torch.load("../icpr2020dfdc/weights/binclass/net-EfficientNetB4_traindb-"+ds_+"_face-scale_size-224_seed-41_1/bestval.pth", map_location=torch.device('cpu')).keys())
# test_folder = torch.load("../icpr2020dfdc/weights/binclass/net-EfficientNetB4_traindb-"+ds_+"_face-scale_size-224_seed-41_1/bestval.pth", map_location=torch.device('cpu'))['test_folder'][0]
# df = pd.read_pickle("../icpr2020dfdc/output/"+ds+"/dfs/from_video_0_to_video_0.pkl")
# df['folder']=np.arange(len(df))
# test_df = df[df.folder.isin(test_folder)]
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