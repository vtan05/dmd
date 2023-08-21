# Reference: https://github.com/Stanford-TML/EDGE
# Extract normalized motion and genre (conditioning signals)

import os
import glob
import pickle
import numpy as np
import librosa
import torch

from pathlib import Path 
from sklearn.preprocessing import OneHotEncoder
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.transforms.rotation_conversions import (axis_angle_to_quaternion,
                                                       quaternion_multiply,
                                                       quaternion_to_axis_angle,
                                                       quaternion_invert)

from utils.quaternion import ax_to_6v
from utils.preprocess import Normalizer
from utils.vis import SMPLSkeleton


ABLATION_LIST = ["all", "pos", "orn", "linvel", "angvel"]


def concat_aistpp(data_path: str, backup_path: str, is_train: bool = True):
    # motion data specification
    raw_fps = 60
    data_fps = 30
    assert data_fps <= raw_fps
    data_stride = raw_fps // data_fps
    
    # file save specificiation
    split_data_path = os.path.join(
        data_path, "train" if is_train else "test")
    
    backup_path = Path(backup_path)
    backup_path.mkdir(parents=True, exist_ok=True)
    pickle_name = "processed_train_data.pkl" if is_train else "processed_test_data.pkl"
    if pickle_name in os.listdir(backup_path):
        return
    
    # load dataset
    print("Loading dataset...")
    motion_path = os.path.join(split_data_path, "motions_sliced")
    
    # sort motions and sounds
    motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))

    # stack the motions and features together
    all_pos = []
    all_q = []
    all_names = []
    all_genre = []

    for motion in motions:
        # make sure name is matching
        m_name = os.path.splitext(os.path.basename(motion))[0]
        
        # load motion
        data = pickle.load(open(motion, "rb"))
        pos = data["pos"]
        q = data["q"]
        all_pos.append(pos)
        all_q.append(q)
        all_names.append(m_name)
        all_genre.append(m_name.split("_")[0])

    all_pos = np.array(all_pos)  # N x seq x 3
    all_q = np.array(all_q)  # N x seq x (joint * 3)
    
    # downsample the motions to the data fps
    print(f"total concated pos dim : {all_pos.shape}")
    
    all_pos = all_pos[:, :: data_stride, :]
    all_q = all_q[:, :: data_stride, :]
    data = {"root_pos": all_pos, "joint_orn": all_q, "filenames": all_names, "genre": all_genre}
    
    with open(os.path.join(backup_path, pickle_name), "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
        
def cal_joint_ang_vel(joint_orn: torch.Tensor, fps: int=30)-> torch.Tensor:
    bs, num_frame, num_joint, _ = joint_orn.size()
    joint_ang_vel = torch.zeros((bs, num_frame, num_joint, 3), dtype=torch.float32)
    frame_list = np.arange(0, num_frame, dtype=int)
    prev_frame = np.maximum(0, frame_list-1)
    next_frame = np.minimum(num_frame-1, frame_list+1)
    
    dframe = next_frame - prev_frame
    
    frame_list = torch.tensor(frame_list, dtype=torch.int64)
    prev_frame = torch.tensor(prev_frame, dtype=torch.int64)
    next_frame = torch.tensor(next_frame, dtype=torch.int64)
    dframe     = torch.tensor(dframe, dtype=torch.int64).view(1, num_frame, 1, 1)
    
    dorn = quaternion_invert(joint_orn[:,prev_frame, : ,:])*joint_orn[:,next_frame, :, :]
    dorn_ax = quaternion_to_axis_angle(dorn)
    joint_ang_vel[:, frame_list, :, : ] = fps * torch.div(dorn_ax, dframe)

    return joint_ang_vel


def cal_joint_lin_vel(joint_pos: torch.Tensor, fps: int =30)-> torch.Tensor:
    _, num_frame, _, _ = joint_pos.size()
    joint_lin_vel = torch.zeros_like(joint_pos)
    frame_list = np.arange(0, num_frame, dtype=int)
    prev_frame = np.maximum(0, frame_list-1)
    next_frame = np.minimum(num_frame-1, frame_list+1)
    
    dframe = next_frame - prev_frame
    
    frame_list = torch.tensor(frame_list, dtype=torch.int64)
    prev_frame = torch.tensor(prev_frame, dtype=torch.int64)
    next_frame = torch.tensor(next_frame, dtype=torch.int64)
    dframe     = torch.tensor(dframe, dtype=torch.int64).view(1, num_frame, 1, 1)

    joint_lin_vel[:, frame_list, :, :] = fps * torch.div((joint_pos[:, next_frame, :, :] - joint_pos[:, prev_frame, : ,:]), dframe)

    return joint_lin_vel
    

def extract_motion(root_pos: np.ndarray, joint_orn: np.ndarray, data_sort: str = "all")-> torch.Tensor:
    smpl = SMPLSkeleton()
    # to Tensor
    root_pos = torch.Tensor(root_pos)
    joint_orn = torch.Tensor(joint_orn)
    # to ax
    bs, sq, _ = joint_orn.shape
    joint_orn = joint_orn.reshape((bs, sq, -1, 3))

    # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
    root_q = joint_orn[:, :, :1, :]  # sequence x 1 x 3
    root_q_quat = axis_angle_to_quaternion(root_q)
    rotation = torch.Tensor(
        [0.7071068, 0.7071068, 0, 0]
    )  # 90 degrees about the x axis
    root_q_quat = quaternion_multiply(rotation, root_q_quat)
    root_q = quaternion_to_axis_angle(root_q_quat)
    joint_orn[:, :, :1, :] = root_q

    pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
    root_pos = pos_rotation.transform_points(
        root_pos
    )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

    # get joint pos
    joint_pos = smpl.forward(joint_orn, root_pos)  # batch x sequence x 24 x 3

    # get joint linear vel
    joint_lin_vel = cal_joint_lin_vel(joint_pos, 30)    # batch x sequence x 24 x 3
    
    # get joint angular vel
    joint_ang_vel = cal_joint_ang_vel(axis_angle_to_quaternion(joint_orn), 30)
    
    # get joint orn with 6D repr
    joint_orn = ax_to_6v(joint_orn) # batch x sequence x 24 x 6
    
    
    ## generate motion feature along data_sort
    if data_sort == "all":
        motion_feature = torch.cat([joint_orn, joint_pos, joint_lin_vel, joint_ang_vel], axis= -1)
        print(np.array(motion_feature).shape)
    elif data_sort == "pos":
        motion_feature = torch.cat([joint_orn, joint_lin_vel, joint_ang_vel], axis= -1)
        print(np.array(motion_feature).shape)
    elif data_sort == "orn":
        motion_feature = torch.cat([joint_pos, joint_lin_vel, joint_ang_vel], axis= -1)
        print(np.array(motion_feature).shape)
    elif data_sort == "linvel":
        motion_feature = torch.cat([joint_orn, joint_pos, joint_ang_vel], axis= -1)
        print(np.array(motion_feature).shape)
    elif data_sort == "angvel":
        motion_feature = torch.cat([joint_orn, joint_pos, joint_lin_vel], axis= -1)
        print(np.array(motion_feature).shape)
    else:
        assert False, f"data_sort is not supported : {data_sort}"
    
    return motion_feature.view(bs, sq, -1)  # batch x sequence x (24 x 15)

    
def preprocess_aistpp(pickle_path: str, is_train: bool = True, data_sort: str = "all"):
    pickle_name = "processed_train_data.pkl" if is_train else "processed_test_data.pkl"
    with open(os.path.join(pickle_path, pickle_name), "rb") as f:
        data = pickle.load(f)
        
    motion_root_pos = data["root_pos"]
    motion_joint_orn = data["joint_orn"]
    names = data["filenames"]
    genres = data["genre"]

    encoder = OneHotEncoder()
    encoded_genres = encoder.fit_transform(np.array(genres).reshape(-1,1))
    
    motion_feature_array: torch.Tensor = extract_motion(motion_root_pos, motion_joint_orn, sort)    # bs x sq x (num_joints x motion_feature(15))
    
    # normalizer
    normalizer_name = f"normalizer_{data_sort}_01.pkl"
    
    if is_train:
        normalizer = Normalizer(motion_feature_array)
        motion_feature_array = normalizer.normalize(motion_feature_array)
        with open(os.path.join(pickle_path, normalizer_name), "wb") as f:
            pickle.dump(normalizer, f, pickle.HIGHEST_PROTOCOL)
    else:
        normalizer: Normalizer = pickle.load(open(os.path.join(pickle_path, normalizer_name), "rb"))
        motion_feature_array = normalizer.normalize(motion_feature_array)
    
    motion_feature_array = motion_feature_array.numpy()
    
    encodings = {}
    print(len(names))
    for i in range(0, len(names)):
        genres = np.tile(encoded_genres[i].todense(), (150, 1))
        features = np.concatenate((motion_feature_array[i], genres), axis=-1)
        encodings[names[i]] = features
    
    processed_data_name = f"normalized_{data_sort}_train_data_01.pkl" if is_train else f"normalized_{data_sort}_test_data_01.pkl"
    with open(os.path.join(pickle_path, processed_data_name), "wb") as f:
        pickle.dump(encodings, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"Finished data preprocess for {data_sort}")

    
if __name__ == '__main__':
    #data_path = f"./data/"
    #backup_path = f"./data/merged/"
    #concat_aistpp(data_path, backup_path, False)
    
    # Server Paths
    backup_path = r"/host_data/van/edge_aistpp/test/concat/" # test folder
    #backup_path = r"/host_data/van/edge_aistpp/encoding/" # train folder
    for sort in ABLATION_LIST:
        preprocess_aistpp(pickle_path = backup_path, is_train = False, data_sort = sort)
    
    