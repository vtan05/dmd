# References: https://github.com/lisiyao21/Bailando
# https://github.com/google-research/mint
# https://github.com/Stanford-TML/EDGE
# Compute the beat aligned score for the test set from CDCD list

import numpy as np
import matplotlib.pyplot as plt 
import pickle 
import json
import os
import glob
import torch
import librosa

from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
from utils.vis import SMPLSkeleton
from scipy import linalg
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.transforms.rotation_conversions import (axis_angle_to_quaternion,
                                                       quaternion_multiply,
                                                       quaternion_to_axis_angle,
                                                       quaternion_invert)

from eval_cdcd import beat_detect
from kinetic import extract_kinetic_features
from manual import extract_manual_features


def cal_motion_beat(root_pos : np.ndarray, joint_orn : np.ndarray, fps :int=30):
    smpl = SMPLSkeleton()

    root_pos = torch.Tensor(root_pos)
    root_pos = root_pos.reshape((1, 300, 3))
    joint_orn = torch.Tensor(joint_orn)
    joint_orn = joint_orn.reshape((1, 300, -1, 3))

    # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
    root_q = joint_orn[:, :, :1, :]  # sequence x 1 x 3
    root_q_quat = axis_angle_to_quaternion(root_q)
    rotation = torch.Tensor([0.7071068, 0.7071068, 0, 0])  # 90 degrees about the x axis
    root_q_quat = quaternion_multiply(rotation, root_q_quat)
    root_q = quaternion_to_axis_angle(root_q_quat)
    joint_orn[:, :, :1, :] = root_q

    pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
    root_pos = pos_rotation.transform_points(root_pos)  # basically (y, z) -> (-z, y), expressed as a rotation for readability

    # get joint pos
    joint_pos = smpl.forward(joint_orn, root_pos)  # batch x sequence x 24 x 3

    joint_pos = np.array(joint_pos).reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((joint_pos[1:] - joint_pos[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beat = argrelextrema(kinetic_vel, np.less)
        
    return motion_beat, len(kinetic_vel)


def ba_score(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))


def calc_ba_score(motion_dir, music_dir):

    ba_scores = []

    audio_files = [line.rstrip() for line in open('CDCD_aist.txt')]
    for motion in audio_files:
        m_name = motion[:-4]
        data = pickle.load(open(motion_dir + m_name + '.pkl', "rb"))

        root_pos = data["pos"]
        joint_orn = data["q"]
        dance_beats, length = cal_motion_beat(root_pos, joint_orn)

        # Beat Extractor: Librosa
        music, sr = librosa.load('{}{}.wav'.format(music_dir, m_name))
        onset_env = librosa.onset.onset_strength(y=music)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        ba_scores.append(ba_score(beats, dance_beats))
        
    return np.mean(ba_scores)


if __name__ == '__main__':

    motion_dir = r"/host_data/van/edge_aistpp/test/motions_sliced/"
    music_dir = r"/host_data/van/edge_aistpp/outputv2/angvel_01/normalized/"
    print(calc_ba_score(motion_dir, music_dir))
