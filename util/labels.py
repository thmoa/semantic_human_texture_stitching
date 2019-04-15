#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np

LABELS_FULL = {
    'BG': [0, 0, 0],
    'Hat': [128, 0, 0],
    'Hair': [255, 0, 0],
    'Glove': [0, 85, 0],
    'Sunglasses': [170, 0, 51],
    'UpperClothes': [255, 85, 0],
    'Dress': [0, 0, 85],
    'Coat': [0, 119, 221],
    'Socks': [85, 85, 0],
    'Pants': [0, 85, 85],
    'Torso-skin': [85, 51, 0],
    'Scarf': [52, 86, 128],
    'Skirt': [0, 128, 0],
    'Face': [0, 0, 255],
    'LeftArm': [51, 170, 221],
    'RightArm': [0, 255, 255],
    'LeftLeg': [85, 255, 170],
    'RightLeg': [170, 255, 85],
    'LeftShoe': [255, 255, 0],
    'RightShoe': [255, 170, 0],
}

LABELS_REDUCED = {
    'BG': [0, 0, 0],
    'Hat': [128, 0, 0],
    'Hair': [255, 0, 0],
    'Glove': [0, 85, 0],
    'Face': [0, 0, 255],
    'UpperClothes': [255, 85, 0],
    'Dress': [0, 0, 85],
    'Coat': [0, 119, 221],
    'Socks': [85, 85, 0],
    'Pants': [0, 85, 85],
    'Torso-skin': [85, 51, 0],
    'Scarf': [52, 86, 128],
    'Skirt': [0, 128, 0],
    'Arms': [51, 170, 221],
    'Legs': [85, 255, 170],
    'Shoes': [255, 255, 0],

    'Unseen': [125, 125, 125],
}

LABELS_MIXTURES = {
    'BG': 1,
    'Hat': 3,
    'Hair': 3,
    'Glove': 2,
    'Face': 5,
    'UpperClothes': 4,
    'Dress': 4,
    'Coat': 4,
    'Socks': 2,
    'Pants': 4,
    'Torso-skin': 2,
    'Scarf': 3,
    'Skirt': 4,
    'Arms': 1,
    'Legs': 1,
    'Shoes': 3,

    'Unseen': 1,
}

LABEL_COMP = np.ones(len(LABELS_REDUCED)) - np.eye(len(LABELS_REDUCED))
LABEL_COMP[0, 0] = 1.


def read_segmentation(file):
    segm = cv2.imread(file)[:, :, ::-1]

    segm[np.all(segm == LABELS_FULL['Sunglasses'], axis=2)] = LABELS_REDUCED['Face']
    segm[np.all(segm == LABELS_FULL['LeftArm'], axis=2)] = LABELS_REDUCED['Arms']
    segm[np.all(segm == LABELS_FULL['RightArm'], axis=2)] = LABELS_REDUCED['Arms']
    segm[np.all(segm == LABELS_FULL['LeftLeg'], axis=2)] = LABELS_REDUCED['Legs']
    segm[np.all(segm == LABELS_FULL['RightLeg'], axis=2)] = LABELS_REDUCED['Legs']
    segm[np.all(segm == LABELS_FULL['LeftShoe'], axis=2)] = LABELS_REDUCED['Shoes']
    segm[np.all(segm == LABELS_FULL['RightShoe'], axis=2)] = LABELS_REDUCED['Shoes']

    return segm

def to_ids(segm):
    ids = np.zeros(segm.shape[:2], dtype=np.uint8)
    i = 0

    for id in LABELS_REDUCED:
        ids[np.all(segm == LABELS_REDUCED[id], axis=2)] = i
        i += 1

    return ids
