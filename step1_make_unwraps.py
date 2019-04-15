#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import os
import argparse
import numpy as np
import cPickle as pkl

from tqdm import tqdm
from glob import glob
from opendr.camera import ProjectPoints

from util.visibility import VisibilityChecker
from util.labels import read_segmentation
from tex.texture import TextureData


def main(data_file, frame_dir, segm_dir, out):
    data = pkl.load(open(data_file, 'rb'))

    segm_files = np.array(sorted(glob(os.path.join(segm_dir, '*.png')) + glob(os.path.join(segm_dir, '*.jpg'))))
    frame_files = np.array(sorted(glob(os.path.join(frame_dir, '*.png')) + glob(os.path.join(frame_dir, '*.jpg'))))

    vt = np.load('assets/basicModel_vt.npy')
    ft = np.load('assets/basicModel_ft.npy')
    f = np.load('assets/basicModel_f.npy')

    verts = data['vertices']

    camera_c = data['camera_c']
    camera_f = data['camera_f']
    width = data['width']
    height = data['height']

    camera = ProjectPoints(t=np.zeros(3), rt=np.array([-np.pi, 0., 0.]), c=camera_c, f=camera_f, k=np.zeros(5))

    visibility = VisibilityChecker(width, height, f)

    texture = TextureData(1000, f, vt, ft, visibility)

    for i, (v, frame_file, segm_file) in enumerate(tqdm(zip(verts, frame_files, segm_files))):
        frame = cv2.imread(frame_file) / 255.
        segm = read_segmentation(segm_file) / 255.
        mask = np.float32(np.any(segm > 0, axis=-1))

        camera.set(v=v)

        id = os.path.splitext(os.path.basename(frame_file))[0]

        vis, iso, iso_segm = texture.get_data(frame, camera, mask, segm)

        cv2.imwrite('{}/{}_unwrap.jpg'.format(out, id), np.uint8(iso * 255))
        cv2.imwrite('{}/{}_visibility.jpg'.format(out, id), np.uint8(vis * 255))
        cv2.imwrite('{}/{}_segm.png'.format(out, id), np.uint8(iso_segm[:, :, ::-1] * 255))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'data_file',
        type=str,
        help="pkl data file")

    parser.add_argument(
        'frame_dir',
        type=str,
        help="Directory that contains frame files")

    parser.add_argument(
        'segm_dir',
        type=str,
        help="Directory that contains clothes segmentation files")

    parser.add_argument(
        'out',
        type=str,
        help="Output directory")

    args = parser.parse_args()

    main(args.data_file, args.frame_dir, args.segm_dir, args.out)
