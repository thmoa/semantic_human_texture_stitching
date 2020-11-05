#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse
import cPickle as pkl

from tqdm import trange

from util.labels import read_segmentation
from glob import glob
from tex.texture import Texture


def main(unwrap_dir, segm_template_file, gmm_file, out_file, num_iter):
    """
    Main function.

    Args:
        unwrap_dir: (str): write your description
        segm_template_file: (str): write your description
        gmm_file: (str): write your description
        out_file: (str): write your description
        num_iter: (int): write your description
    """
    iso_files = np.array(sorted(glob(os.path.join(unwrap_dir, '*_unwrap.jpg'))))
    vis_files = np.array(sorted(glob(os.path.join(unwrap_dir, '*_visibility.jpg'))))

    seams = np.load('assets/basicModel_seams.npy')
    mask = cv2.imread('assets/tex_mask_1000.png', flags=cv2.IMREAD_GRAYSCALE) / 255.

    segm_template = read_segmentation(segm_template_file)
    gmm = pkl.load(open(gmm_file, 'rb'))

    num_labels = len(iso_files)
    texture = Texture(1000, seams, mask, segm_template, gmm)

    isos = []
    visibilities = []
    for iso_file, vis_file in zip(iso_files, vis_files):
        print('reading file {}...'.format(os.path.basename(iso_file)))
        iso = cv2.imread(iso_file) / 255.
        vis = cv2.imread(vis_file, flags=cv2.IMREAD_GRAYSCALE) / 255.

        isos.append(iso)
        visibilities.append(vis)

    texture_agg = isos[0]
    visibility_agg = np.array(visibilities[0])

    tex, _ = texture.add_iso(texture_agg, visibility_agg, np.zeros_like(visibility_agg), inpaint=False)

    for i in trange(num_iter):

        rl = np.random.choice(num_labels)
        texture_agg, labels = texture.add_iso(isos[rl], visibilities[rl], rl, inpaint=i == (num_iter-1))

    print('saving {}...'.format(os.path.basename(out_file)))
    cv2.imwrite(out_file, np.uint8(255 * texture_agg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'unwrap_dir',
        type=str,
        help="Directory that contains unwrap files")

    parser.add_argument(
        'segm_template',
        type=str,
        help="Segmentation template file")

    parser.add_argument(
        'gmm',
        type=str,
        help="Mixture model prior file")

    parser.add_argument(
        'out_file',
        type=str,
        help="Texture output file (JPG or PNG)")

    parser.add_argument(
        '--iter', '-t', default=15, type=int,
        help="Texture optimization steps")

    args = parser.parse_args()

    main(args.unwrap_dir, args.segm_template, args.gmm, args.out_file, args.iter)
