#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import os
import gco
import argparse
import numpy as np
import cPickle as pkl

from glob import glob
from scipy import signal

from util.labels import LABELS_REDUCED, LABEL_COMP, LABELS_MIXTURES, read_segmentation
from sklearn.mixture import GaussianMixture


def edges_seams(seams, tex_res, edge_idx):
    edges = np.zeros((0, 2), dtype=np.int32)

    for _, e0, _, e1 in seams:
        idx0 = np.array(edge_idx[e0][0]) * tex_res + np.array(edge_idx[e0][1])
        idx1 = np.array(edge_idx[e1][0]) * tex_res + np.array(edge_idx[e1][1])

        if len(idx0) and len(idx1):
            if idx0.shape[0] < idx1.shape[0]:
                idx0 = cv2.resize(idx0.reshape(-1, 1), (1, idx1.shape[0]), interpolation=cv2.INTER_NEAREST)
            elif idx0.shape[0] > idx1.shape[0]:
                idx1 = cv2.resize(idx1.reshape(-1, 1), (1, idx0.shape[0]), interpolation=cv2.INTER_NEAREST)

            edges_new = np.hstack((idx0.reshape(-1, 1), idx1.reshape(-1, 1)))
            edges = np.vstack((edges, edges_new))

    edges = np.sort(edges, axis=1)

    return edges[:, 0], edges[:, 1]


def main(unwrap_dir, segm_out_file, gmm_out_file):
    iso_files = np.array(sorted(glob(os.path.join(unwrap_dir, '*_unwrap.jpg'))))
    segm_files = np.array(sorted(glob(os.path.join(unwrap_dir, '*_segm.png'))))
    vis_files = np.array(sorted(glob(os.path.join(unwrap_dir, '*_visibility.jpg'))))

    iso_mask = cv2.imread('assets/tex_mask_1000.png', flags=cv2.IMREAD_GRAYSCALE) / 255.
    iso_mask = cv2.resize(iso_mask, (1000, 1000), interpolation=cv2.INTER_NEAREST)

    voting = np.zeros((1000, 1000, len(LABELS_REDUCED)))

    gmms = {}
    gmm_pixels = {}

    for color_id in LABELS_REDUCED:
        gmms[color_id] = GaussianMixture(LABELS_MIXTURES[color_id])
        gmm_pixels[color_id] = []

    for frame_file, segm_file, vis_file in zip(iso_files, segm_files, vis_files):
        print('extract from {}...'.format(os.path.basename(frame_file)))

        frame = cv2.cvtColor(cv2.imread(frame_file), cv2.COLOR_BGR2HSV) / 255.
        tex_segm = read_segmentation(segm_file)
        tex_weights = 1 - cv2.imread(vis_file) / 255.
        tex_weights = np.sqrt(tex_weights)

        for i, color_id in enumerate(LABELS_REDUCED):
            if color_id != 'Unseen' and color_id != 'BG':
                where = np.all(tex_segm == LABELS_REDUCED[color_id], axis=2)
                voting[where, i] += tex_weights[where, 0]
                gmm_pixels[color_id].extend(frame[where].tolist())

    for color_id in LABELS_REDUCED:
        if gmm_pixels[color_id]:
            print('GMM fit {}...'.format(color_id))
            gmms[color_id].fit(np.array(gmm_pixels[color_id]))

    for i, color_id in enumerate(LABELS_REDUCED):
        if color_id == 'Unseen' or color_id == 'BG':
            voting[:, i] = -10

    voting[iso_mask == 0] = 0
    voting[iso_mask == 0, 0] = 1

    unaries = np.ascontiguousarray((1 - voting / len(iso_files)) * 10)
    pairwise = np.ascontiguousarray(LABEL_COMP)

    seams = np.load('assets/basicModel_seams.npy')
    edge_idx = pkl.load(open('assets/basicModel_edge_idx_1000.pkl', 'rb'))

    dr_v = signal.convolve2d(iso_mask, [[-1, 1]])[:, 1:]
    dr_h = signal.convolve2d(iso_mask, [[-1], [1]])[1:, :]

    where_v = iso_mask - dr_v
    where_h = iso_mask - dr_h

    idxs = np.arange(1000 ** 2).reshape(1000, 1000)
    v_edges_from = idxs[:-1, :][where_v[:-1, :] == 1].flatten()
    v_edges_to = idxs[1:, :][where_v[:-1, :] == 1].flatten()
    h_edges_from = idxs[:, :-1][where_h[:, :-1] == 1].flatten()
    h_edges_to = idxs[:, 1:][where_h[:, :-1] == 1].flatten()

    s_edges_from, s_edges_to = edges_seams(seams, 1000, edge_idx)

    edges_from = np.r_[v_edges_from, h_edges_from, s_edges_from]
    edges_to = np.r_[v_edges_to, h_edges_to, s_edges_to]
    edges_w = np.r_[np.ones_like(v_edges_from), np.ones_like(h_edges_from), np.ones_like(s_edges_from)]

    gc = gco.GCO()
    gc.create_general_graph(1000 ** 2, pairwise.shape[0], True)
    gc.set_data_cost(unaries.reshape(1000 ** 2, pairwise.shape[0]))

    gc.set_all_neighbors(edges_from, edges_to, edges_w)
    gc.set_smooth_cost(pairwise)
    gc.swap(-1)

    labels = gc.get_labels().reshape(1000, 1000)
    gc.destroy_graph()

    segm_colors = np.zeros((1000, 1000, 3), dtype=np.uint8)

    for i, color_id in enumerate(LABELS_REDUCED):
        segm_colors[labels == i] = LABELS_REDUCED[color_id]

    cv2.imwrite('{}'.format(segm_out_file), segm_colors[:, :, ::-1])
    pkl.dump(gmms, open(gmm_out_file, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'unwrap_dir',
        type=str,
        help="Directory that contains unwrap files")

    parser.add_argument(
        'segm_out_file',
        type=str,
        help="Output file for segmentation")

    parser.add_argument(
        'gmm_out_file',
        type=str,
        help="Output file for GMMs")

    args = parser.parse_args()

    main(args.unwrap_dir, args.segm_out_file, args.gmm_out_file)
