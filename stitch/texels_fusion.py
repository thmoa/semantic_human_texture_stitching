#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import gco
import cv2
import os
import numpy as np
import cPickle as pkl

from scipy import signal
from skimage import color
from skimage.color import delta_e

from render.renderer import OrthoColoredRenderer
from render.camera import OrthoProjectPoints

from opendr.topology import get_faces_per_edge


class Stitcher:

    def __init__(self, seams, tex_res, mask, edge_idx_file='assets/basicModel_edge_idx_1000.pkl'):
        self.tex_res = tex_res
        self.seams = seams
        self.edge_idx = pkl.load(open(edge_idx_file, 'rb'))

        dr_v = signal.convolve2d(mask, [[-1, 1]])[:, 1:]
        dr_h = signal.convolve2d(mask, [[-1], [1]])[1:, :]

        self.where_v = mask - dr_v
        self.where_h = mask - dr_h

        idxs = np.arange(tex_res ** 2).reshape(tex_res, tex_res)
        v_edges_from = idxs[:-1, :][self.where_v[:-1, :] == 1].flatten()
        v_edges_to = idxs[1:, :][self.where_v[:-1, :] == 1].flatten()
        h_edges_from = idxs[:, :-1][self.where_h[:, :-1] == 1].flatten()
        h_edges_to = idxs[:, 1:][self.where_h[:, :-1] == 1].flatten()

        self.s_edges_from, self.s_edges_to = self._edges_seams()

        self.edges_from = np.r_[v_edges_from, h_edges_from, self.s_edges_from]
        self.edges_to = np.r_[v_edges_to, h_edges_to, self.s_edges_to]

    def stich(self, im0, im1, unaries0, unaries1, labels0, labels1, pairwise_mask, segmentation):

        gc = gco.GCO()
        gc.create_general_graph(self.tex_res ** 2, 2, True)
        gc.set_data_cost(np.dstack((unaries0, unaries1)).reshape(-1, 2))

        edges_w = self._rgb_grad(im0, im1, labels0, labels1, pairwise_mask, segmentation)

        gc.set_all_neighbors(self.edges_from, self.edges_to, edges_w)
        gc.set_smooth_cost((1 - np.eye(2)) * 65)
        gc.swap()

        labels = gc.get_labels()
        gc.destroy_graph()

        labels = labels.reshape(self.tex_res, self.tex_res).astype(np.float32)
        label_maps = np.zeros((2, self.tex_res, self.tex_res))

        for l in range(2):
            label_maps[l] = cv2.blur(np.float32(labels == l), (self.tex_res / 100, self.tex_res / 100))  # TODO

        norm_masks = np.sum(label_maps, axis=0)
        result = (np.atleast_3d(label_maps[0]) * im0 + np.atleast_3d(label_maps[1]) * im1)
        result[norm_masks != 0] /= np.atleast_3d(norm_masks)[norm_masks != 0]

        return result, labels

    def _edges_seams(self):
        edges = np.zeros((0, 2), dtype=np.int32)

        for _, e0, _, e1 in self.seams:
            idx0 = np.array(self.edge_idx[e0][0]) * self.tex_res + np.array(self.edge_idx[e0][1])
            idx1 = np.array(self.edge_idx[e1][0]) * self.tex_res + np.array(self.edge_idx[e1][1])

            if len(idx0) and len(idx1):
                if idx0.shape[0] < idx1.shape[0]:
                    idx0 = cv2.resize(idx0.reshape(-1, 1), (1, idx1.shape[0]), interpolation=cv2.INTER_NEAREST)
                elif idx0.shape[0] > idx1.shape[0]:
                    idx1 = cv2.resize(idx1.reshape(-1, 1), (1, idx0.shape[0]), interpolation=cv2.INTER_NEAREST)

                edges_new = np.hstack((idx0.reshape(-1, 1), idx1.reshape(-1, 1)))
                edges = np.vstack((edges, edges_new))

        edges = np.sort(edges, axis=1)

        return edges[:, 0], edges[:, 1]

    def _rgb_grad(self, im0, im1, labels0, labels1, pairwise_mask, segmentation):
        gray0 = color.rgb2gray(im0) * pairwise_mask
        gray1 = color.rgb2gray(im1) * pairwise_mask

        grad0 = np.abs(gray0.flatten()[self.edges_from] - gray1.flatten()[self.edges_to])
        grad1 = np.abs(gray1.flatten()[self.edges_from] - gray0.flatten()[self.edges_to])

        label_grad = np.logical_not(np.equal(labels0.flatten()[self.edges_from], labels1.flatten()[self.edges_to]))
        if segmentation is not None:
            seg_grad = np.equal(segmentation.flatten()[self.edges_from], segmentation.flatten()[self.edges_to])
        else:
            seg_grad = 1.

        return np.maximum(grad0, grad1) * np.float32(label_grad) * np.float32(seg_grad)
