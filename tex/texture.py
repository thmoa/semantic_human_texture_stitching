#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import cPickle as pkl

from skimage.measure import compare_ssim

from stitch.texels_fusion import Stitcher
from iso import Isomapper
from util.visibility import VisibilityRenderer
from util.labels import LABELS_REDUCED, to_ids


class TextureData:
    def __init__(self, tex_res, f, vt, ft, visibility):
        self.tex_res = tex_res
        self.visibility = visibility
        self.f = f

        self.isomapper = Isomapper(vt, ft, tex_res)
        self.iso_nearest = Isomapper(vt, ft, tex_res, bgcolor=np.array(LABELS_REDUCED['Unseen']) / 255.)
        self.mask = self.isomapper.iso_mask

        self.visibility_rn = VisibilityRenderer(vt, ft, tex_res, f)

    def get_data(self, frame, camera, silh, segm):
        f_vis = self.visibility.face_visibility(camera, silh)
        vis = self.visibility_rn.render(self._vis_angle(camera, silh))

        iso = self.isomapper.render(frame, camera, self.f, visible_faces=f_vis)
        iso_segm = self.iso_nearest.render(segm, camera, self.f, visible_faces=f_vis, inpaint=False)

        return vis, iso, iso_segm

    def _vis_angle(self, camera, silh):
        v_vis = self.visibility.vertex_visibility(camera, silh)
        v_angle = self.visibility.vertex_visibility_angle(camera)
        v_angle[np.logical_not(v_vis)] = 0

        return 1 - v_angle ** 2


class Texture:

    def __init__(self, tex_res, seams, mask, segm_template, gmm):
        self.tex_res = tex_res
        self.mask = mask
        self.face_mask = cv2.imread('assets/tex_face_mask_1000.png', flags=cv2.IMREAD_GRAYSCALE) / 255.
        self.face_mask = cv2.resize(self.face_mask, (tex_res, tex_res), interpolation=cv2.INTER_NEAREST)

        self.stitcher = Stitcher(seams, tex_res, self.mask)

        self.segm_template = None
        self.segm_template_id = None
        self.gmms = None

        self.segm_template = segm_template
        self.segm_template_id = to_ids(self.segm_template)
        self.gmms = gmm

        self.tex_agg = None
        self.vis_agg = None
        self.gmm_agg = None

    def add_iso(self, tex_current, vis, current_label, silh_err=0., inpaint=True):

        if self.tex_agg is None:
            self.vis_agg = vis
            self.tex_agg = tex_current
            self.silh_err_agg = np.ones_like(vis) * silh_err
            self.labels_agg = np.ones_like(vis) * current_label
            self.gmm_agg = np.zeros((self.tex_res, self.tex_res))
            self.init_face = np.mean(self.tex_agg, axis=2)

            if inpaint:
                return self.inpaint_segments(self.tex_agg, self.vis_agg), self.labels_agg
            else:
                return self.tex_agg, self.labels_agg

        pairwise_mask = np.logical_or(self.vis_agg < 1, vis < 1)

        _, ssim = compare_ssim(self.init_face * self.face_mask, np.mean(tex_current, axis=2) * self.face_mask,
                               full=True, data_range=1)
        ssim = (1 - ssim) / 2.
        ssim[ssim < 0] = 0

        gmm = np.zeros((self.tex_res, self.tex_res))
        self.gmm_agg = np.zeros((self.tex_res, self.tex_res))

        tex_agg_hsv = cv2.cvtColor(np.uint8(self.tex_agg * 255), cv2.COLOR_RGB2HSV) / 255.
        tex_current_hsv = cv2.cvtColor(np.uint8(tex_current * 255), cv2.COLOR_RGB2HSV) / 255.

        if self.segm_template is not None and self.gmms is not None:
            for i, color_id in enumerate(LABELS_REDUCED):
                if color_id != 'Unseen' and color_id != 'BG':
                    where = np.all(self.segm_template == LABELS_REDUCED[color_id], axis=2)
                    w = 10. if color_id in ['Arms', 'Legs'] else 1.

                    if np.max(where):
                        c = self.gmms[color_id].n_components

                        data = tex_current_hsv[where]
                        diff = data.reshape(-1, 1, 3) - self.gmms[color_id].means_
                        mahal = np.sqrt(np.sum((np.sum(diff.reshape(-1, c, 1, 3) * self.gmms[color_id].covariances_, axis=3) * diff), axis=2))
                        gmm[where] = np.min(mahal, axis=1) * w

                        data = tex_agg_hsv[where]
                        diff = data.reshape(-1, 1, 3) - self.gmms[color_id].means_
                        mahal = np.sqrt(np.sum((np.sum(diff.reshape(-1, c, 1, 3) * self.gmms[color_id].covariances_, axis=3) * diff), axis=2))
                        self.gmm_agg[where] = np.min(mahal, axis=1) * w

        unaries_agg = 2. * self.vis_agg + 20 * self.gmm_agg + 0.7 * self.silh_err_agg
        unaries_current = 2. * vis + 20 * gmm + 10 * ssim + 0.7 * silh_err

        labels_current = np.ones_like(self.labels_agg) * current_label

        self.tex_agg, update = self.stitcher.stich(self.tex_agg, tex_current, unaries_agg, unaries_current,
                                                   self.labels_agg, labels_current, pairwise_mask, self.segm_template_id)
        self.vis_agg[update == 1] = vis[update == 1]
        self.silh_err_agg[update == 1] = silh_err
        self.labels_agg[update == 1] = current_label

        if inpaint:
            return self._grow_tex(self.inpaint_segments(self.tex_agg, self.vis_agg)), self.labels_agg * self.mask
        else:
            return self.tex_agg, self.labels_agg * self.mask

    def inpaint_segments(self, tex, vis):

        if self.segm_template_id is not None:
            visible = vis < 0.95

            tmp = np.array(tex)
            for i, l in enumerate(LABELS_REDUCED):
                if l != 'Unseen' and l != 'BG':
                    seen = np.float32(np.logical_and(visible, self.segm_template_id == i))
                    area = 1 - cv2.erode(seen, np.ones((3, 3), dtype=np.uint8), iterations=2)

                    if np.max(seen):
                        part = cv2.inpaint(np.uint8(tex * 255), np.uint8(area * 255), 3, cv2.INPAINT_TELEA) / 255.
                        where = np.logical_and(np.logical_not(visible), self.segm_template_id == i)
                        tmp[where] = part[where]

            return tmp

        return self.inpaint(tex, vis)

    def inpaint(self, tex, vis):

        visible = np.float32(vis < 0.7)
        visible[self.mask < 1] = 0

        area = cv2.dilate(1 - visible, np.ones((3, 3), dtype=np.uint8), iterations=2)
        tex = cv2.inpaint(np.uint8(tex * 255), np.uint8(area * 255), 3, cv2.INPAINT_TELEA) / 255.

        return tex

    def _grow_tex(self, tex):
        kernel_size = np.int(self.vis_agg.shape[1] * 0.005)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        inpaint_area = cv2.dilate(1 - self.mask, np.ones((3, 3), dtype=np.uint8), iterations=3)

        tex_inpaint = cv2.inpaint(np.uint8(tex * 255), np.uint8(inpaint_area * 255), 3, cv2.INPAINT_TELEA)
        return (tex_inpaint * np.atleast_3d(cv2.dilate(self.mask, kernel))) / 255.
