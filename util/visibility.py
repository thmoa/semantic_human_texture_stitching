#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from opendr.renderer import DepthRenderer
from opendr.camera import ProjectPoints3D
from opendr.geometry import VertNormals

from render.renderer import OrthoColoredRenderer
from render.camera import OrthoProjectPoints


class VisibilityChecker:
    def __init__(self, w, h, f):
        self.w = w
        self.h = h
        self.f = f
        self.rn_d = DepthRenderer(frustum={'near': 0.1, 'far': 10., 'width': w, 'height': h}, f=f)

    def vertex_visibility(self, camera, mask=None):
        cam3d = ProjectPoints3D(**{k: getattr(camera, k) for k in camera.dterms if hasattr(camera, k)})

        in_viewport = np.logical_and(
            np.logical_and(np.round(camera.r[:, 0]) >= 0, np.round(camera.r[:, 0]) < self.w),
            np.logical_and(np.round(camera.r[:, 1]) >= 0, np.round(camera.r[:, 1]) < self.h),
        )

        if not hasattr(self.rn_d, 'camera') or not np.all(self.rn_d.camera.r == camera.r):
            self.rn_d.set(camera=camera)
        depth = self.rn_d.r

        proj = cam3d.r[in_viewport]
        d = proj[:, 2]
        idx = np.round(proj[:, [1, 0]].T).astype(np.int).tolist()

        visible = np.zeros(cam3d.shape[0], dtype=np.bool)
        visible[in_viewport] = np.abs(d - depth[tuple(idx)]) < 0.01

        if mask is not None:
            mask = cv2.erode(mask, np.ones((5, 5)))
            visible[in_viewport] = np.logical_and(visible[in_viewport], mask[tuple(idx)])

        return visible

    def face_visibility(self, camera, mask=None):
        v_vis = self.vertex_visibility(camera, mask)

        return np.min(v_vis[self.f], axis=1)

    def vertex_visibility_angle(self, camera):
        n = VertNormals(camera.v, self.f)
        v_cam = camera.v.r.dot(cv2.Rodrigues(camera.rt.r)[0]) + camera.t.r
        n_cam = n.r.dot(cv2.Rodrigues(camera.rt.r)[0])

        return np.sum(v_cam / (np.linalg.norm(v_cam, axis=1).reshape(-1, 1)) * -1 * n_cam, axis=1)

    def face_visibility_angle(self, camera):
        v_cam = camera.v.r.dot(cv2.Rodrigues(camera.rt.r)[0]) + camera.t.r
        f_cam = v_cam[self.f]
        f_norm = np.cross(f_cam[:, 0] - f_cam[:, 1], f_cam[:, 0] - f_cam[:, 2], axisa=1, axisb=1)
        f_norm /= np.linalg.norm(f_norm, axis=1).reshape(-1, 1)
        center = np.mean(f_cam, axis=1)

        return np.sum(center / (np.linalg.norm(center, axis=1).reshape(-1, 1)) * -1 * f_norm, axis=1)


class VisibilityRenderer:
    def __init__(self, vt, ft, tex_res, f):
        ortho = OrthoProjectPoints(rt=np.zeros(3), t=np.zeros(3), near=-1, far=1, left=-0.5, right=0.5, bottom=-0.5,
                                   top=0.5, width=tex_res, height=tex_res)
        vt3d = np.dstack((vt[:, 0] - 0.5, 1 - vt[:, 1] - 0.5, np.zeros(vt.shape[0])))[0]
        vt3d = vt3d[ft].reshape(-1, 3)
        self.f = f
        self.rn = OrthoColoredRenderer(bgcolor=np.zeros(3), ortho=ortho, v=vt3d, f=np.arange(ft.size).reshape(-1, 3),
                                       num_channels=1)

    def render(self, vertex_visibility):
        vc = vertex_visibility.reshape(-1, 1)
        vc = np.hstack((vc, vc, vc))
        self.rn.set(vc=vc[self.f].reshape(-1, 3))

        return np.array(self.rn.r)

    def mask(self):
        self.rn.set(vc=np.ones_like(self.rn.v))
        return np.array(self.rn.r)