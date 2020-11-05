#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import chumpy as ch
from chumpy import Ch, depends_on
from chumpy.utils import col
from opendr.geometry import Rodrigues


class OrthoProjectPoints(Ch):
    terms = 'near', 'far', 'width', 'height'
    dterms = 'v', 'rt', 't', 'left', 'right', 'bottom', 'top'

    def compute_r(self):
        """
        Compute the r_and_derivatives

        Args:
            self: (todo): write your description
        """
        return self.r_and_derivatives.r

    def compute_dr_wrt(self, wrt):
        """
        R compute of this rectangle.

        Args:
            self: (todo): write your description
            wrt: (array): write your description
        """
        if wrt not in [self.v, self.rt, self.t, self.left, self.right, self.bottom, self.top]:
            return None

        return self.r_and_derivatives.dr_wrt(wrt)

    def unproject_points(self, uvd, camera_space=False):
        """
        Unprojects a 2d of the 2d.

        Args:
            self: (todo): write your description
            uvd: (todo): write your description
            camera_space: (str): write your description
        """
        tmp = np.hstack((
            col(2. * uvd[:, 0] / self.width - 1 + (self.right + self.left) / (self.right - self.left)).r * (self.right - self.left).r / 2.,
            col(2. * uvd[:, 1] / self.height - 1 + (self.bottom + self.top) / (self.bottom - self.top)).r * (self.bottom - self.top).r / 2.,
            np.ones((uvd.shape[0], 1))
        ))

        if camera_space:
            return tmp
        tmp -= self.t.r  # translate

        return tmp.dot(Rodrigues(self.rt).r.T)  # rotate

    @depends_on('t', 'rt')
    def view_mtx(self):
        """
        View the current view

        Args:
            self: (todo): write your description
        """
        R = cv2.Rodrigues(self.rt.r)[0]
        return np.hstack((R, col(self.t.r)))

    @property
    def r_and_derivatives(self):
        """
        Returns the derivative of the derivative of self.

        Args:
            self: (todo): write your description
        """
        tmp = self.v.dot(Rodrigues(self.rt)) + self.t

        return ch.hstack((
            col(2. / (self.right - self.left) * tmp[:, 0] - (self.right + self.left) / (self.right - self.left) + 1.) * self.width / 2.,
            col(2. / (self.bottom - self.top) * tmp[:, 1] - (self.bottom + self.top) / (self.bottom - self.top) + 1.) * self.height / 2.,
        ))

