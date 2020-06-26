import numpy as np

from revgraph.core.functions.base.generic_function import GenericFunction, gradient_wrt_arg


class Conv2D(GenericFunction):
    """
    Reference: This implementation is hugely based on the repository
    https://www.github.com/renmengye/np-conv2d
    """
    def apply(self, a, b, padding='VALID', stride=(1, 1)) -> np.ndarray:
        # Buggy Implementation when padding='SAME'
        # if padding is 'SAME':
        #     raise NotImplementedError('SAME padding not implemented')
        # elif padding is not 'VALID':
        #     raise ValueError(f'Invalid padding option: {padding}')
        stride = self.convert_scalar_stride(stride)
        kernel_size = b.shape[:2]
        a = self.get_patches(a, kernel_size, padding, stride)
        n_, h_, w_, c_ = b.shape
        b = b.reshape((n_ * h_ * w_, c_))
        n, h, w, *_ = a.shape
        a = a.reshape((n * h * w, -1))
        output = a.dot(b).reshape((n, h, w, -1))
        return output

    @gradient_wrt_arg(0)
    def da(self, gradient, a, b, padding='VALID', stride=(1, 1)) -> np.ndarray:
        stride = self.convert_scalar_stride(stride)
        kh, kw = b.shape[:2]
        _, h, w, _ = a.shape

        if padding == 'SAME':
            gradient_shape = gradient.shape[1:3]
            pad2h = int(self.pad_size('SAME',
                                      max(gradient_shape[0],
                                          gradient_shape[0] * stride[0] - 1),
                                      h, 1, kh))
            pad2w = int(self.pad_size('SAME',
                                      max(gradient_shape[0],
                                          gradient_shape[0] * stride[1] - 1),
                                      w, 1, kw))
            pad2 = (pad2h, pad2w)
        elif padding == 'VALID':
            pad2h = int(self.pad_size('SAME', 0, 0, 1, kh)) * 2
            pad2w = int(self.pad_size('SAME', 0, 0, 1, kw)) * 2
            pad2 = (pad2h, pad2w)
        else:
            pad2 = padding
        b = np.transpose(b, (0, 1, 3, 2))
        da = self.get_patches_da(gradient, a, b, pad2, stride)
        x, y, z, *_ = da.shape
        da = da.reshape((x * y * z, -1))
        b = b[::-1, ::-1, :, :]
        m, n, o, p = b.shape
        b = b.reshape((m * n * o, p))
        return da.dot(b).reshape((x, y, z, -1))

    @gradient_wrt_arg(1)
    def db(self, gradient, a, b, padding='VALID', stride=(1, 1)) -> np.ndarray:
        stride = self.convert_scalar_stride(stride)
        kernel_size = b.shape[:2]
        gradient = np.transpose(gradient, (1, 2, 0, 3))
        a = np.transpose(a, (3, 1, 2, 0))
        a = self.get_patches_db(gradient, a, b, padding, stride)
        x, y, z, m = gradient.shape
        gradient = gradient.reshape((x*y*z, m))
        i, j, k, *_ = a.shape
        return a.reshape((i*j*k, -1))\
                .dot(gradient)\
                .reshape((i, j, k, -1))\
                .transpose((1, 2, 0, 3))[:kernel_size[0], :kernel_size[1], :, :]

    def get_patches(self, a, kernel_size, padding, stride):
        n, h, w, c = a.shape
        kh, kw = kernel_size
        sh, sw = stride

        h2 = int(self.output_size(h, kh, padding, sh))
        w2 = int(self.output_size(w, kw, padding, sw))
        ph = int(self.pad_size(padding, h, h2, sh, kh))
        pw = int(self.pad_size(padding, w, w2, sw, kw))

        ph0 = int(np.floor(ph / 2))
        ph1 = int(np.ceil(ph / 2))
        pw0 = int(np.floor(pw / 2))
        pw1 = int(np.ceil(pw / 2))

        pph = (ph0, ph1)
        ppw = (pw0, pw1)

        a = np.pad(a,
                   ((0, 0), pph, ppw, (0, 0)),
                   mode='constant',
                   constant_values=(0.0,))
        a_sn, a_sh, a_sw, a_sc = a.strides
        y_strides = (a_sn, sh * a_sh, sw * a_sw, a_sh, a_sw, a_sc)
        return np.ndarray((n, h2, w2, kh, kw, c),
                          dtype=a.dtype,
                          buffer=a.data,
                          offset=self.array_offset(a),
                          strides=y_strides)

    def get_patches_da(self, g, a, b, padding, stride):
        kernel_size = b.shape[:2]
        n, h, w, c = g.shape
        kh, kw = kernel_size
        ph, pw = padding
        sh, sw = stride
        _, ah, aw, _ = a.shape
        gs = np.zeros((n, h, sh, w, sw, c))
        gs[:, :, 0, :, 0, :] = g
        (i, j, k, x, y, z) = gs.shape
        g = gs.reshape((i, j * k, x * y, z))[:, :ah, :aw, :]

        ph2 = int(np.ceil(ph / 2))
        ph3 = int(np.floor(ph / 2))
        pw2 = int(np.ceil(pw / 2))
        pw3 = int(np.floor(pw / 2))

        pph = (ph2, ph3)
        ppw = (pw2, pw3)

        padding = ((0, 0),
                   pph,
                   ppw,
                   (0, 0))
        g = np.pad(g,
                   padding,
                   mode='constant',
                   constant_values=(0.0,))
        g_sn, g_sh, g_sw, g_sc = g.strides
        y_strides = (g_sn, g_sh, g_sw, g_sh, g_sw, g_sc)
        return np.ndarray((n, ah, aw, kh, kw, c),
                          dtype=g.dtype,
                          buffer=g.data,
                          offset=self.array_offset(g),
                          strides=y_strides)

    def get_patches_db(self, gradient, a, b, padding, stride):
        n, h, w, c = a.shape
        kh, kw = gradient.shape[:2]
        sh, sw = stride
        bh, bw, *_ = b.shape
        ph = int(self.pad_size(padding, h, bh, 1, ((kh-1) * sh + 1)))
        pw = int(self.pad_size(padding, w, bw, 1, ((kw-1) * sw + 1)))

        ph2 = int(np.ceil(ph / 2))
        ph3 = int(np.floor(ph / 2))
        pw2 = int(np.ceil(pw / 2))
        pw3 = int(np.floor(pw / 2))

        padding = ((0, 0),
                   (ph3, ph2),
                   (pw3, pw2),
                   (0, 0))

        a = np.pad(a,
                   padding,
                   mode='constant',
                   constant_values=(0.0,))
        p2h = (-a.shape[1]) % sh
        p2w = (-a.shape[2]) % sw
        if p2h > 0 or p2w > 0:
            a = np.pad(a,
                       ((0, 0), (0, p2h), (0, p2w), (0, 0)),
                       mode='constant',
                       constant_values=(0.0,))
        a_sn, a_sh, a_sw, a_sc = a.strides
        y_strides = (a_sn, a_sh, a_sw, sh * a_sh, sw * a_sw, a_sc)
        return np.ndarray((n, bh, bw, kh, kw, c),
                          dtype=a.dtype,
                          buffer=a.data,
                          offset=self.array_offset(a),
                          strides=y_strides)

    @staticmethod
    def pad_size(padding, in_size, out_size, stride, kernel_size):
        if padding == 'SAME':
            return (out_size - 1) * stride + kernel_size - in_size
        elif padding == 'VALID':
            return 0
        else:
            return padding

    @staticmethod
    def array_offset(x):
        if x.base is None:
            return 0
        base_start = x.base.__array_interface__['data'][0]
        start = x.__array_interface__['data'][0]
        return start - base_start

    @staticmethod
    def output_size(height, kernel_height, padding, stride_height):
        if padding == 'VALID':
            return np.ceil((height - kernel_height + 1) / stride_height)
        elif padding == 'SAME':
            return np.ceil(height / stride_height)
        else:
            return int(np.ceil((height - kernel_height + padding + 1) / stride_height))

    @staticmethod
    def convert_scalar_stride(stride):
        if isinstance(stride, int):
            return stride, stride
        return stride
