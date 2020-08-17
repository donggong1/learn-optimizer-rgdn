import numpy as np
from scipy.interpolate import splrep, splev
from scipy.signal import convolve2d

# spline-based blur kernel simulation
# Python implementation of the kernel simulation method in "A. Chakrabarti, “A neural approach to blind motion deblurring,” in European Conference on Computer Vision (ECCV), 2016".
def kernel_sim_spline(psz, mxsz, nc, num):
    k = np.zeros([mxsz, mxsz, num], dtype=np.float32)
    imp = np.zeros([mxsz, mxsz], dtype=np.float32)
    imp[(mxsz + 1) // 2, (mxsz + 1) // 2] = 1.0

    xg, yg = np.meshgrid(np.arange(0, psz), np.arange(0, psz))

    for i in range(num):
        while True:
            x = np.random.randint(0, psz, nc)
            y = np.random.randint(0, psz, nc)

            # print(x.shape)
            # print(np.linspace(0, 1, nc).shape)
            order = min(nc - 1, 3)

            spx = splrep(np.linspace(0, 1, nc), x.astype(np.float), k=order)
            x = splev(np.linspace(0, 1, nc * 5000), spx)
            x = np.clip(x, 0, psz - 1)
            x = np.round(x).astype(np.int32)

            spy = splrep(np.linspace(0, 1, nc), y.astype(np.float), k=order)
            y = splev(np.linspace(0, 1, nc * 5000), spy)
            y = np.clip(y, 0, psz - 1)
            y = np.round(y).astype(np.int32)

            idx = x * psz + y
            idx = np.unique(idx)

            weight = np.random.randn(np.prod(idx.shape)) * 0.5 + 1
            weight = np.clip(weight, 0, None)

            if (np.sum(weight) == 0):
                continue

            weight = weight / np.sum(weight)
            kernel = np.zeros([psz * psz])

            kernel[idx] = weight

            kernel = np.reshape(kernel, [psz, psz])

            cx = int(np.round(np.sum(kernel * xg)))
            cy = int(np.round(np.sum(kernel * yg)))

            if cx <= psz / 2:
                padding = np.zeros([psz, psz - 2 * cx + 1])
                kernel = np.concatenate([padding, kernel], axis=1)
            else:
                padding = np.zeros([psz, 2 * cx - psz - 1])
                kernel = np.concatenate([kernel, padding], axis=1)

            p2 = kernel.shape[1]

            if cy <= psz / 2:
                padding = np.zeros([psz - 2 * cy + 1, p2])
                kernel = np.concatenate([padding, kernel], axis=0)
            else:
                padding = np.zeros([2 * cy - psz - 1, p2])
                kernel = np.concatenate([kernel, padding], axis=0)
            if np.max(kernel.shape) <= mxsz:
                break
        kernel = kernel.astype(np.float32)

        ck = convolve2d(imp, kernel, 'same')
        k[:, :, i] = ck
    return k


# test
if __name__ == '__main__':
    k_num_per_img = 5
    num_spl_ctrl = 6
    k_size = 41
    sp_size = 31

    k = kernel_sim_spline(sp_size, k_size, num_spl_ctrl, k_num_per_img)
