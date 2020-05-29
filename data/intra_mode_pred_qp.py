import os
from data import srdata
import glob
from help_func.CompArea import PictureFormat
from data import common
import imageio
import numpy as np
import random
from help_func.CompArea import TuList
from help_func.CompArea import TuDataIndex

import matplotlib.pyplot as plt
CTU_SIZE = 128


class intra_mode_pred_qp(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        self.end -=self.begin
        self.begin = 1
        super(intra_mode_pred_qp, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
        self.images_tu = self._scanTuData()
        self.idxes = [TuDataIndex.NAME_DIC[x] for x in self.args.tu_data_type]


    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.data_types]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.data_types):
                names_lr[si].append(os.path.join(
                    self.dir_lr, '{}/{}{}'.format(
                        s, filename, self.ext[1]
                    )
                ))

        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        def deltestpath(l):
            new_one = []
            for path in l:
                # if '1352' in path:
                new_one.append(path)
            return new_one
        if not self.train:
            names_hr = deltestpath(names_hr)
            for i in range(len(names_lr)):
                names_lr[i] = deltestpath(names_lr[i])


        return names_hr, names_lr

    def _scanTuData(self):
        named_tu = []
        s = self.args.tu_data
        for f in self.images_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            named_tu.append(os.path.join(
                self.dir_lr, '{}/{}{}'.format(
                    s, filename, self.ext[1]
                )
            ))
        return named_tu

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data)
        self.dir_hr = os.path.join(self.apath, 'TU')
        self.dir_lr = os.path.join(self.apath)
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.npz', '.npz')


    @staticmethod
    def read_npz_file(f):
        def UpSamplingChroma(UVPic):
            return UVPic.repeat(2, axis=0).repeat(2, axis=1)
        f = np.load(f)
        # return np.stack((f['Y'], UpSamplingChroma(f['Cb']), UpSamplingChroma(f['Cr'])), axis=2)
        return np.stack((f['Y'],), axis=2)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('npz') >= 0:
            lr = []
            for flr in self.images_lr:
                lr.append(self.read_npz_file(flr[idx]))
            h, w, _ = lr[0].shape
            hr = self.readTuInfo(f_hr, h, w,tree=('LUMA',))

            # lr = self.read_npz_file(f_lr)
        else:
            assert 0

        return lr, hr, filename

    def get_patch(self, lr, hr):
        def _get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
            ih, iw = args[1][0].shape[:2]

            if not input_large:
                p = scale if multi else 1
                tp = p * patch_size
                ip = tp // scale
            else:
                tp = patch_size
                ip = patch_size



            ix = random.randrange(0, (iw - ip + 1), 4)
            iy = random.randrange(0, (ih - ip + 1), 4)
            if not input_large:
                tx, ty = scale * ix, scale * iy
            else:
                tx, ty = ix, iy

            ret = [
                args[0][ty:ty + tp, tx:tx + tp, ...],
                [a[iy:iy + ip, ix:ix + ip, ...] for a in args[1]]
            ]

            return ret[0], ret[1], (ty, tp, tx, tp)

        scale = self.scale[self.idx_scale]
        tpy, tpx = hr.shape[:2]
        imgy, imgx = hr.shape[:2]
        ty, tx = 0, 0
        if self.train:
            hr, lr, (ty, tpy, tx, tpx) = _get_patch(
                hr, lr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=True
            )
            if self.args.no_augment: lr, hr = common.augment(lr, hr)

        return lr, hr, (ty, tpy, tx, tpx), (imgy, imgx)


    def getTuMask(self, ty, tpy, tx, tpx, h, w, filename):
        tulist = TuList(np.load(
            os.path.join(self.dir_lr, '{}/{}{}'.format(
                self.args.tu_data, os.path.splitext(os.path.basename(filename))[0], self.ext[1]
            ))
        )['LUMA'])
        return TuList.NormalizedbyMinMaxAll(tulist.getTuMaskFromIndexes(self.idxes, h, w)[:, ty:ty+tpy, tx:tx+tpx], self.idxes).astype('float32')


    def readTuInfo(self, filename, height, width, tree = ('LUMA', 'CHROMA')):
        input_tu_idx = (TuDataIndex.MODE,)
        if not len(tree):
            return
        np_load = np.load(
            os.path.join(self.dir_lr, '{}/{}{}'.format(
                self.args.tu_data, os.path.splitext(os.path.basename(filename))[0], self.ext[1]
            ))
        )
        arr = TuList(np_load['LUMA'])
        arr.setWideAngle()
        return [
            TuList(np_load[x], setIntraWide=x).getTuMaskFromIndexes(
                input_tu_idx, height, width, True if x=='CHROMA' else False)[0] for x in tree]

    @staticmethod
    def plotMap(img,filename):
        vminmax = (img.min(), img.max())
        fig, ax = plt.subplots()

        color_num = len(np.unique(img))
        cmaps = plt.cm.get_cmap('rainbow', color_num)
        cmaps.set_under('white')
        eps = np.spacing(1.0)

        imgs = ax.imshow((img).astype(int), vmin=eps, vmax=vminmax[1], interpolation='nearest',
                         cmap=cmaps)
        v1 = np.round(np.linspace(vminmax[0], vminmax[1], 10, endpoint=True))
        cb = fig.colorbar(imgs, ticks=v1)
        cb.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
        plt.savefig('{}.png'.format(filename), dpi=300)
        return


    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)

        lr, hr, pos, imgshape = self.get_patch(lr, hr[0])
        intra_mode_pred_qp.plotMap(hr, filename)
        hr = hr[::4,::4,...]
        extra_data = self.getTuMask(*pos, *imgshape, filename)
        # pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        # pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        return common.np2Tensor2(lr), common.np2Tensor3([hr])[0], extra_data, filename


