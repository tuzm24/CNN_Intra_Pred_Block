import os
from data import srdata
import glob
from help_func.CompArea import PictureFormat
from data import common
import numpy as np

class YUVData(srdata.SRData):
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
        super(YUVData, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

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
                new_one.append(path)
            return new_one
        if not self.train:
            names_hr = deltestpath(names_hr)
            for i in range(len(names_lr)):
                names_lr[i] = deltestpath(names_lr[i])


        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data)
        self.dir_hr = os.path.join(self.apath, PictureFormat.INDEX_DIC[PictureFormat.ORIGINAL])
        self.dir_lr = os.path.join(self.apath)
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.npz', '.npz')

    def __getitem__(self, idx):

        lr, hr, filename = self._load_file(idx)
        lr, hr = self.get_patch(lr, hr)
        # pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        # pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return common.np2Tensor2(lr), common.np2Tensor2([hr])[0], np.array(float('Nan')), filename


