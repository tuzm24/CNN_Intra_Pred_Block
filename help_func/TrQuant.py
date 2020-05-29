#Source VTM Version <8.0>

from help_func.RomTr import *
import numpy as np
import torch
from math import log2


JVET_C0024_ZERO_OUT_TH = 32
BIT_DEPTH = 10

torch_dtype_dic = {'float32':torch.float32, 'int32':torch.int}





class Transfrom:

    def __init__(self, isTorch=True, dtype='float32', isGpu='cpu', byBatch=False):
        self.isTorch = isTorch
        self.gpu = torch.device(isGpu)
        self.byBatch = byBatch
        self.Ttuple = self.getTtuple()
        self.dtype = dtype
        self.trans_array = TransArray
        self.trans_arrayT = TransArrayT
        self.dotfunc = self.dot()
        if self.isTorch:
            self.setTorch(self.trans_array)
            self.setTorch(self.trans_arrayT)
        self.FwdTrans = [self.ForwardDCT2, None, self.ForwardDST7]

    def setTorch(self, tdic):
        for key, value in tdic.items():
            tdic[key] = torch.tensor(value, dtype=torch_dtype_dic[self.dtype],device=self.gpu)
        return

    def getTtuple(self):
        if self.byBatch:
            return (0, -1, -2)
        return (-1, -2)

    def dot(self):
        if self.isTorch:
            if self.byBatch:
                return torch.bmm
            return torch.mm
        return np.dot


    def ForwardDCT2(self, src, shift, line, iSkipLine, iSkipLine2, length):
        iT = self.trans_arrayT['g_trCoreDCT2P' + str(length)][0]
        reduceLine = line - iSkipLine
        dst = self.dotfunc(src, iT).transpose(*self.Ttuple) / shift
        if iSkipLine:
            dst[..., reduceLine:] = 0
        return dst

    def ForwardDST7(self, src, shift, line, iSkipLine, iSkipLine2, length):
        iT = self.trans_arrayT['g_trCoreDST7P' + str(length)][0]
        reduceLine = line - iSkipLine
        cutoff = length - iSkipLine2
        dst = self.dotfunc(src, iT).transpose(*self.Ttuple) / shift
        if iSkipLine:
            dst[..., :cutoff, reduceLine:] = 0
        if iSkipLine2:
            dst[..., cutoff:, :] = 0
        return dst






    def xT(self, resi, trTypeHor, trTypeVer):
        height, width = resi.shape
        skipWidth = 16 if (trTypeHor!=TransType.DCT2 and width==32) else (width - JVET_C0024_ZERO_OUT_TH if width > JVET_C0024_ZERO_OUT_TH else 0)
        skipHeight = 16 if (trTypeVer!=TransType.DCT2 and height==32) else (height - JVET_C0024_ZERO_OUT_TH if height > JVET_C0024_ZERO_OUT_TH else 0)
        if width > 1 and height > 1:
            shift_1st = 2**(((int(log2(width))) + BIT_DEPTH + 6) - 15)
            shift_2nd = 2**((int(log2(height)))            + 6)
            tmp = self.FwdTrans[trTypeHor](resi, shift_1st, height, 0, skipWidth, width)
            coef = self.FwdTrans[trTypeVer](tmp, shift_2nd, width, skipWidth, skipHeight, height)
        elif height==1:
            shift = (int(log2(width))) + BIT_DEPTH + 6 - 15
            coef = self.FwdTrans[trTypeHor](resi, shift, 1, 0, skipWidth, width)
        else:
            shift = (int(log2(height))) + BIT_DEPTH + 6 - 15
            coef = self.FwdTrans[trTypeVer](resi, shift, 1, 0, skipHeight, height)
        return coef

T = Transfrom()
resi = torch.tensor(np.array(np.array(range(32*32)).reshape(32, 32)), dtype=torch.float32)
T.xT(resi, TransType.DCT2, TransType.DCT2)

# xT(np.array(range(64*32)).reshape((64,32)), None, 0, 0)