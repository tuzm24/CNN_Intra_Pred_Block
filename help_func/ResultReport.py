import numpy as np
import pandas as pd
from io import StringIO
import math
from collections import namedtuple
from parse import parse
import os

class Reporting:


    def __init__(self, init_text=None):
        if init_text is None:
            self.init_text = """seq,qp,bitrate,ypsnr,upsnr,vpsnr,encT,decT,isanchor,isCTC
Tango2_3840x2160_60fps_10bit_420,22,18105.17969,41.941502,49.799198,48.682301,14808.44336,7.456
Tango2_3840x2160_60fps_10bit_420,27,5442.075195,40.386501,48.797501,47.075401,5454.808105,5.872
Tango2_3840x2160_60fps_10bit_420,32,3022.155029,39.567101,47.817001,45.745499,1798.703979,5.297
Tango2_3840x2160_60fps_10bit_420,37,1797.689941,38.396301,46.823399,44.473099,654.070984,4.642
FoodMarket4_3840x2160_60fps_10bit_420,22,5110.814941,45.9743,51.243,52.625,4531.777832,5.757
FoodMarket4_3840x2160_60fps_10bit_420,27,2816.01001,45.261501,49.789001,50.651501,1040.017944,4.092
FoodMarket4_3840x2160_60fps_10bit_420,32,1810.560059,44.278801,48.335201,48.761398,441.403015,4.729
FoodMarket4_3840x2160_60fps_10bit_420,37,1198.275024,42.709599,47.033199,47.160301,160.686005,4.281
Campfire_3840x2160_30fps_10bit_420_bt709_videoRange,22,20806.69531,43.583801,45.9613,43.8652,14815.29688,8.793
Campfire_3840x2160_30fps_10bit_420_bt709_videoRange,27,9233.415039,39.996399,42.185001,41.4757,9323.134766,6.975
Campfire_3840x2160_30fps_10bit_420_bt709_videoRange,32,3845.294922,37.588799,39.498199,39.743,5268.473145,5.701
Campfire_3840x2160_30fps_10bit_420_bt709_videoRange,37,2100.060059,35.911301,37.542999,38.3297,2489.5,4.863
CatRobot1_3840x2160p_60_10_709_420,22,33470.57813,42.492001,42.492199,44.1483,17230.87891,7.985
CatRobot1_3840x2160p_60_10_709_420,27,11436.64453,39.966599,41.4165,42.567799,9017.143555,7.027
CatRobot1_3840x2160p_60_10_709_420,32,6597.089844,38.585999,40.686001,41.196701,4300.125,5.828
CatRobot1_3840x2160p_60_10_709_420,37,3937.050049,36.714901,39.989498,39.992401,1984.55896,5.217
DaylightRoad2_3840x2160_60fps_10bit_420,22,57475.98047,41.346298,46.023499,43.962101,21328.7168,6.578
DaylightRoad2_3840x2160_60fps_10bit_420,27,11722.16992,37.653999,44.846699,42.417099,12153.48047,6.233
DaylightRoad2_3840x2160_60fps_10bit_420,32,6266.399902,36.6563,43.8279,41.190701,5092.028809,5.748
DaylightRoad2_3840x2160_60fps_10bit_420,37,3483.975098,35.2421,42.973598,40.2873,2197.89209,4.971
ParkRunning3_3840x2160_50fps_10bit_420,22,49081.46094,46.506302,40.8013,41.219002,14879.04297,10.888
ParkRunning3_3840x2160_50fps_10bit_420,27,27797.32422,42.660801,37.014301,37.879902,10245.6377,6.652
ParkRunning3_3840x2160_50fps_10bit_420,32,15390.5,38.957298,34.395901,35.6106,7205.867188,7.525
ParkRunning3_3840x2160_50fps_10bit_420,37,8381.599609,35.504501,32.695099,34.075298,4076.466064,6.903
MarketPlace_1920x1080_60fps_10bit_420,22,8829.089844,42.5681,45.165001,45.987099,4506.062012,2.251
MarketPlace_1920x1080_60fps_10bit_420,27,4259.205078,40.057499,43.348598,44.5089,2316.010986,1.895
MarketPlace_1920x1080_60fps_10bit_420,32,2242.665039,37.8046,42.087898,43.239601,1374.706055,1.409
MarketPlace_1920x1080_60fps_10bit_420,37,1153.094971,35.406799,41.176998,42.2089,621.171997,1.431
RitualDance_1920x1080_60fps_10bit_420,22,3345.554932,48.047901,49.321701,50.6413,1536.73999,1.761
RitualDance_1920x1080_60fps_10bit_420,27,2018.984985,45.509899,46.807598,47.972599,963.745972,1.526
RitualDance_1920x1080_60fps_10bit_420,32,1208.910034,42.5998,44.921501,45.939999,635.236023,1.329
RitualDance_1920x1080_60fps_10bit_420,37,710.820007,39.520901,43.5261,44.436901,375.317993,1.186
Cactus_1920x1080_50,22,13100.15039,41.147499,41.6465,44.1786,6204.346191,2.649
Cactus_1920x1080_50,27,5735.737305,38.167599,39.473099,42.2104,3813.897949,2.263
Cactus_1920x1080_50,32,3113.162598,35.978001,38.544399,40.637402,2310.529053,1.299
Cactus_1920x1080_50,37,1689.3125,33.567001,37.764999,39.441399,1263.937012,1.6
BasketballDrive_1920x1080_50,22,5080.9375,41.3475,45.568501,47.542,4642.691895,2.047
BasketballDrive_1920x1080_50,27,2136.425049,39.6894,44.3801,45.621399,2234.303955,1.773
BasketballDrive_1920x1080_50,32,1150.150024,38.356899,43.280399,43.876701,997.271973,1.539
BasketballDrive_1920x1080_50,37,660.662476,36.691299,42.256401,42.494598,413.489014,1.313
BQTerrace_1920x1080_60,22,20900.95508,42.857201,43.132301,44.7481,3795.087891,2.102
BQTerrace_1920x1080_60,27,10832.80469,38.079201,40.987999,42.866299,3751.354004,2.606
BQTerrace_1920x1080_60,32,5872.22998,34.918499,39.4081,41.539902,2868.48999,1.612
BQTerrace_1920x1080_60,37,3175.39502,31.981701,38.340401,40.626099,1935.970947,1.62
BasketballDrill_832x480_50,22,2183.649902,42.330299,44.068298,44.922798,868.971985,0.685
BasketballDrill_832x480_50,27,1157.050049,38.981899,41.584801,42.2183,659.039001,0.516
BasketballDrill_832x480_50,32,618.099976,36.036499,39.8367,40.3755,478.970001,0.401
BasketballDrill_832x480_50,37,344.4375,33.402802,38.538399,38.686401,247.806,0.337
BQMall_832x480_60,22,3241.725098,42.0588,43.766701,45.295502,1025.847046,0.665
BQMall_832x480_60,27,2006.609985,38.988899,41.4538,42.748299,731.479004,0.536
BQMall_832x480_60,32,1205.655029,35.6992,39.828899,40.870399,598.065979,0.447
BQMall_832x480_60,37,672.164978,32.288799,38.569901,39.403,433.424011,0.364
PartyScene_832x480_50,22,5873,41.445,42.1576,42.824902,1264.069946,0.877
PartyScene_832x480_50,27,3743.162598,36.974998,39.146999,39.692799,1038.224976,0.795
PartyScene_832x480_50,32,2270.324951,33.009201,37.220402,37.551102,842.210022,0.633
PartyScene_832x480_50,37,1243.875,29.181601,35.806,36.063301,679.432983,0.504
RaceHorses_832x480_30,22,1945.522461,42.696899,43.452301,44.089199,1068.859985,0.524
RaceHorses_832x480_30,27,1186.402466,39.042999,40.184299,41.339401,814.166016,0.555
RaceHorses_832x480_30,32,691.087524,35.417801,38.033699,39.479301,624.107971,0.473
RaceHorses_832x480_30,37,357.915009,31.750999,36.718899,37.998699,433.789001,0.261
BasketballPass_416x240_50,22,581.125,43.161598,45.375198,45.743401,241.403,0.159
BasketballPass_416x240_50,27,345.600006,39.6591,42.5453,42.643398,194.281998,0.128
BasketballPass_416x240_50,32,194.449997,36.297199,40.304199,40.3134,136.813995,0.106
BasketballPass_416x240_50,37,105.449997,33.166,38.504902,38.564301,88.305,0.065
BQSquare_416x240_60,22,1507.155029,41.956001,43.8578,44.793999,241.932999,0.22
BQSquare_416x240_60,27,999.780029,37.541302,41.067501,42.172199,182.520004,0.172
BQSquare_416x240_60,32,637.275024,33.667198,39.649799,40.419601,159.934998,0.139
BQSquare_416x240_60,37,391.334991,29.9818,38.667301,39.090099,136.598007,0.115
BlowingBubbles_416x240_50,22,1101.099976,41.496498,42.352299,44.056198,335.32901,0.128
BlowingBubbles_416x240_50,27,649.287476,37.4678,39.4524,41.145699,282.694,0.17
BlowingBubbles_416x240_50,32,357.362488,33.8358,37.3134,38.8895,210.638,0.134
BlowingBubbles_416x240_50,37,183.862503,30.552799,35.817501,37.3913,150.509995,0.108
RaceHorses_416x240_30,22,580.027527,42.9077,43.218899,43.736698,294.014008,0.203
RaceHorses_416x240_30,27,363.532501,38.7756,39.990299,40.7108,221.876007,0.168
RaceHorses_416x240_30,32,211.222504,34.7826,37.861698,38.352901,186.367996,0.14
RaceHorses_416x240_30,37,111.525002,31.1199,36.2397,36.451698,128.087006,0.117
FourPeople_1280x720_60,22,3301.875,44.371899,47.365002,48.584801,1482.285034,0.818
FourPeople_1280x720_60,27,2032.694946,41.901299,45.2533,46.521099,982.65802,0.844
FourPeople_1280x720_60,32,1260.98999,39.091801,43.509201,44.770599,696.674011,0.621
FourPeople_1280x720_60,37,760.26001,35.910599,42.266102,43.378899,444.725006,0.65
Johnny_1280x720_60,22,2167.935059,44.488899,48.638199,49.308399,1085.142944,0.777
Johnny_1280x720_60,27,1191.344971,42.405899,46.709202,47.511501,624.531982,0.669
Johnny_1280x720_60,32,682.919983,40.1875,45.052601,45.744499,438.42099,0.598
Johnny_1280x720_60,37,395.984985,37.662899,43.644901,44.315201,249.160004,0.53
KristenAndSara_1280x720_60,22,2406.300049,44.928699,48.016998,48.933399,1179.161011,0.845
KristenAndSara_1280x720_60,27,1443.944946,42.7122,45.955101,47.0975,681.310974,0.645
KristenAndSara_1280x720_60,32,875.25,40.168701,44.2216,45.4137,481.562012,0.563
KristenAndSara_1280x720_60,37,528.299988,37.345901,42.8521,44.060001,273.746002,0.462
BasketballDrillText_832x480_50,22,2188.787598,42.621498,44.1717,45.022701,860.627014,0.707
BasketballDrillText_832x480_50,27,1195.574951,39.218201,41.530102,42.178299,636.255005,0.514
BasketballDrillText_832x480_50,32,659.262512,36.213699,39.614899,40.1796,434.855011,0.407
BasketballDrillText_832x480_50,37,379.225006,33.486698,38.069302,38.381401,258.056,0.348
ChinaSpeed_1024x768_30,22,2423.827393,45.6633,47.315899,47.317799,969.096985,0.655
ChinaSpeed_1024x768_30,27,1633.162476,41.697201,44.298599,43.980701,857.643005,0.949
ChinaSpeed_1024x768_30,32,1080.089966,37.829498,42.137901,41.218601,718.317993,0.82
ChinaSpeed_1024x768_30,37,694.034973,34.100201,40.399101,39.052399,579.901978,0.697
SlideEditing_1280x720_30,22,3761.399902,46.9702,45.1208,45.693901,900.520996,0.823
SlideEditing_1280x720_30,27,2829.13501,42.449402,41.024101,42.140701,840.200989,0.712
SlideEditing_1280x720_30,32,2195.16748,37.8624,39.074699,40.416302,692.809998,0.634
SlideEditing_1280x720_30,37,1641.097534,33.052399,38.199001,39.515701,648.926025,0.581
SlideShow_1280x720_20,22,588.97998,48.2146,51.686798,53.196899,527.942017,0.441
SlideShow_1280x720_20,27,358.225006,45.059101,49.419998,50.7868,350.460999,0.401
SlideShow_1280x720_20,32,239.145004,42.242901,47.637798,49.0257,258.901001,0.579
SlideShow_1280x720_20,37,161.639999,39.275398,45.6077,47.177299,183.113007,0.521
"""
        self.init_text = StringIO(self.init_text)
        self.df = pd.read_csv(self.init_text, sep=",")
        self.df.isanchor = True
        self.df.isCTC = True
        self.named = namedtuple('named', list(self.df.columns))

        print(' ')

    def makeNamedTuple(self, seq, qp, bitrate, ypsnr, upsnr, vpsnr, encT, decT, isAnchor=False, isCTC=False):
        return self.named(seq, qp, bitrate, ypsnr, upsnr, vpsnr, encT, decT, isAnchor, isCTC)

    def isCTC(self, seq_name):
        try:
            return self.df[self.df.seq==seq_name].iloc[0].isCTC
        except:
            return False

    def getEncResultFromFile(self, path):
        path = os.path.basename(path)
        ras_str = str(os.path.splitext(path)[0].split('_')[-1]).lower()
        isras = False
        rasid = 0
        if 'rs' in ras_str:
            isras = True
            rasid = int(ras_str.split('rs')[-1])
        elif 'ras' in ras_str:
            isras = True
            rasid = int(ras_str.split('ras')[-1])
        try:
            framerate = int(path.split('_')[2])
        except:
            framerate = int(path.split('_')[2].split('f')[0])
        with open(path, 'r') as f:
            data = f.read()
            totaldata = parse('{}Total Frames{}\n{}\n{}', data)
            totaldata = totaldata[2].split()
            codedframe = int(totaldata[0])
            kbps = float(totaldata[2])
            Ypsnr = float(totaldata[3])
            Upsnr = float(totaldata[4])
            Vpsnr = float(totaldata[5])
            # print(totalframe, kbps, Ypnsr, Upsnr, Vpsnr)
            encT = parse('{} Total Time:     {} sec{}', data)
            encT = float(encT[1])
            # print(totalenctime)
            ikbps = iYpsnr = iUpsnr = iVpsnr = iencT = 0
            if isras and rasid != 0:
                poc1data = parse(
                    '{}POC    0 TId: {} ( I-SLICE, QP {} )     {} bits [Y {} dB    U {} dB    V {} dB]{}[ET   {} ] {}',
                    data)
                ikbps = float(poc1data[3])
                iYpsnr = float(poc1data[4])
                iUpsnr = float(poc1data[5])
                iVpsnr = float(poc1data[6])
                iencT = float(poc1data[8])
                ikbps = (ikbps * self.seq_info.frameRate) / 1000.0
                # print(ikbps, iYpsnr, iUpsnr, iVpsnr, iencT)
            if isras > 1:
                kbps = kbps * codedframe - ikbps
                Ypsnr = Ypsnr * codedframe - iYpsnr
                Upsnr = Upsnr * codedframe - iUpsnr
                Vpsnr = Vpsnr * codedframe - iVpsnr
                encT = encT - iencT


    def setEncDecData(self, encpath, decpath):
        with open(encpath, 'r') as f:
            data = f.read()
            totaldata = parse('{}Total Frames{}\n{}\n{}', data)
            totaldata = totaldata[2].split()
            codedframe = int(totaldata[0])
            kbps = float(totaldata[2])
            Ypsnr = float(totaldata[3])
            Upsnr = float(totaldata[4])
            Vpsnr = float(totaldata[5])
            # print(totalframe, kbps, Ypnsr, Upsnr, Vpsnr)
            encT = parse('{} Total Time:     {} sec{}', data)
            encT = float(encT[1])
            # print(totalenctime)
            ikbps = iYpsnr = iUpsnr = iVpsnr = iencT = 0
            if self.rasnum>1 and self.rasid !=0:
                poc1data = parse(
                    '{}POC    0 TId: {} ( I-SLICE, QP {} )     {} bits [Y {} dB    U {} dB    V {} dB]{}[ET   {} ] {}',
                    data)
                ikbps = float(poc1data[3])
                iYpsnr = float(poc1data[4])
                iUpsnr = float(poc1data[5])
                iVpsnr = float(poc1data[6])
                iencT = float(poc1data[8])
                ikbps = (ikbps*self.seq_info.frameRate)/1000.0
                # print(ikbps, iYpsnr, iUpsnr, iVpsnr, iencT)
            if self.rasnum>1:
                kbps = kbps*codedframe - ikbps
                Ypsnr = Ypsnr*codedframe - iYpsnr
                Upsnr = Upsnr*codedframe - iUpsnr
                Vpsnr = Vpsnr*codedframe - iVpsnr
                encT = encT - iencT

        with open(decpath, 'r') as f:
            data = f.read()
            decT = parse('{} Total Time:{}sec{}', data)
            decT = float(decT[1].split()[0])
            # print(decT[1])
            idecT = 0
            if self.rasnum > 1 and self.rasid != 0:
                idecT = parse('{}POC    0 TId{}[DT{}]{}', data)
                idecT = float(idecT[2].split()[0])
                decT = decT - idecT
            # print(idecT[2])
            iserror = parse('{}ERROR{}', data)
            if iserror != None:
                self.ismd5error = True
            if self.rasid>0:
                decT = decT - idecT
        return kbps, Ypsnr, Upsnr, Vpsnr, encT, decT, codedframe
            # print(iserror)

    @staticmethod
    def getBDrate(rateA, distA, rateB, distB):
        minPSNR = max(min(distA), min(distB))
        maxPSNR = min(max(distA), max(distB))
        def getBDint(rate, dist):
            def pchipend(h1, h2, del1, del2):
                _d = ((2 * h1 + h2) * del1 - h1 * del2) / (h1 + h2)
                if (_d*del1)<0:
                    _d = 0
                elif ((del1 * del2 < 0) and (abs(_d) > abs(3 * del1))):
                    _d = 3 * del1
                return _d


            log_rate = []
            log_dist = []
            for i in range(4):
                log_rate.append(math.log10(rate[3-i]))
                log_dist.append(dist[3-i])
            H = []
            delta = []
            for i in range(3):
                H.append(log_dist[i+1] - log_dist[i])
                delta.append((log_rate[i+1] - log_rate[i])/H[i])
            d = []
            d.append(pchipend(H[0], H[1], delta[0], delta[1]))
            for i in range(1,3):
                d.append((3 * H[i - 1] + 3 * H[i]) / ((2 * H[i] + H[i - 1]) / delta[i - 1] + (H[i] + 2 * H[i - 1]) / delta[i]))
            d.append(pchipend(H[2], H[1], delta[2], delta[1]))
            c = []
            b = []
            for i in range(3):
                c.append((3 * delta[i] - 2 * d[i] - d[i + 1]) / H[i])
                b.append((d[i] - 2 * delta[i] + d[i + 1]) / (H[i] * H[i]))
            result = 0
            for i in range(3):
                s0 = log_dist[i]
                s1 = log_dist[i+1]

                s0 = max(s0, minPSNR)
                s0 = min(s0, maxPSNR)

                s1 = max(s1, minPSNR)
                s1 = min(s1, maxPSNR)

                s0 -= log_dist[i]
                s1 -= log_dist[i]

                if s1>s0:
                    result += (s1 - s0) * log_rate[i]
                    result += (s1 * s1 - s0 * s0) * d[i] / 2
                    result += (s1 * s1 * s1 - s0 * s0 * s0) * c[i] / 3
                    result += (s1 * s1 * s1 * s1 - s0 * s0 * s0 * s0) * b[i] / 4
            return result
        vA = getBDint(rateA, distA)
        vB = getBDint(rateB, distB)
        avg = (vB - vA) / (maxPSNR - minPSNR)
        return math.pow(10, avg) - 1


if __name__=='__main__':
    r = Reporting()