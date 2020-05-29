import numpy as np
import pandas as pd
from io import StringIO


init_text = """seq,qp,bitrate,ypsnr,upsnr,vpsnr,encT,decT 
Tango2_3840x2160_60fps_10bit_420,22,18105.17969,0,0,0,14418.91699,7.604
Tango2_3840x2160_60fps_10bit_420,27,5442.075195,0,0,0,5155.830078,4.324
Tango2_3840x2160_60fps_10bit_420,32,3022.155029,0,0,0,1644.702026,5.714
Tango2_3840x2160_60fps_10bit_420,37,1797.689941,0,0,0,539.791016,4.737
FoodMarket4_3840x2160_60fps_10bit_420,22,5110.814941,0,0,0,4072.976074,5.792
FoodMarket4_3840x2160_60fps_10bit_420,27,2816.01001,0,0,0,844.830994,5.355
FoodMarket4_3840x2160_60fps_10bit_420,32,1810.560059,0,0,0,426.346008,4.816
FoodMarket4_3840x2160_60fps_10bit_420,37,1198.275024,0,0,0,160.623993,4.013
Campfire_3840x2160_30fps_10bit_420_bt709_videoRange,22,20806.69531,0,0,0,13760.45898,9.356
Campfire_3840x2160_30fps_10bit_420_bt709_videoRange,27,9233.415039,0,0,0,9047.286133,7.001
Campfire_3840x2160_30fps_10bit_420_bt709_videoRange,32,3845.294922,0,0,0,5249.178223,4.149
Campfire_3840x2160_30fps_10bit_420_bt709_videoRange,37,2100.060059,0,0,0,2368.667969,4.862
CatRobot1_3840x2160p_60_10_709_420,22,33470.57813,0,0,0,15727.54981,8.426
CatRobot1_3840x2160p_60_10_709_420,27,11436.64453,0,0,0,8494.822266,6.808
CatRobot1_3840x2160p_60_10_709_420,32,6597.089844,0,0,0,3851.084961,5.917
CatRobot1_3840x2160p_60_10_709_420,37,3937.050049,0,0,0,1869.125,5.16
DaylightRoad2_3840x2160_60fps_10bit_420,22,57475.98047,0,0,0,20912.69531,6.458
DaylightRoad2_3840x2160_60fps_10bit_420,27,11722.16992,0,0,0,11689.13867,6.781
DaylightRoad2_3840x2160_60fps_10bit_420,32,6266.399902,0,0,0,4822.611816,5.277
DaylightRoad2_3840x2160_60fps_10bit_420,37,3483.975098,0,0,0,2038.996948,4.965
ParkRunning3_3840x2160_50fps_10bit_420,22,49081.46094,0,0,0,13726.3291,6.331
ParkRunning3_3840x2160_50fps_10bit_420,27,27797.32422,0,0,0,9497.167969,8.944
ParkRunning3_3840x2160_50fps_10bit_420,32,15390.5,0,0,0,6552.12793,8.001
ParkRunning3_3840x2160_50fps_10bit_420,37,8381.599609,0,0,0,3936.697021,6.285
MarketPlace_1920x1080_60fps_10bit_420,22,8829.089844,0,0,0,4202.289063,1.723
MarketPlace_1920x1080_60fps_10bit_420,27,4259.205078,0,0,0,2153.551025,1.929
MarketPlace_1920x1080_60fps_10bit_420,32,2242.665039,0,0,0,1230.046997,1.295
MarketPlace_1920x1080_60fps_10bit_420,37,1153.094971,0,0,0,523.684998,1.465
RitualDance_1920x1080_60fps_10bit_420,22,3345.554932,0,0,0,1437.420044,1.762
RitualDance_1920x1080_60fps_10bit_420,27,2018.984985,0,0,0,859.442017,1.5
RitualDance_1920x1080_60fps_10bit_420,32,1208.910034,0,0,0,604.539978,1.332
RitualDance_1920x1080_60fps_10bit_420,37,710.820007,0,0,0,342.539001,1.173
Cactus_1920x1080_50,22,13100.15039,0,0,0,5753.750977,2.995
Cactus_1920x1080_50,27,5735.737305,0,0,0,3526.952881,2.201
Cactus_1920x1080_50,32,3113.162598,0,0,0,2176.055908,1.881
Cactus_1920x1080_50,37,1689.3125,0,0,0,1194.421021,1.397
BasketballDrive_1920x1080_50,22,5080.9375,0,0,0,4244.852051,2.085
BasketballDrive_1920x1080_50,27,2136.425049,0,0,0,2053.218994,1.45
BasketballDrive_1920x1080_50,32,1150.150024,0,0,0,933.369019,1.547
BasketballDrive_1920x1080_50,37,660.662476,0,0,0,415.471008,1.19
BQTerrace_1920x1080_60,22,20900.95508,0,0,0,3660.245117,2.63
BQTerrace_1920x1080_60,27,10832.80469,0,0,0,3624.51709,1.933
BQTerrace_1920x1080_60,32,5872.22998,0,0,0,2535.185059,1.259
BQTerrace_1920x1080_60,37,3175.39502,0,0,0,1774.062988,1.4
BasketballDrill_832x480_50,22,2183.649902,0,0,0,779.35199,0.681
BasketballDrill_832x480_50,27,1157.050049,0,0,0,634.468018,0.514
BasketballDrill_832x480_50,32,618.099976,0,0,0,449.713989,0.401
BasketballDrill_832x480_50,37,344.4375,0,0,0,225.434006,0.341
BQMall_832x480_60,22,3241.725098,0,0,0,963.713013,0.672
BQMall_832x480_60,27,2006.609985,0,0,0,700.362976,0.534
BQMall_832x480_60,32,1205.655029,0,0,0,487.584991,0.444
BQMall_832x480_60,37,672.164978,0,0,0,397.42099,0.363
PartyScene_832x480_50,22,5873,0,0,0,1122.686035,1.001
PartyScene_832x480_50,27,3743.162598,0,0,0,863.872986,0.792
PartyScene_832x480_50,32,2270.324951,0,0,0,835.94397,0.636
PartyScene_832x480_50,37,1243.875,0,0,0,652.185974,0.503
RaceHorses_832x480_30,22,1945.522461,0,0,0,1052.364014,0.663
RaceHorses_832x480_30,27,1186.402466,0,0,0,762.228027,0.398
RaceHorses_832x480_30,32,691.087524,0,0,0,578.93103,0.476
RaceHorses_832x480_30,37,357.915009,0,0,0,415.737,0.393
BasketballPass_416x240_50,22,581.125,0,0,0,244.703003,0.156
BasketballPass_416x240_50,27,345.600006,0,0,0,199.050003,0.089
BasketballPass_416x240_50,32,194.449997,0,0,0,106.046997,0.107
BasketballPass_416x240_50,37,105.449997,0,0,0,69.310997,0.087
BQSquare_416x240_60,22,1507.155029,0,0,0,223.007996,0.222
BQSquare_416x240_60,27,999.780029,0,0,0,199.011993,0.171
BQSquare_416x240_60,32,637.275024,0,0,0,152.501007,0.14
BQSquare_416x240_60,37,391.334991,0,0,0,102.223,0.115
BlowingBubbles_416x240_50,22,1101.099976,0,0,0,319.256989,0.218
BlowingBubbles_416x240_50,27,649.287476,0,0,0,247.557999,0.11
BlowingBubbles_416x240_50,32,357.362488,0,0,0,173.069,0.135
BlowingBubbles_416x240_50,37,183.862503,0,0,0,129.998993,0.109
RaceHorses_416x240_30,22,580.027527,0,0,0,281.036011,0.2
RaceHorses_416x240_30,27,363.532501,0,0,0,143.725006,0.167
RaceHorses_416x240_30,32,211.222504,0,0,0,146.768005,0.139
RaceHorses_416x240_30,37,111.525002,0,0,0,138.600998,0.116
FourPeople_1280x720_60,22,3301.875,0,0,0,1283.062988,0.593
FourPeople_1280x720_60,27,2032.694946,0,0,0,851.658997,0.846
FourPeople_1280x720_60,32,1260.98999,0,0,0,639.047974,0.741
FourPeople_1280x720_60,37,760.26001,0,0,0,385.813995,0.647
Johnny_1280x720_60,22,2167.935059,0,0,0,990.935974,0.776
Johnny_1280x720_60,27,1191.344971,0,0,0,661.054993,0.664
Johnny_1280x720_60,32,682.919983,0,0,0,400.832001,0.599
Johnny_1280x720_60,37,395.984985,0,0,0,242.813995,0.526
KristenAndSara_1280x720_60,22,2406.300049,0,0,0,961.83197,0.855
KristenAndSara_1280x720_60,27,1443.944946,0,0,0,654.596008,0.743
KristenAndSara_1280x720_60,32,875.25,0,0,0,436.63501,0.656
KristenAndSara_1280x720_60,37,528.299988,0,0,0,252.787994,0.601
BasketballDrillText_832x480_50,22,2188.787598,0,0,0,709.143982,0.679
BasketballDrillText_832x480_50,27,1195.574951,0,0,0,565.890991,0.515
BasketballDrillText_832x480_50,32,659.262512,0,0,0,428.098999,0.41
BasketballDrillText_832x480_50,37,379.225006,0,0,0,267.911011,0.343
ChinaSpeed_1024x768_30,22,2423.827393,0,0,0,729.29303,0.647
ChinaSpeed_1024x768_30,27,1633.162476,0,0,0,597.820984,0.55
ChinaSpeed_1024x768_30,32,1080.089966,0,0,0,539.085999,0.477
ChinaSpeed_1024x768_30,37,694.034973,0,0,0,395.497009,0.425
SlideEditing_1280x720_30,22,3761.399902,0,0,0,742.507019,0.825
SlideEditing_1280x720_30,27,2829.13501,0,0,0,590.383972,0.712
SlideEditing_1280x720_30,32,2195.16748,0,0,0,518.760986,0.628
SlideEditing_1280x720_30,37,1641.097534,0,0,0,464.996002,0.563
SlideShow_1280x720_20,22,588.97998,0,0,0,413.683014,0.438
SlideShow_1280x720_20,27,358.225006,0,0,0,239.686996,0.413
SlideShow_1280x720_20,32,239.145004,0,0,0,152.507004,0.356
SlideShow_1280x720_20,37,161.639999,0,0,0,108.771004,0.318
"""

init_text = StringIO(init_text)


if __name__=='__main__':
    df = pd.read_csv(init_text, sep=",")
    print(' ')