% --- Linear
A_lin = []
B_lin = []
% --- Linear with error
A_line = [
    2.6505 8.9590;
    3.4403 5.3366;
    3.4010 6.1624;
    5.2153 8.2529;
    7.6393 9.4764;
    1.5041 3.3370;
    3.9855 8.3138;
    1.8500 5.0079;
    1.2631 8.6463;
    3.8957 4.9014;
    1.9751 8.6199;
    1.2565 6.4558;
    4.3732 6.1261;
    0.4297 8.3551;
    3.6931 6.6134;
    7.8164 9.8767;
    4.8561 8.7376;
    6.7750 7.9386;
    2.3734 4.7740;
    0.8746 3.0892;
    2.3088 9.0919;
    2.5520 9.0469;
    3.3773 6.1886;
    0.8690 3.7550;
    1.8738 8.1053;
    0.9469 5.1476;
    0.4309 7.5763;
    2.2699 7.1371]
B_line = [
    7.7030 5.0965;
    5.7670 2.8791;
    3.6610 1.5002;
    9.4633 6.5084;
    9.8221 1.9383;
    8.2874 4.9380;
    5.9078 0.4489;
    4.9810 0.5962;
    5.1516 0.5319;
    8.4363 5.9467;
    8.4240 4.9696;
    7.6240 1.7988;
    3.4473 0.2725;
    9.0528 4.7106;
    9.1046 3.2798;
    6.9110 0.1745;
    5.1235 3.3181;
    7.5051 3.3392;
    6.3283 4.1555;
    6.1585 1.5058;
    8.3827 7.2617;
    5.2841 2.7510;
    5.1412 1.9314;
    5.8863 1.0087;
    9.5110 1.3298;
    6.5170 1.4606;
    9.8621 4.3674;
    6.0000 8.0000]

% --- Non linear
A_nl = [
    0.0113 0.2713;
    0.9018 -0.1121;
    0.2624 -0.2899;
    0.3049 0.2100;
    -0.2255 -0.7156;
    -0.9497 -0.1578;
    -0.6318 0.4516;
    -0.2593 0.6831;
    0.4685 0.1421;
    -0.4694 0.8492;
    -0.5525 -0.2529;
    -0.8250 0.2802;
    0.4463 -0.3051;
    0.3212 -0.2323;
    0.2547 -0.9567;
    0.4917 0.6262;
    -0.2334 0.2346;
    0.1510 0.0601;
    -0.4499 -0.5027;
    -0.0967 -0.5446]
B_nl = [
    1.2178 1.9444;
    -1.8800 0.1427;
    -1.6517 1.2084;
    1.9566 -1.7322;
    1.7576 -1.9273;
    0.7354 1.1349;
    0.1366 1.5414;
    1.5960 0.5038;
    -1.4485 -1.1288;
    -1.2714 -1.8327;
    -1.5722 0.4658;
    1.7586 -0.5822;
    -0.3575 1.9374;
    1.7823 0.7066;
    1.9532 1.0673;
    -1.0233 -0.8180;
    1.0021 0.3341;
    0.0473 -1.6696;
    0.8783 1.9846;
    -0.5819 1.8850]

% --- Regression
x_reg = [
    -5.0000;
    -4.8000;
    -4.6000;
    -4.4000;
    -4.2000;
    -4.0000;
    -3.8000;
    -3.6000;
    -3.4000;
    -3.2000;
    -3.0000;
    -2.8000;
    -2.6000;
    -2.4000;
    -2.2000;
    -2.0000;
    -1.8000;
    -1.6000;
    -1.4000;
    -1.2000;
    -1.0000;
    -0.8000;
    -0.6000;
    -0.4000;
    -0.2000;
    0;
    0.2000;
    0.4000;
    0.6000;
    0.8000;
    1.0000;
    1.2000;
    1.4000;
    1.6000;
    1.8000;
    2.0000;
    2.2000;
    2.4000;
    2.6000;
    2.8000;
    3.0000;
    3.2000;
    3.4000;
    3.6000;
    3.8000;
    4.0000;
    4.2000;
    4.4000;
    4.6000;
    4.8000;
    5.0000]
y_reg = [
    -96.2607;
    -85.9893;
    -55.2451;
    -55.6153;
    -44.8827;
    -24.1306;
    -19.4970;
    -10.3972;
    -2.2633;
    0.2196;
    4.5852;
    7.1974;
    8.2207;
    16.0614;
    16.4224;
    17.5381;
    11.4895;
    14.1065;
    16.8220;
    16.1584;
    11.6846;
    5.9991;
    7.8277;
    2.8236;
    2.7129;
    1.1669;
    -1.4223;
    -3.8489;
    -4.7101;
    -8.1538;
    -7.3364;
    -13.6464;
    -15.2607;
    -14.8747;
    -9.9271;
    -10.5022;
    -7.7297;
    -11.7859;
    -10.2662;
    -7.1364;
    -2.1166;
    1.9428;
    4.0905;
    16.3151;
    16.9854;
    17.6418;
    46.3117;
    53.2609;
    72.3538;
    49.9166;
    89.1652]

% --- Linear esv
x_lin = [
    0;
    0.5000;
    1.0000;
    1.5000;
    2.0000;
    2.5000;
    3.0000;
    3.5000;
    4.0000;
    4.5000;
    5.0000;
    5.5000;
    6.0000;
    6.5000;
    7.0000;
    7.5000;
    8.0000;
    8.5000;
    9.0000;
    9.5000;
    10.0000]
y_lin = [
    2.5584;
    2.6882;
    2.9627;
    3.2608;
    3.6235;
    3.9376;
    4.0383;
    4.1570;
    4.8498;
    4.6561;
    4.5119;
    4.8346;
    5.6039;
    5.5890;
    6.1914;
    5.8966;
    6.3866;
    6.6909;
    6.5224;
    7.1803;
    7.2537]