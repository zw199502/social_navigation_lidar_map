/*
 * Automatically Generated from Mathematica.
 * Thu 10 Nov 2022 14:17:08 GMT-05:00
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "T_LeftToeBottom_src.h"

#ifdef _MSC_VER
  #define INLINE __forceinline /* use __forceinline (VC++ specific) */
#else
  #define INLINE inline        /* use standard inline */
#endif

/**
 * Copied from Wolfram Mathematica C Definitions file mdefs.hpp
 * Changed marcos to inline functions (Eric Cousineau)
 */
INLINE double Power(double x, double y) { return pow(x, y); }
INLINE double Sqrt(double x) { return sqrt(x); }

INLINE double Abs(double x) { return fabs(x); }

INLINE double Exp(double x) { return exp(x); }
INLINE double Log(double x) { return log(x); }

INLINE double Sin(double x) { return sin(x); }
INLINE double Cos(double x) { return cos(x); }
INLINE double Tan(double x) { return tan(x); }

INLINE double Csc(double x) { return 1.0/sin(x); }
INLINE double Sec(double x) { return 1.0/cos(x); }

INLINE double ArcSin(double x) { return asin(x); }
INLINE double ArcCos(double x) { return acos(x); }
//INLINE double ArcTan(double x) { return atan(x); }

/* update ArcTan function to use atan2 instead. */
INLINE double ArcTan(double x, double y) { return atan2(y,x); }

INLINE double Sinh(double x) { return sinh(x); }
INLINE double Cosh(double x) { return cosh(x); }
INLINE double Tanh(double x) { return tanh(x); }

#define E 2.71828182845904523536029
#define Pi 3.14159265358979323846264
#define Degree 0.01745329251994329576924

/*
 * Sub functions
 */
static void output1(double *p_output1,const double *var1)
{
  double t259;
  double t243;
  double t268;
  double t244;
  double t269;
  double t219;
  double t220;
  double t231;
  double t238;
  double t285;
  double t297;
  double t298;
  double t246;
  double t271;
  double t273;
  double t299;
  double t200;
  double t206;
  double t207;
  double t282;
  double t300;
  double t324;
  double t342;
  double t354;
  double t360;
  double t381;
  double t393;
  double t470;
  double t471;
  double t473;
  double t474;
  double t475;
  double t477;
  double t480;
  double t488;
  double t510;
  double t511;
  double t325;
  double t333;
  double t339;
  double t376;
  double t398;
  double t399;
  double t419;
  double t424;
  double t519;
  double t433;
  double t438;
  double t439;
  double t444;
  double t451;
  double t459;
  double t460;
  double t461;
  double t171;
  double t173;
  double t189;
  double t146;
  double t425;
  double t428;
  double t431;
  double t468;
  double t524;
  double t533;
  double t534;
  double t536;
  double t548;
  double t585;
  double t588;
  double t589;
  double t591;
  double t592;
  double t593;
  double t604;
  double t606;
  double t617;
  double t650;
  double t669;
  double t683;
  double t685;
  double t697;
  double t751;
  double t756;
  double t763;
  double t764;
  double t855;
  double t878;
  double t884;
  double t547;
  double t578;
  double t581;
  double t652;
  double t657;
  double t658;
  double t776;
  double t782;
  double t801;
  double t810;
  double t813;
  double t823;
  double t824;
  double t835;
  double t843;
  double t844;
  double t914;
  double t922;
  double t927;
  double t930;
  double t961;
  double t965;
  double t967;
  double t972;
  double t988;
  double t991;
  double t1012;
  double t1013;
  double t1025;
  double t29;
  double t33;
  double t34;
  double t36;
  double t54;
  double t154;
  double t167;
  double t784;
  double t787;
  double t794;
  double t854;
  double t885;
  double t886;
  double t994;
  double t999;
  double t1034;
  double t1035;
  double t1037;
  double t1047;
  double t1051;
  double t1055;
  double t1089;
  double t1090;
  double t1122;
  double t1135;
  double t1137;
  double t1140;
  double t1141;
  double t1142;
  double t1148;
  double t1187;
  double t65;
  double t71;
  double t1008;
  double t1111;
  double t1118;
  double t1120;
  double t1193;
  double t1194;
  double t1214;
  double t1234;
  double t1236;
  double t1254;
  double t1259;
  double t1261;
  double t1270;
  double t1284;
  double t35;
  double t1307;
  double t1313;
  double t1314;
  double t1322;
  double t1324;
  double t1328;
  double t1329;
  double t1340;
  double t1359;
  double t1367;
  double t1397;
  double t1199;
  double t42;
  double t43;
  double t1603;
  double t1616;
  double t1617;
  double t1552;
  double t1558;
  double t1562;
  double t1585;
  double t1638;
  double t1659;
  double t1673;
  double t1678;
  double t1691;
  double t1798;
  double t1802;
  double t1803;
  double t1807;
  double t1671;
  double t1693;
  double t1699;
  double t1737;
  double t1753;
  double t1754;
  double t1756;
  double t1757;
  double t1747;
  double t1764;
  double t1817;
  double t1828;
  double t1846;
  double t1848;
  double t1868;
  double t1875;
  double t1888;
  double t1898;
  double t1927;
  double t1933;
  double t1839;
  double t1882;
  double t1982;
  double t1994;
  double t2002;
  double t2007;
  double t2021;
  double t2029;
  double t2035;
  double t2036;
  double t2052;
  double t2057;
  double t1200;
  double t1212;
  double t1998;
  double t2031;
  double t2059;
  double t2074;
  double t2091;
  double t2093;
  double t2109;
  double t2112;
  double t2136;
  double t2141;
  double t2147;
  double t2149;
  double t1297;
  double t1300;
  double t1391;
  double t1392;
  double t2087;
  double t2125;
  double t2151;
  double t2152;
  double t1399;
  double t1400;
  double t2155;
  double t2156;
  double t2158;
  double t2164;
  double t1405;
  double t1415;
  double t2177;
  double t2181;
  double t2195;
  double t2197;
  double t1457;
  double t1469;
  double t1474;
  double t1482;
  double t1492;
  double t1498;
  double t2386;
  double t2392;
  double t2407;
  double t2417;
  double t2425;
  double t2426;
  double t2567;
  double t2573;
  double t2580;
  double t2583;
  double t2416;
  double t2439;
  double t2467;
  double t2495;
  double t2527;
  double t2536;
  double t2539;
  double t2558;
  double t2508;
  double t2560;
  double t2591;
  double t2593;
  double t2606;
  double t2614;
  double t2661;
  double t2662;
  double t2673;
  double t2676;
  double t2678;
  double t2680;
  double t2604;
  double t2664;
  double t2681;
  double t2683;
  double t2746;
  double t2748;
  double t2758;
  double t2763;
  double t2776;
  double t2783;
  double t2793;
  double t2803;
  double t2714;
  double t2769;
  double t2805;
  double t2812;
  double t2827;
  double t2833;
  double t2857;
  double t2867;
  double t2903;
  double t2916;
  double t2918;
  double t2920;
  double t2816;
  double t2898;
  double t2927;
  double t2937;
  double t3006;
  double t3010;
  double t3014;
  double t3019;
  double t3027;
  double t3028;
  double t3035;
  double t3037;
  double t1395;
  double t1401;
  double t1420;
  double t1448;
  double t1473;
  double t1490;
  double t1503;
  double t1524;
  double t2277;
  double t2284;
  double t2290;
  double t2300;
  double t2315;
  double t2319;
  double t2329;
  double t2341;
  double t3045;
  double t3055;
  double t3067;
  double t3069;
  double t3079;
  double t3080;
  double t3106;
  double t3108;
  double t1197;
  double t1294;
  double t1379;
  double t1385;
  double t2154;
  double t2172;
  double t2208;
  double t2212;
  double t2940;
  double t3020;
  double t3038;
  double t3040;
  double t3484;
  double t3533;
  double t3470;
  double t3303;
  double t3308;
  double t3312;
  double t3313;
  double t3318;
  double t3323;
  double t3351;
  double t3353;
  double t3356;
  double t3386;
  double t3395;
  double t3402;
  double t3406;
  double t3408;
  double t3412;
  double t3416;
  double t3422;
  double t3424;
  double t3462;
  double t3469;
  double t3472;
  double t3475;
  double t3486;
  double t3491;
  double t3509;
  double t3516;
  double t3518;
  double t3522;
  double t3523;
  double t3539;
  double t3545;
  double t3546;
  double t3560;
  double t3562;
  double t3572;
  double t3595;
  double t3604;
  double t3636;
  double t3639;
  double t3656;
  double t3663;
  double t3665;
  double t3666;
  double t3673;
  double t3675;
  double t3687;
  double t3691;
  double t3700;
  double t3707;
  double t3716;
  double t3717;
  double t3721;
  double t3729;
  double t3730;
  double t3735;
  double t3743;
  double t3748;
  double t3750;
  double t3751;
  double t3753;
  double t3765;
  double t3767;
  double t3768;
  double t3774;
  double t3791;
  double t3792;
  double t3798;
  double t3806;
  double t3812;
  double t3817;
  double t3819;
  double t3828;
  double t3832;
  double t3833;
  double t3834;
  double t3849;
  double t3872;
  double t3874;
  double t3879;
  double t3897;
  double t3898;
  double t3900;
  double t3902;
  double t3912;
  double t3913;
  double t3916;
  double t3919;
  t259 = Cos(var1[3]);
  t243 = Cos(var1[5]);
  t268 = Sin(var1[4]);
  t244 = Sin(var1[3]);
  t269 = Sin(var1[5]);
  t219 = Cos(var1[7]);
  t220 = -1.*t219;
  t231 = 1. + t220;
  t238 = Cos(var1[6]);
  t285 = t259*t243*t268;
  t297 = t244*t269;
  t298 = t285 + t297;
  t246 = -1.*t243*t244;
  t271 = t259*t268*t269;
  t273 = t246 + t271;
  t299 = Sin(var1[6]);
  t200 = Cos(var1[8]);
  t206 = -1.*t200;
  t207 = 1. + t206;
  t282 = t238*t273;
  t300 = -1.*t298*t299;
  t324 = t282 + t300;
  t342 = t238*t298;
  t354 = t273*t299;
  t360 = t342 + t354;
  t381 = Cos(var1[4]);
  t393 = Sin(var1[7]);
  t470 = -1.000000637725*t231;
  t471 = 1. + t470;
  t473 = t259*t381*t471;
  t474 = -0.930418*t393;
  t475 = 0. + t474;
  t477 = t324*t475;
  t480 = -0.366501*t393;
  t488 = 0. + t480;
  t510 = t360*t488;
  t511 = t473 + t477 + t510;
  t325 = -0.340999127418*t231*t324;
  t333 = -0.134322983001*t231;
  t339 = 1. + t333;
  t376 = t339*t360;
  t398 = 0.366501*t393;
  t399 = 0. + t398;
  t419 = t259*t381*t399;
  t424 = t325 + t376 + t419;
  t519 = Sin(var1[8]);
  t433 = -0.8656776547239999*t231;
  t438 = 1. + t433;
  t439 = t438*t324;
  t444 = -0.340999127418*t231*t360;
  t451 = 0.930418*t393;
  t459 = 0. + t451;
  t460 = t259*t381*t459;
  t461 = t439 + t444 + t460;
  t171 = Cos(var1[9]);
  t173 = -1.*t171;
  t189 = 1. + t173;
  t146 = Sin(var1[10]);
  t425 = 0.340999127418*t207*t424;
  t428 = -0.134322983001*t207;
  t431 = 1. + t428;
  t468 = t431*t461;
  t524 = -0.366501*t519;
  t533 = 0. + t524;
  t534 = t511*t533;
  t536 = t425 + t468 + t534;
  t548 = Sin(var1[9]);
  t585 = -1.000000637725*t207;
  t588 = 1. + t585;
  t589 = t588*t511;
  t591 = -0.930418*t519;
  t592 = 0. + t591;
  t593 = t424*t592;
  t604 = 0.366501*t519;
  t606 = 0. + t604;
  t617 = t461*t606;
  t650 = t589 + t593 + t617;
  t669 = -0.8656776547239999*t207;
  t683 = 1. + t669;
  t685 = t683*t424;
  t697 = 0.340999127418*t207*t461;
  t751 = 0.930418*t519;
  t756 = 0. + t751;
  t763 = t511*t756;
  t764 = t685 + t697 + t763;
  t855 = Cos(var1[10]);
  t878 = -1.*t855;
  t884 = 1. + t878;
  t547 = 0.340999127418*t189*t536;
  t578 = -0.930418*t548;
  t581 = 0. + t578;
  t652 = t581*t650;
  t657 = -0.8656776547239999*t189;
  t658 = 1. + t657;
  t776 = t658*t764;
  t782 = t547 + t652 + t776;
  t801 = -0.134322983001*t189;
  t810 = 1. + t801;
  t813 = t810*t536;
  t823 = 0.366501*t548;
  t824 = 0. + t823;
  t835 = t824*t650;
  t843 = 0.340999127418*t189*t764;
  t844 = t813 + t835 + t843;
  t914 = -0.366501*t548;
  t922 = 0. + t914;
  t927 = t922*t536;
  t930 = -1.000000637725*t189;
  t961 = 1. + t930;
  t965 = t961*t650;
  t967 = 0.930418*t548;
  t972 = 0. + t967;
  t988 = t972*t764;
  t991 = t927 + t965 + t988;
  t1012 = Cos(var1[11]);
  t1013 = -1.*t1012;
  t1025 = 1. + t1013;
  t29 = Cos(var1[12]);
  t33 = -1.*t29;
  t34 = 1. + t33;
  t36 = Sin(var1[12]);
  t54 = Sin(var1[11]);
  t154 = 0.930418*t146;
  t167 = 0. + t154;
  t784 = t167*t782;
  t787 = -0.366501*t146;
  t794 = 0. + t787;
  t854 = t794*t844;
  t885 = -1.000000637725*t884;
  t886 = 1. + t885;
  t994 = t886*t991;
  t999 = t784 + t854 + t994;
  t1034 = -0.8656776547239999*t884;
  t1035 = 1. + t1034;
  t1037 = t1035*t782;
  t1047 = 0.340999127418*t884*t844;
  t1051 = -0.930418*t146;
  t1055 = 0. + t1051;
  t1089 = t1055*t991;
  t1090 = t1037 + t1047 + t1089;
  t1122 = 0.340999127418*t884*t782;
  t1135 = -0.134322983001*t884;
  t1137 = 1. + t1135;
  t1140 = t1137*t844;
  t1141 = 0.366501*t146;
  t1142 = 0. + t1141;
  t1148 = t1142*t991;
  t1187 = t1122 + t1140 + t1148;
  t65 = 0.366501*t54;
  t71 = 0. + t65;
  t1008 = t71*t999;
  t1111 = 0.340999127418*t1025*t1090;
  t1118 = -0.134322983001*t1025;
  t1120 = 1. + t1118;
  t1193 = t1120*t1187;
  t1194 = t1008 + t1111 + t1193;
  t1214 = -0.930418*t54;
  t1234 = 0. + t1214;
  t1236 = t1234*t999;
  t1254 = -0.8656776547239999*t1025;
  t1259 = 1. + t1254;
  t1261 = t1259*t1090;
  t1270 = 0.340999127418*t1025*t1187;
  t1284 = t1236 + t1261 + t1270;
  t35 = -0.175248972904*t34;
  t1307 = -1.000000637725*t1025;
  t1313 = 1. + t1307;
  t1314 = t1313*t999;
  t1322 = 0.930418*t54;
  t1324 = 0. + t1322;
  t1328 = t1324*t1090;
  t1329 = -0.366501*t54;
  t1340 = 0. + t1329;
  t1359 = t1340*t1187;
  t1367 = t1314 + t1328 + t1359;
  t1397 = -0.120666640478*t34;
  t1199 = 0.444895486988*t34;
  t42 = 0.553471*t36;
  t43 = t35 + t42;
  t1603 = t243*t244*t268;
  t1616 = -1.*t259*t269;
  t1617 = t1603 + t1616;
  t1552 = t259*t243;
  t1558 = t244*t268*t269;
  t1562 = t1552 + t1558;
  t1585 = t238*t1562;
  t1638 = -1.*t1617*t299;
  t1659 = t1585 + t1638;
  t1673 = t238*t1617;
  t1678 = t1562*t299;
  t1691 = t1673 + t1678;
  t1798 = t381*t471*t244;
  t1802 = t1659*t475;
  t1803 = t1691*t488;
  t1807 = t1798 + t1802 + t1803;
  t1671 = -0.340999127418*t231*t1659;
  t1693 = t339*t1691;
  t1699 = t381*t244*t399;
  t1737 = t1671 + t1693 + t1699;
  t1753 = t438*t1659;
  t1754 = -0.340999127418*t231*t1691;
  t1756 = t381*t244*t459;
  t1757 = t1753 + t1754 + t1756;
  t1747 = 0.340999127418*t207*t1737;
  t1764 = t431*t1757;
  t1817 = t1807*t533;
  t1828 = t1747 + t1764 + t1817;
  t1846 = t588*t1807;
  t1848 = t1737*t592;
  t1868 = t1757*t606;
  t1875 = t1846 + t1848 + t1868;
  t1888 = t683*t1737;
  t1898 = 0.340999127418*t207*t1757;
  t1927 = t1807*t756;
  t1933 = t1888 + t1898 + t1927;
  t1839 = 0.340999127418*t189*t1828;
  t1882 = t581*t1875;
  t1982 = t658*t1933;
  t1994 = t1839 + t1882 + t1982;
  t2002 = t810*t1828;
  t2007 = t824*t1875;
  t2021 = 0.340999127418*t189*t1933;
  t2029 = t2002 + t2007 + t2021;
  t2035 = t922*t1828;
  t2036 = t961*t1875;
  t2052 = t972*t1933;
  t2057 = t2035 + t2036 + t2052;
  t1200 = 0.218018*t36;
  t1212 = t1199 + t1200;
  t1998 = t167*t1994;
  t2031 = t794*t2029;
  t2059 = t886*t2057;
  t2074 = t1998 + t2031 + t2059;
  t2091 = t1035*t1994;
  t2093 = 0.340999127418*t884*t2029;
  t2109 = t1055*t2057;
  t2112 = t2091 + t2093 + t2109;
  t2136 = 0.340999127418*t884*t1994;
  t2141 = t1137*t2029;
  t2147 = t1142*t2057;
  t2149 = t2136 + t2141 + t2147;
  t1297 = -0.353861996165*t34;
  t1300 = 1. + t1297;
  t1391 = -0.952469601425*t34;
  t1392 = 1. + t1391;
  t2087 = t71*t2074;
  t2125 = 0.340999127418*t1025*t2112;
  t2151 = t1120*t2149;
  t2152 = t2087 + t2125 + t2151;
  t1399 = 0.803828*t36;
  t1400 = t1397 + t1399;
  t2155 = t1234*t2074;
  t2156 = t1259*t2112;
  t2158 = 0.340999127418*t1025*t2149;
  t2164 = t2155 + t2156 + t2158;
  t1405 = -0.553471*t36;
  t1415 = t35 + t1405;
  t2177 = t1313*t2074;
  t2181 = t1324*t2112;
  t2195 = t1340*t2149;
  t2197 = t2177 + t2181 + t2195;
  t1457 = -0.803828*t36;
  t1469 = t1397 + t1457;
  t1474 = -0.693671301908*t34;
  t1482 = 1. + t1474;
  t1492 = -0.218018*t36;
  t1498 = t1199 + t1492;
  t2386 = t381*t238*t269;
  t2392 = -1.*t381*t243*t299;
  t2407 = t2386 + t2392;
  t2417 = t381*t243*t238;
  t2425 = t381*t269*t299;
  t2426 = t2417 + t2425;
  t2567 = -1.*t471*t268;
  t2573 = t2407*t475;
  t2580 = t2426*t488;
  t2583 = t2567 + t2573 + t2580;
  t2416 = -0.340999127418*t231*t2407;
  t2439 = t339*t2426;
  t2467 = -1.*t268*t399;
  t2495 = t2416 + t2439 + t2467;
  t2527 = t438*t2407;
  t2536 = -0.340999127418*t231*t2426;
  t2539 = -1.*t268*t459;
  t2558 = t2527 + t2536 + t2539;
  t2508 = 0.340999127418*t207*t2495;
  t2560 = t431*t2558;
  t2591 = t2583*t533;
  t2593 = t2508 + t2560 + t2591;
  t2606 = t588*t2583;
  t2614 = t2495*t592;
  t2661 = t2558*t606;
  t2662 = t2606 + t2614 + t2661;
  t2673 = t683*t2495;
  t2676 = 0.340999127418*t207*t2558;
  t2678 = t2583*t756;
  t2680 = t2673 + t2676 + t2678;
  t2604 = 0.340999127418*t189*t2593;
  t2664 = t581*t2662;
  t2681 = t658*t2680;
  t2683 = t2604 + t2664 + t2681;
  t2746 = t810*t2593;
  t2748 = t824*t2662;
  t2758 = 0.340999127418*t189*t2680;
  t2763 = t2746 + t2748 + t2758;
  t2776 = t922*t2593;
  t2783 = t961*t2662;
  t2793 = t972*t2680;
  t2803 = t2776 + t2783 + t2793;
  t2714 = t167*t2683;
  t2769 = t794*t2763;
  t2805 = t886*t2803;
  t2812 = t2714 + t2769 + t2805;
  t2827 = t1035*t2683;
  t2833 = 0.340999127418*t884*t2763;
  t2857 = t1055*t2803;
  t2867 = t2827 + t2833 + t2857;
  t2903 = 0.340999127418*t884*t2683;
  t2916 = t1137*t2763;
  t2918 = t1142*t2803;
  t2920 = t2903 + t2916 + t2918;
  t2816 = t71*t2812;
  t2898 = 0.340999127418*t1025*t2867;
  t2927 = t1120*t2920;
  t2937 = t2816 + t2898 + t2927;
  t3006 = t1234*t2812;
  t3010 = t1259*t2867;
  t3014 = 0.340999127418*t1025*t2920;
  t3019 = t3006 + t3010 + t3014;
  t3027 = t1313*t2812;
  t3028 = t1324*t2867;
  t3035 = t1340*t2920;
  t3037 = t3027 + t3028 + t3035;
  t1395 = t1392*t1194;
  t1401 = t1400*t1284;
  t1420 = t1415*t1367;
  t1448 = t1395 + t1401 + t1420;
  t1473 = t1469*t1194;
  t1490 = t1482*t1284;
  t1503 = t1498*t1367;
  t1524 = t1473 + t1490 + t1503;
  t2277 = t1392*t2152;
  t2284 = t1400*t2164;
  t2290 = t1415*t2197;
  t2300 = t2277 + t2284 + t2290;
  t2315 = t1469*t2152;
  t2319 = t1482*t2164;
  t2329 = t1498*t2197;
  t2341 = t2315 + t2319 + t2329;
  t3045 = t1392*t2937;
  t3055 = t1400*t3019;
  t3067 = t1415*t3037;
  t3069 = t3045 + t3055 + t3067;
  t3079 = t1469*t2937;
  t3080 = t1482*t3019;
  t3106 = t1498*t3037;
  t3108 = t3079 + t3080 + t3106;
  t1197 = t43*t1194;
  t1294 = t1212*t1284;
  t1379 = t1300*t1367;
  t1385 = t1197 + t1294 + t1379;
  t2154 = t43*t2152;
  t2172 = t1212*t2164;
  t2208 = t1300*t2197;
  t2212 = t2154 + t2172 + t2208;
  t2940 = t43*t2937;
  t3020 = t1212*t3019;
  t3038 = t1300*t3037;
  t3040 = t2940 + t3020 + t3038;
  t3484 = 0.120666640478*t34;
  t3533 = -0.444895486988*t34;
  t3470 = 0.175248972904*t34;
  t3303 = -1.*t238;
  t3308 = 1. + t3303;
  t3312 = 0.091*t3308;
  t3313 = 0. + t3312;
  t3318 = 0.091*t299;
  t3323 = 0. + t3318;
  t3351 = -0.04500040093286238*t231;
  t3353 = 0.07877663122399998*t475;
  t3356 = 0.031030906668*t488;
  t3386 = 0. + t3351 + t3353 + t3356;
  t3395 = -3.2909349868922137e-7*var1[7];
  t3402 = 0.03103092645718495*t231;
  t3406 = -0.045000372235*t399;
  t3408 = t3395 + t3402 + t3406;
  t3412 = 1.296332362046933e-7*var1[7];
  t3416 = 0.07877668146182712*t231;
  t3422 = -0.045000372235*t459;
  t3424 = t3412 + t3416 + t3422;
  t3462 = 2.838136523330542e-8*var1[12];
  t3469 = 0.2845150083511607*t34;
  t3472 = t3470 + t1405;
  t3475 = 0.44503472296900004*t3472;
  t3486 = t3484 + t1399;
  t3491 = -0.5286755231320001*t3486;
  t3509 = t3462 + t3469 + t3475 + t3491;
  t3516 = -7.20503013377005e-8*var1[12];
  t3518 = -0.3667270384178674*t34;
  t3522 = t3484 + t1457;
  t3523 = 0.29871295412*t3522;
  t3539 = t3533 + t1492;
  t3545 = 0.44503472296900004*t3539;
  t3546 = t3516 + t3518 + t3523 + t3545;
  t3560 = -1.0464152525368286e-7*var1[12];
  t3562 = 0.15748087543254813*t34;
  t3572 = t3533 + t1200;
  t3595 = -0.5286755231320001*t3572;
  t3604 = t3470 + t42;
  t3636 = 0.29871295412*t3604;
  t3639 = t3560 + t3562 + t3595 + t3636;
  t3656 = 0.06199697675299678*t1025;
  t3663 = 0.324290713329*t1340;
  t3665 = -0.823260828522*t1324;
  t3666 = 0. + t3656 + t3663 + t3665;
  t3673 = 2.95447451120871e-8*var1[11];
  t3675 = -0.8232613535360118*t1025;
  t3687 = 0.061996937216*t1234;
  t3691 = t3673 + t3675 + t3687;
  t3700 = 7.500378623168247e-8*var1[11];
  t3707 = 0.32429092013729516*t1025;
  t3716 = 0.061996937216*t71;
  t3717 = t3700 + t3707 + t3716;
  t3721 = 2.281945176511838e-8*var1[10];
  t3729 = -0.5905366811997648*t884;
  t3730 = -0.262809976934*t1055;
  t3735 = t3721 + t3729 + t3730;
  t3743 = 5.7930615939377813e-8*var1[10];
  t3748 = 0.23261833304643187*t884;
  t3750 = -0.262809976934*t1142;
  t3751 = t3743 + t3748 + t3750;
  t3753 = -0.26281014453449253*t884;
  t3765 = 0.23261818470000004*t794;
  t3767 = -0.5905363046000001*t167;
  t3768 = 0. + t3753 + t3765 + t3767;
  t3774 = 3.2909349868922137e-7*var1[8];
  t3791 = 0.055653945343889656*t207;
  t3792 = -0.045000372235*t533;
  t3798 = t3774 + t3791 + t3792;
  t3806 = -1.5981976069815686e-7*var1[9];
  t3812 = 0.08675267452931407*t189;
  t3817 = 0.039853013046*t824;
  t3819 = t3806 + t3812 + t3817;
  t3828 = -0.04500040093286238*t207;
  t3832 = -0.141285834136*t592;
  t3833 = 0.055653909852*t606;
  t3834 = 0. + t3828 + t3832 + t3833;
  t3849 = 0.039853038461262744*t189;
  t3872 = 0.086752619205*t922;
  t3874 = -0.22023459268999998*t972;
  t3879 = 0. + t3849 + t3872 + t3874;
  t3897 = 1.296332362046933e-7*var1[8];
  t3898 = -0.14128592423750855*t207;
  t3900 = -0.045000372235*t756;
  t3902 = t3897 + t3898 + t3900;
  t3912 = -6.295460977284962e-8*var1[9];
  t3913 = -0.22023473313910558*t189;
  t3916 = 0.039853013046*t581;
  t3919 = t3912 + t3913 + t3916;
  p_output1[0]=0.993567*t1385 - 0.041508*t1448 + 0.105375*t1524;
  p_output1[1]=0.993567*t2212 - 0.041508*t2300 + 0.105375*t2341;
  p_output1[2]=0.993567*t3040 - 0.041508*t3069 + 0.105375*t3108;
  p_output1[3]=0.;
  p_output1[4]=0.930418*t1448 + 0.366501*t1524;
  p_output1[5]=0.930418*t2300 + 0.366501*t2341;
  p_output1[6]=0.930418*t3069 + 0.366501*t3108;
  p_output1[7]=0.;
  p_output1[8]=-0.113255*t1385 - 0.364143*t1448 + 0.924432*t1524;
  p_output1[9]=-0.113255*t2212 - 0.364143*t2300 + 0.924432*t2341;
  p_output1[10]=-0.113255*t3040 - 0.364143*t3069 + 0.924432*t3108;
  p_output1[11]=0.;
  p_output1[12]=0. + 0.066017*t1385 + 0.428961*t1448 - 0.845686*t1524 + t273*t3313 + t298*t3323 + t324*t3424 + t1194*t3509 + t1284*t3546 + t3408*t360 + t1367*t3639 + t1090*t3691 + t1187*t3717 + t259*t3386*t381 + t3902*t424 + t3798*t461 + t3834*t511 + t3819*t536 + t3879*t650 + t3919*t764 + t3735*t782 + t3751*t844 + t3768*t991 + t3666*t999 + var1[0];
  p_output1[13]=0. + 0.066017*t2212 + 0.428961*t2300 - 0.845686*t2341 + t1562*t3313 + t1617*t3323 + t1691*t3408 + t1659*t3424 + t2152*t3509 + t2164*t3546 + t2197*t3639 + t2074*t3666 + t2112*t3691 + t2149*t3717 + t1994*t3735 + t2029*t3751 + t2057*t3768 + t1757*t3798 + t244*t3386*t381 + t1828*t3819 + t1807*t3834 + t1875*t3879 + t1737*t3902 + t1933*t3919 + var1[1];
  p_output1[14]=0. + 0.066017*t3040 + 0.428961*t3069 - 0.845686*t3108 - 1.*t268*t3386 + t2426*t3408 + t2407*t3424 + t2937*t3509 + t3019*t3546 + t3037*t3639 + t2812*t3666 + t2867*t3691 + t2920*t3717 + t2683*t3735 + t2763*t3751 + t2803*t3768 + t2558*t3798 + t269*t3313*t381 + t243*t3323*t381 + t2593*t3819 + t2583*t3834 + t2662*t3879 + t2495*t3902 + t2680*t3919 + var1[2];
  p_output1[15]=1.;
}



void T_LeftToeBottom_src(double *p_output1, const double *var1)
{
  // Call Subroutines
  output1(p_output1, var1);

}
