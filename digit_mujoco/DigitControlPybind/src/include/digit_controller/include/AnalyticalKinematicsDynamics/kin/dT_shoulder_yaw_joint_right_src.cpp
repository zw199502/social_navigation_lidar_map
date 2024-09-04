/*
 * Automatically Generated from Mathematica.
 * Mon 4 Jul 2022 20:55:32 GMT-04:00
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "dT_shoulder_yaw_joint_right_src.h"

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
static void output1(double *p_output1,const double *var1,const double *var2)
{
  double t380;
  double t681;
  double t565;
  double t668;
  double t669;
  double t908;
  double t783;
  double t788;
  double t1082;
  double t109;
  double t143;
  double t251;
  double t254;
  double t433;
  double t528;
  double t853;
  double t1089;
  double t1092;
  double t1179;
  double t1213;
  double t1218;
  double t481;
  double t518;
  double t551;
  double t1131;
  double t1132;
  double t1176;
  double t1222;
  double t1226;
  double t1334;
  double t1335;
  double t1338;
  double t1345;
  double t1368;
  double t1413;
  double t1422;
  double t1423;
  double t252;
  double t1541;
  double t1542;
  double t1556;
  double t1576;
  double t1592;
  double t1597;
  double t1608;
  double t1632;
  double t1642;
  double t1661;
  double t1732;
  double t1251;
  double t1306;
  double t1325;
  double t2093;
  double t2057;
  double t2155;
  double t2158;
  double t2168;
  double t2170;
  double t2172;
  double t288;
  double t291;
  double t2165;
  double t2183;
  double t2211;
  double t2231;
  double t2238;
  double t2251;
  double t1521;
  double t1534;
  double t1740;
  double t1754;
  double t2224;
  double t2257;
  double t2261;
  double t1687;
  double t1723;
  double t2310;
  double t2321;
  double t2343;
  double t1781;
  double t1783;
  double t2360;
  double t2376;
  double t2393;
  double t1895;
  double t1917;
  double t1850;
  double t1887;
  double t1930;
  double t1986;
  double t2598;
  double t2610;
  double t2616;
  double t2576;
  double t2619;
  double t2623;
  double t2640;
  double t2644;
  double t2648;
  double t2629;
  double t2653;
  double t2654;
  double t2664;
  double t2665;
  double t2672;
  double t2701;
  double t2708;
  double t2723;
  double t2857;
  double t2871;
  double t2907;
  double t2849;
  double t2874;
  double t2884;
  double t2909;
  double t2929;
  double t2937;
  double t2965;
  double t2988;
  double t2993;
  double t2994;
  double t2997;
  double t2984;
  double t3045;
  double t3186;
  double t3191;
  double t3193;
  double t3219;
  double t3247;
  double t3260;
  double t3268;
  double t3279;
  double t3300;
  double t3303;
  double t3306;
  double t3312;
  double t3520;
  double t3524;
  double t3531;
  double t3535;
  double t3538;
  double t3541;
  double t3533;
  double t3572;
  double t3576;
  double t3583;
  double t3589;
  double t3603;
  double t3519;
  double t3577;
  double t3606;
  double t3608;
  double t3613;
  double t3630;
  double t3637;
  double t3642;
  double t3653;
  double t3659;
  double t3675;
  double t3687;
  double t3826;
  double t3828;
  double t3848;
  double t3878;
  double t3879;
  double t3895;
  double t3821;
  double t3849;
  double t3904;
  double t3909;
  double t3922;
  double t3924;
  double t3937;
  double t3940;
  double t3945;
  double t3948;
  double t3953;
  double t3962;
  double t4182;
  double t4187;
  double t4189;
  double t4192;
  double t4197;
  double t4212;
  double t4215;
  double t4230;
  double t4203;
  double t4234;
  double t4239;
  double t4254;
  double t4262;
  double t4263;
  double t4270;
  double t4272;
  double t4274;
  double t4382;
  double t4383;
  double t4387;
  double t4381;
  double t4394;
  double t4395;
  double t4407;
  double t4410;
  double t4415;
  double t4402;
  double t4416;
  double t4426;
  double t4437;
  double t4443;
  double t4449;
  double t4455;
  double t4458;
  double t4460;
  double t2904;
  double t2908;
  double t4567;
  double t4568;
  double t2973;
  double t2985;
  double t3017;
  double t3018;
  double t4561;
  double t4569;
  double t4570;
  double t4572;
  double t4573;
  double t4574;
  double t4576;
  double t3041;
  double t3048;
  double t4593;
  double t4594;
  double t4596;
  double t4600;
  double t3073;
  double t3076;
  double t3123;
  double t3150;
  double t4671;
  double t4672;
  double t4673;
  double t4675;
  double t4679;
  double t4680;
  double t4683;
  double t4686;
  double t4690;
  double t4694;
  double t4701;
  double t4708;
  double t4788;
  double t4790;
  double t4791;
  double t4813;
  double t4815;
  double t4816;
  double t4799;
  double t4822;
  double t4826;
  double t4831;
  double t4834;
  double t4840;
  double t4846;
  double t4847;
  double t4864;
  double t4947;
  double t4948;
  double t4950;
  double t4955;
  double t4956;
  double t4957;
  double t4952;
  double t4958;
  double t4966;
  double t4969;
  double t4970;
  double t4973;
  double t4976;
  double t4978;
  double t4982;
  double t5113;
  double t5124;
  double t5133;
  double t5145;
  double t5162;
  double t5163;
  double t5176;
  double t5179;
  double t5192;
  double t5198;
  double t5356;
  double t5357;
  double t5361;
  double t5364;
  double t5371;
  double t5385;
  double t5388;
  double t5392;
  double t5399;
  double t5400;
  double t5403;
  double t5411;
  double t5601;
  double t5621;
  double t5622;
  double t5629;
  double t5642;
  double t5645;
  double t5584;
  double t5624;
  double t5649;
  double t5651;
  double t5669;
  double t5672;
  double t5675;
  double t5676;
  double t5683;
  double t5686;
  double t5689;
  double t5696;
  double t1727;
  double t1765;
  double t1803;
  double t1837;
  double t1894;
  double t1918;
  double t2027;
  double t2033;
  double t2448;
  double t2457;
  double t2458;
  double t2459;
  double t2505;
  double t2522;
  double t2524;
  double t2560;
  double t2771;
  double t2776;
  double t2778;
  double t2779;
  double t2791;
  double t2806;
  double t2821;
  double t2826;
  double t2898;
  double t2967;
  double t2999;
  double t3001;
  double t3021;
  double t3040;
  double t3060;
  double t3062;
  double t3239;
  double t3294;
  double t3317;
  double t3321;
  double t3334;
  double t3340;
  double t3342;
  double t3353;
  double t3711;
  double t3716;
  double t3723;
  double t3734;
  double t3745;
  double t3750;
  double t3751;
  double t3764;
  double t3979;
  double t3990;
  double t4006;
  double t4009;
  double t4013;
  double t4035;
  double t4036;
  double t4044;
  double t4086;
  double t4092;
  double t4094;
  double t4109;
  double t4119;
  double t4123;
  double t4139;
  double t4153;
  double t4291;
  double t4299;
  double t4303;
  double t4341;
  double t4350;
  double t4352;
  double t4354;
  double t4355;
  double t4487;
  double t4499;
  double t4502;
  double t4503;
  double t4540;
  double t4541;
  double t4547;
  double t4552;
  double t4571;
  double t4592;
  double t4604;
  double t4612;
  double t4618;
  double t4619;
  double t4621;
  double t4626;
  double t4678;
  double t4688;
  double t4711;
  double t4727;
  double t4730;
  double t4735;
  double t4740;
  double t4747;
  double t4891;
  double t4893;
  double t4896;
  double t4898;
  double t4906;
  double t4911;
  double t4932;
  double t4934;
  double t5029;
  double t5044;
  double t5047;
  double t5051;
  double t5067;
  double t5077;
  double t5088;
  double t5092;
  double t5143;
  double t5173;
  double t5210;
  double t5218;
  double t5228;
  double t5230;
  double t5237;
  double t5275;
  double t5369;
  double t5397;
  double t5429;
  double t5450;
  double t5454;
  double t5471;
  double t5487;
  double t5490;
  double t5731;
  double t5754;
  double t5755;
  double t5765;
  double t5780;
  double t5787;
  double t5788;
  double t5789;
  double t1230;
  double t1477;
  double t1666;
  double t1667;
  double t1678;
  double t2307;
  double t2353;
  double t2395;
  double t2422;
  double t2445;
  double t2657;
  double t2694;
  double t2749;
  double t2762;
  double t2767;
  double t3122;
  double t3155;
  double t3167;
  double t3174;
  double t3178;
  double t3372;
  double t3381;
  double t3384;
  double t3412;
  double t3430;
  double t3610;
  double t3651;
  double t3698;
  double t3699;
  double t3708;
  double t3912;
  double t3942;
  double t3967;
  double t3972;
  double t3977;
  double t4065;
  double t4069;
  double t4070;
  double t4077;
  double t4082;
  double t4242;
  double t4268;
  double t4277;
  double t4281;
  double t4283;
  double t4435;
  double t4452;
  double t4463;
  double t4470;
  double t4478;
  double t4631;
  double t4637;
  double t4642;
  double t4644;
  double t4665;
  double t4756;
  double t4759;
  double t4763;
  double t4767;
  double t4770;
  double t4830;
  double t4845;
  double t4866;
  double t4877;
  double t4878;
  double t4967;
  double t4975;
  double t4987;
  double t5005;
  double t5010;
  double t5285;
  double t5291;
  double t5297;
  double t5303;
  double t5320;
  double t5510;
  double t5528;
  double t5540;
  double t5548;
  double t5551;
  double t5656;
  double t5679;
  double t5700;
  double t5716;
  double t5717;
  double t6582;
  double t6587;
  double t6720;
  double t6794;
  double t6743;
  double t6646;
  double t6648;
  double t6650;
  double t6599;
  double t6600;
  double t6606;
  double t6681;
  double t6684;
  double t6687;
  double t6689;
  double t6652;
  double t6660;
  double t6671;
  double t6678;
  double t6783;
  double t6787;
  double t6796;
  double t6800;
  double t6816;
  double t6817;
  double t6818;
  double t6696;
  double t6705;
  double t6739;
  double t6742;
  double t6746;
  double t6748;
  double t6758;
  double t6832;
  double t6833;
  double t6840;
  double t6841;
  double t6842;
  double t6843;
  double t6844;
  double t6967;
  double t7045;
  double t7014;
  double t6618;
  double t6621;
  double t6622;
  double t6637;
  double t6903;
  double t6911;
  double t6912;
  double t6917;
  double t6922;
  double t6976;
  double t7009;
  double t7016;
  double t7018;
  double t7022;
  double t7023;
  double t7035;
  double t7040;
  double t7051;
  double t7054;
  double t7055;
  double t7057;
  double t7071;
  double t7072;
  double t7073;
  double t7077;
  double t7078;
  double t7079;
  double t7114;
  double t7118;
  double t7119;
  double t7121;
  double t7122;
  double t7124;
  double t7126;
  double t7135;
  double t7138;
  t380 = Cos(var1[3]);
  t681 = Cos(var1[4]);
  t565 = Cos(var1[25]);
  t668 = -1.*t565;
  t669 = 1. + t668;
  t908 = Cos(var1[24]);
  t783 = Cos(var1[5]);
  t788 = Sin(var1[24]);
  t1082 = Sin(var1[5]);
  t109 = Cos(var1[26]);
  t143 = -1.*t109;
  t251 = 1. + t143;
  t254 = Sin(var1[26]);
  t433 = Sin(var1[25]);
  t528 = Sin(var1[4]);
  t853 = -1.*t380*t681*t783*t788;
  t1089 = t908*t380*t681*t1082;
  t1092 = t853 + t1089;
  t1179 = t908*t380*t681*t783;
  t1213 = t380*t681*t788*t1082;
  t1218 = t1179 + t1213;
  t481 = -0.994522*t433;
  t518 = 0. + t481;
  t551 = -1.*t380*t518*t528;
  t1131 = -0.103955395616*t669*t1092;
  t1132 = -0.9890740084840001*t669;
  t1176 = 1. + t1132;
  t1222 = t1176*t1218;
  t1226 = t551 + t1131 + t1222;
  t1334 = -0.104528*t433;
  t1335 = 0. + t1334;
  t1338 = -1.*t380*t1335*t528;
  t1345 = -0.010926102783999999*t669;
  t1368 = 1. + t1345;
  t1413 = t1368*t1092;
  t1422 = -0.103955395616*t669*t1218;
  t1423 = t1338 + t1413 + t1422;
  t252 = -0.49726168403800003*t251;
  t1541 = -1.0000001112680001*t669;
  t1542 = 1. + t1541;
  t1556 = -1.*t1542*t380*t528;
  t1576 = 0.104528*t433;
  t1592 = 0. + t1576;
  t1597 = t1592*t1092;
  t1608 = 0.994522*t433;
  t1632 = 0. + t1608;
  t1642 = t1632*t1218;
  t1661 = t1556 + t1597 + t1642;
  t1732 = 0.051978134642000004*t251;
  t1251 = -0.05226439969100001*t251;
  t1306 = -0.703234*t254;
  t1325 = t1251 + t1306;
  t2093 = Sin(var1[3]);
  t2057 = t380*t783*t528;
  t2155 = t2093*t1082;
  t2158 = t2057 + t2155;
  t2168 = t783*t2093;
  t2170 = -1.*t380*t528*t1082;
  t2172 = t2168 + t2170;
  t288 = 0.073913*t254;
  t291 = t252 + t288;
  t2165 = t788*t2158;
  t2183 = t908*t2172;
  t2211 = t2165 + t2183;
  t2231 = t908*t2158;
  t2238 = -1.*t788*t2172;
  t2251 = t2231 + t2238;
  t1521 = -0.500001190325*t251;
  t1534 = 1. + t1521;
  t1740 = -0.707107*t254;
  t1754 = t1732 + t1740;
  t2224 = -0.103955395616*t669*t2211;
  t2257 = t1368*t2251;
  t2261 = t2224 + t2257;
  t1687 = -0.5054634410180001*t251;
  t1723 = 1. + t1687;
  t2310 = t1176*t2211;
  t2321 = -0.103955395616*t669*t2251;
  t2343 = t2310 + t2321;
  t1781 = -0.073913*t254;
  t1783 = t252 + t1781;
  t2360 = t1632*t2211;
  t2376 = t1592*t2251;
  t2393 = t2360 + t2376;
  t1895 = -0.9945383682050002*t251;
  t1917 = 1. + t1895;
  t1850 = 0.707107*t254;
  t1887 = t1732 + t1850;
  t1930 = 0.703234*t254;
  t1986 = t1251 + t1930;
  t2598 = -1.*t783*t2093;
  t2610 = t380*t528*t1082;
  t2616 = t2598 + t2610;
  t2576 = -1.*t788*t2158;
  t2619 = t908*t2616;
  t2623 = t2576 + t2619;
  t2640 = -1.*t908*t2158;
  t2644 = -1.*t788*t2616;
  t2648 = t2640 + t2644;
  t2629 = -0.103955395616*t669*t2623;
  t2653 = t1368*t2648;
  t2654 = t2629 + t2653;
  t2664 = t1176*t2623;
  t2665 = -0.103955395616*t669*t2648;
  t2672 = t2664 + t2665;
  t2701 = t1632*t2623;
  t2708 = t1592*t2648;
  t2723 = t2701 + t2708;
  t2857 = t788*t2616;
  t2871 = t2231 + t2857;
  t2907 = 0.051978134642000004*t254;
  t2849 = t380*t681*t518;
  t2874 = t1176*t2871;
  t2884 = t2849 + t2629 + t2874;
  t2909 = t380*t681*t1335;
  t2929 = t1368*t2623;
  t2937 = -0.103955395616*t669*t2871;
  t2965 = t2909 + t2929 + t2937;
  t2988 = t1542*t380*t681;
  t2993 = t1592*t2623;
  t2994 = t1632*t2871;
  t2997 = t2988 + t2993 + t2994;
  t2984 = -0.49726168403800003*t254;
  t3045 = -0.05226439969100001*t254;
  t3186 = -1.0000001112680001*t380*t681*t433;
  t3191 = 0.104528*t565*t2623;
  t3193 = 0.994522*t565*t2871;
  t3219 = t3186 + t3191 + t3193;
  t3247 = -0.994522*t565*t380*t681;
  t3260 = -0.103955395616*t433*t2623;
  t3268 = -0.9890740084840001*t433*t2871;
  t3279 = t3247 + t3260 + t3268;
  t3300 = -0.104528*t565*t380*t681;
  t3303 = -0.010926102783999999*t433*t2623;
  t3306 = -0.103955395616*t433*t2871;
  t3312 = t3300 + t3303 + t3306;
  t3520 = -1.*t783*t2093*t528;
  t3524 = t380*t1082;
  t3531 = t3520 + t3524;
  t3535 = -1.*t380*t783;
  t3538 = -1.*t2093*t528*t1082;
  t3541 = t3535 + t3538;
  t3533 = -1.*t788*t3531;
  t3572 = t908*t3541;
  t3576 = t3533 + t3572;
  t3583 = t908*t3531;
  t3589 = t788*t3541;
  t3603 = t3583 + t3589;
  t3519 = -1.*t681*t518*t2093;
  t3577 = -0.103955395616*t669*t3576;
  t3606 = t1176*t3603;
  t3608 = t3519 + t3577 + t3606;
  t3613 = -1.*t681*t1335*t2093;
  t3630 = t1368*t3576;
  t3637 = -0.103955395616*t669*t3603;
  t3642 = t3613 + t3630 + t3637;
  t3653 = -1.*t1542*t681*t2093;
  t3659 = t1592*t3576;
  t3675 = t1632*t3603;
  t3687 = t3653 + t3659 + t3675;
  t3826 = -1.*t681*t783*t788*t2093;
  t3828 = t908*t681*t2093*t1082;
  t3848 = t3826 + t3828;
  t3878 = t908*t681*t783*t2093;
  t3879 = t681*t788*t2093*t1082;
  t3895 = t3878 + t3879;
  t3821 = -1.*t518*t2093*t528;
  t3849 = -0.103955395616*t669*t3848;
  t3904 = t1176*t3895;
  t3909 = t3821 + t3849 + t3904;
  t3922 = -1.*t1335*t2093*t528;
  t3924 = t1368*t3848;
  t3937 = -0.103955395616*t669*t3895;
  t3940 = t3922 + t3924 + t3937;
  t3945 = -1.*t1542*t2093*t528;
  t3948 = t1592*t3848;
  t3953 = t1632*t3895;
  t3962 = t3945 + t3948 + t3953;
  t4182 = t783*t2093*t528;
  t4187 = -1.*t380*t1082;
  t4189 = t4182 + t4187;
  t4192 = t788*t4189;
  t4197 = t4192 + t3572;
  t4212 = t908*t4189;
  t4215 = -1.*t788*t3541;
  t4230 = t4212 + t4215;
  t4203 = -0.103955395616*t669*t4197;
  t4234 = t1368*t4230;
  t4239 = t4203 + t4234;
  t4254 = t1176*t4197;
  t4262 = -0.103955395616*t669*t4230;
  t4263 = t4254 + t4262;
  t4270 = t1632*t4197;
  t4272 = t1592*t4230;
  t4274 = t4270 + t4272;
  t4382 = t380*t783;
  t4383 = t2093*t528*t1082;
  t4387 = t4382 + t4383;
  t4381 = -1.*t788*t4189;
  t4394 = t908*t4387;
  t4395 = t4381 + t4394;
  t4407 = -1.*t908*t4189;
  t4410 = -1.*t788*t4387;
  t4415 = t4407 + t4410;
  t4402 = -0.103955395616*t669*t4395;
  t4416 = t1368*t4415;
  t4426 = t4402 + t4416;
  t4437 = t1176*t4395;
  t4443 = -0.103955395616*t669*t4415;
  t4449 = t4437 + t4443;
  t4455 = t1632*t4395;
  t4458 = t1592*t4415;
  t4460 = t4455 + t4458;
  t2904 = -0.707107*t109;
  t2908 = t2904 + t2907;
  t4567 = t788*t4387;
  t4568 = t4212 + t4567;
  t2973 = -0.073913*t109;
  t2985 = t2973 + t2984;
  t3017 = 0.707107*t109;
  t3018 = t3017 + t2907;
  t4561 = t681*t518*t2093;
  t4569 = t1176*t4568;
  t4570 = t4561 + t4402 + t4569;
  t4572 = t681*t1335*t2093;
  t4573 = t1368*t4395;
  t4574 = -0.103955395616*t669*t4568;
  t4576 = t4572 + t4573 + t4574;
  t3041 = 0.703234*t109;
  t3048 = t3041 + t3045;
  t4593 = t1542*t681*t2093;
  t4594 = t1592*t4395;
  t4596 = t1632*t4568;
  t4600 = t4593 + t4594 + t4596;
  t3073 = 0.073913*t109;
  t3076 = t3073 + t2984;
  t3123 = -0.703234*t109;
  t3150 = t3123 + t3045;
  t4671 = -1.0000001112680001*t681*t433*t2093;
  t4672 = 0.104528*t565*t4395;
  t4673 = 0.994522*t565*t4568;
  t4675 = t4671 + t4672 + t4673;
  t4679 = -0.994522*t565*t681*t2093;
  t4680 = -0.103955395616*t433*t4395;
  t4683 = -0.9890740084840001*t433*t4568;
  t4686 = t4679 + t4680 + t4683;
  t4690 = -0.104528*t565*t681*t2093;
  t4694 = -0.010926102783999999*t433*t4395;
  t4701 = -0.103955395616*t433*t4568;
  t4708 = t4690 + t4694 + t4701;
  t4788 = -1.*t681*t783*t788;
  t4790 = t908*t681*t1082;
  t4791 = t4788 + t4790;
  t4813 = -1.*t908*t681*t783;
  t4815 = -1.*t681*t788*t1082;
  t4816 = t4813 + t4815;
  t4799 = -0.103955395616*t669*t4791;
  t4822 = t1368*t4816;
  t4826 = t4799 + t4822;
  t4831 = t1176*t4791;
  t4834 = -0.103955395616*t669*t4816;
  t4840 = t4831 + t4834;
  t4846 = t1632*t4791;
  t4847 = t1592*t4816;
  t4864 = t4846 + t4847;
  t4947 = t681*t783*t788;
  t4948 = -1.*t908*t681*t1082;
  t4950 = t4947 + t4948;
  t4955 = t908*t681*t783;
  t4956 = t681*t788*t1082;
  t4957 = t4955 + t4956;
  t4952 = -0.103955395616*t669*t4950;
  t4958 = t1368*t4957;
  t4966 = t4952 + t4958;
  t4969 = t1176*t4950;
  t4970 = -0.103955395616*t669*t4957;
  t4973 = t4969 + t4970;
  t4976 = t1632*t4950;
  t4978 = t1592*t4957;
  t4982 = t4976 + t4978;
  t5113 = -1.*t518*t528;
  t5124 = t1176*t4957;
  t5133 = t5113 + t4799 + t5124;
  t5145 = -1.*t1335*t528;
  t5162 = t1368*t4791;
  t5163 = t5145 + t5162 + t4970;
  t5176 = -1.*t1542*t528;
  t5179 = t1592*t4791;
  t5192 = t1632*t4957;
  t5198 = t5176 + t5179 + t5192;
  t5356 = 1.0000001112680001*t433*t528;
  t5357 = 0.104528*t565*t4791;
  t5361 = 0.994522*t565*t4957;
  t5364 = t5356 + t5357 + t5361;
  t5371 = 0.994522*t565*t528;
  t5385 = -0.103955395616*t433*t4791;
  t5388 = -0.9890740084840001*t433*t4957;
  t5392 = t5371 + t5385 + t5388;
  t5399 = 0.104528*t565*t528;
  t5400 = -0.010926102783999999*t433*t4791;
  t5403 = -0.103955395616*t433*t4957;
  t5411 = t5399 + t5400 + t5403;
  t5601 = t783*t788*t528;
  t5621 = -1.*t908*t528*t1082;
  t5622 = t5601 + t5621;
  t5629 = -1.*t908*t783*t528;
  t5642 = -1.*t788*t528*t1082;
  t5645 = t5629 + t5642;
  t5584 = -1.*t681*t518;
  t5624 = -0.103955395616*t669*t5622;
  t5649 = t1176*t5645;
  t5651 = t5584 + t5624 + t5649;
  t5669 = -1.*t681*t1335;
  t5672 = t1368*t5622;
  t5675 = -0.103955395616*t669*t5645;
  t5676 = t5669 + t5672 + t5675;
  t5683 = -1.*t1542*t681;
  t5686 = t1592*t5622;
  t5689 = t1632*t5645;
  t5696 = t5683 + t5686 + t5689;
  t1727 = t1723*t1226;
  t1765 = t1754*t1423;
  t1803 = t1783*t1661;
  t1837 = t1727 + t1765 + t1803;
  t1894 = t1887*t1226;
  t1918 = t1917*t1423;
  t2027 = t1986*t1661;
  t2033 = t1894 + t1918 + t2027;
  t2448 = t1754*t2261;
  t2457 = t1723*t2343;
  t2458 = t1783*t2393;
  t2459 = t2448 + t2457 + t2458;
  t2505 = t1917*t2261;
  t2522 = t1887*t2343;
  t2524 = t1986*t2393;
  t2560 = t2505 + t2522 + t2524;
  t2771 = t1754*t2654;
  t2776 = t1723*t2672;
  t2778 = t1783*t2723;
  t2779 = t2771 + t2776 + t2778;
  t2791 = t1917*t2654;
  t2806 = t1887*t2672;
  t2821 = t1986*t2723;
  t2826 = t2791 + t2806 + t2821;
  t2898 = -0.5054634410180001*t254*t2884;
  t2967 = t2908*t2965;
  t2999 = t2985*t2997;
  t3001 = t2898 + t2967 + t2999;
  t3021 = t3018*t2884;
  t3040 = -0.9945383682050002*t254*t2965;
  t3060 = t3048*t2997;
  t3062 = t3021 + t3040 + t3060;
  t3239 = t1986*t3219;
  t3294 = t1887*t3279;
  t3317 = t1917*t3312;
  t3321 = t3239 + t3294 + t3317;
  t3334 = t1783*t3219;
  t3340 = t1723*t3279;
  t3342 = t1754*t3312;
  t3353 = t3334 + t3340 + t3342;
  t3711 = t1723*t3608;
  t3716 = t1754*t3642;
  t3723 = t1783*t3687;
  t3734 = t3711 + t3716 + t3723;
  t3745 = t1887*t3608;
  t3750 = t1917*t3642;
  t3751 = t1986*t3687;
  t3764 = t3745 + t3750 + t3751;
  t3979 = t1723*t3909;
  t3990 = t1754*t3940;
  t4006 = t1783*t3962;
  t4009 = t3979 + t3990 + t4006;
  t4013 = t1887*t3909;
  t4035 = t1917*t3940;
  t4036 = t1986*t3962;
  t4044 = t4013 + t4035 + t4036;
  t4086 = t1723*t2884;
  t4092 = t1754*t2965;
  t4094 = t1783*t2997;
  t4109 = t4086 + t4092 + t4094;
  t4119 = t1887*t2884;
  t4123 = t1917*t2965;
  t4139 = t1986*t2997;
  t4153 = t4119 + t4123 + t4139;
  t4291 = t1754*t4239;
  t4299 = t1723*t4263;
  t4303 = t1783*t4274;
  t4341 = t4291 + t4299 + t4303;
  t4350 = t1917*t4239;
  t4352 = t1887*t4263;
  t4354 = t1986*t4274;
  t4355 = t4350 + t4352 + t4354;
  t4487 = t1754*t4426;
  t4499 = t1723*t4449;
  t4502 = t1783*t4460;
  t4503 = t4487 + t4499 + t4502;
  t4540 = t1917*t4426;
  t4541 = t1887*t4449;
  t4547 = t1986*t4460;
  t4552 = t4540 + t4541 + t4547;
  t4571 = -0.5054634410180001*t254*t4570;
  t4592 = t2908*t4576;
  t4604 = t2985*t4600;
  t4612 = t4571 + t4592 + t4604;
  t4618 = t3018*t4570;
  t4619 = -0.9945383682050002*t254*t4576;
  t4621 = t3048*t4600;
  t4626 = t4618 + t4619 + t4621;
  t4678 = t1986*t4675;
  t4688 = t1887*t4686;
  t4711 = t1917*t4708;
  t4727 = t4678 + t4688 + t4711;
  t4730 = t1783*t4675;
  t4735 = t1723*t4686;
  t4740 = t1754*t4708;
  t4747 = t4730 + t4735 + t4740;
  t4891 = t1754*t4826;
  t4893 = t1723*t4840;
  t4896 = t1783*t4864;
  t4898 = t4891 + t4893 + t4896;
  t4906 = t1917*t4826;
  t4911 = t1887*t4840;
  t4932 = t1986*t4864;
  t4934 = t4906 + t4911 + t4932;
  t5029 = t1754*t4966;
  t5044 = t1723*t4973;
  t5047 = t1783*t4982;
  t5051 = t5029 + t5044 + t5047;
  t5067 = t1917*t4966;
  t5077 = t1887*t4973;
  t5088 = t1986*t4982;
  t5092 = t5067 + t5077 + t5088;
  t5143 = -0.5054634410180001*t254*t5133;
  t5173 = t2908*t5163;
  t5210 = t2985*t5198;
  t5218 = t5143 + t5173 + t5210;
  t5228 = t3018*t5133;
  t5230 = -0.9945383682050002*t254*t5163;
  t5237 = t3048*t5198;
  t5275 = t5228 + t5230 + t5237;
  t5369 = t1986*t5364;
  t5397 = t1887*t5392;
  t5429 = t1917*t5411;
  t5450 = t5369 + t5397 + t5429;
  t5454 = t1783*t5364;
  t5471 = t1723*t5392;
  t5487 = t1754*t5411;
  t5490 = t5454 + t5471 + t5487;
  t5731 = t1723*t5651;
  t5754 = t1754*t5676;
  t5755 = t1783*t5696;
  t5765 = t5731 + t5754 + t5755;
  t5780 = t1887*t5651;
  t5787 = t1917*t5676;
  t5788 = t1986*t5696;
  t5789 = t5780 + t5787 + t5788;
  t1230 = t291*t1226;
  t1477 = t1325*t1423;
  t1666 = t1534*t1661;
  t1667 = t1230 + t1477 + t1666;
  t1678 = 0.707107*t1667;
  t2307 = t1325*t2261;
  t2353 = t291*t2343;
  t2395 = t1534*t2393;
  t2422 = t2307 + t2353 + t2395;
  t2445 = 0.707107*t2422;
  t2657 = t1325*t2654;
  t2694 = t291*t2672;
  t2749 = t1534*t2723;
  t2762 = t2657 + t2694 + t2749;
  t2767 = 0.707107*t2762;
  t3122 = t3076*t2884;
  t3155 = t3150*t2965;
  t3167 = -0.500001190325*t254*t2997;
  t3174 = t3122 + t3155 + t3167;
  t3178 = 0.707107*t3174;
  t3372 = t1534*t3219;
  t3381 = t291*t3279;
  t3384 = t1325*t3312;
  t3412 = t3372 + t3381 + t3384;
  t3430 = 0.707107*t3412;
  t3610 = t291*t3608;
  t3651 = t1325*t3642;
  t3698 = t1534*t3687;
  t3699 = t3610 + t3651 + t3698;
  t3708 = 0.707107*t3699;
  t3912 = t291*t3909;
  t3942 = t1325*t3940;
  t3967 = t1534*t3962;
  t3972 = t3912 + t3942 + t3967;
  t3977 = 0.707107*t3972;
  t4065 = t291*t2884;
  t4069 = t1325*t2965;
  t4070 = t1534*t2997;
  t4077 = t4065 + t4069 + t4070;
  t4082 = 0.707107*t4077;
  t4242 = t1325*t4239;
  t4268 = t291*t4263;
  t4277 = t1534*t4274;
  t4281 = t4242 + t4268 + t4277;
  t4283 = 0.707107*t4281;
  t4435 = t1325*t4426;
  t4452 = t291*t4449;
  t4463 = t1534*t4460;
  t4470 = t4435 + t4452 + t4463;
  t4478 = 0.707107*t4470;
  t4631 = t3076*t4570;
  t4637 = t3150*t4576;
  t4642 = -0.500001190325*t254*t4600;
  t4644 = t4631 + t4637 + t4642;
  t4665 = 0.707107*t4644;
  t4756 = t1534*t4675;
  t4759 = t291*t4686;
  t4763 = t1325*t4708;
  t4767 = t4756 + t4759 + t4763;
  t4770 = 0.707107*t4767;
  t4830 = t1325*t4826;
  t4845 = t291*t4840;
  t4866 = t1534*t4864;
  t4877 = t4830 + t4845 + t4866;
  t4878 = 0.707107*t4877;
  t4967 = t1325*t4966;
  t4975 = t291*t4973;
  t4987 = t1534*t4982;
  t5005 = t4967 + t4975 + t4987;
  t5010 = 0.707107*t5005;
  t5285 = t3076*t5133;
  t5291 = t3150*t5163;
  t5297 = -0.500001190325*t254*t5198;
  t5303 = t5285 + t5291 + t5297;
  t5320 = 0.707107*t5303;
  t5510 = t1534*t5364;
  t5528 = t291*t5392;
  t5540 = t1325*t5411;
  t5548 = t5510 + t5528 + t5540;
  t5551 = 0.707107*t5548;
  t5656 = t291*t5651;
  t5679 = t1325*t5676;
  t5700 = t1534*t5696;
  t5716 = t5656 + t5679 + t5700;
  t5717 = 0.707107*t5716;
  t6582 = -1.*t908;
  t6587 = 1. + t6582;
  t6720 = -0.051978134642000004*t251;
  t6794 = 0.05226439969100001*t251;
  t6743 = 0.49726168403800003*t251;
  t6646 = -0.12*t6587;
  t6648 = -0.4*t788;
  t6650 = 0. + t6646 + t6648;
  t6599 = 0.4*t6587;
  t6600 = -0.12*t788;
  t6606 = 0. + t6599 + t6600;
  t6681 = -1.1924972351948546e-8*var1[25];
  t6684 = 0.38315655000705834*t669;
  t6687 = -0.05650052807*t518;
  t6689 = t6681 + t6684 + t6687;
  t6652 = 1.1345904784751044e-7*var1[25];
  t6660 = 0.04027119345689465*t669;
  t6671 = -0.05650052807*t1335;
  t6678 = t6652 + t6660 + t6671;
  t6783 = 1.639789470231751e-8*var1[26];
  t6787 = -0.22983603018311177*t251;
  t6796 = t6794 + t1930;
  t6800 = 0.164383620275*t6796;
  t6816 = t6720 + t1850;
  t6817 = 0.18957839082800002*t6816;
  t6818 = t6783 + t6787 + t6800 + t6817;
  t6696 = 1.5601527583902087e-7*var1[26];
  t6705 = 0.09582494577057615*t251;
  t6739 = t6720 + t1740;
  t6742 = -0.231098203479*t6739;
  t6746 = t6743 + t1781;
  t6748 = 0.164383620275*t6746;
  t6758 = t6696 + t6705 + t6742 + t6748;
  t6832 = -1.568745163810375e-7*var1[26];
  t6833 = 0.08219200580743281*t251;
  t6840 = t6794 + t1306;
  t6841 = -0.231098203479*t6840;
  t6842 = t6743 + t288;
  t6843 = 0.18957839082800002*t6842;
  t6844 = t6832 + t6833 + t6841 + t6843;
  t6967 = -0.051978134642000004*t254;
  t7045 = 0.05226439969100001*t254;
  t7014 = 0.49726168403800003*t254;
  t6618 = -0.056500534356700764*t669;
  t6621 = 0.040271188976*t1592;
  t6622 = 0.38315650737400003*t1632;
  t6637 = 0. + t6618 + t6621 + t6622;
  t6903 = -0.12*t908;
  t6911 = 0.4*t788;
  t6912 = t6903 + t6911;
  t6917 = -0.4*t908;
  t6922 = t6917 + t6600;
  t6976 = t2904 + t6967;
  t7009 = -0.231098203479*t6976;
  t7016 = t2973 + t7014;
  t7018 = 0.164383620275*t7016;
  t7022 = 0.09582494577057615*t254;
  t7023 = 1.5601527583902087e-7 + t7009 + t7018 + t7022;
  t7035 = t3017 + t6967;
  t7040 = 0.18957839082800002*t7035;
  t7051 = t3041 + t7045;
  t7054 = 0.164383620275*t7051;
  t7055 = -0.22983603018311177*t254;
  t7057 = 1.639789470231751e-8 + t7040 + t7054 + t7055;
  t7071 = t3123 + t7045;
  t7072 = -0.231098203479*t7071;
  t7073 = t3073 + t7014;
  t7077 = 0.18957839082800002*t7073;
  t7078 = 0.08219200580743281*t254;
  t7079 = -1.568745163810375e-7 + t7072 + t7077 + t7078;
  t7114 = 0.3852670428678886*t565;
  t7118 = -0.056500534356700764*t433;
  t7119 = t7114 + t7118;
  t7121 = 0.0059058871981009595*t565;
  t7122 = 0.04027119345689465*t433;
  t7124 = 1.1345904784751044e-7 + t7121 + t7122;
  t7126 = 0.05619101817723254*t565;
  t7135 = 0.38315655000705834*t433;
  t7138 = -1.1924972351948546e-8 + t7126 + t7135;
  p_output1[0]=(t3708 + 0.703234*t3734 + 0.073913*t3764)*var2[3] + (t1678 + 0.703234*t1837 + 0.073913*t2033)*var2[4] + (t2445 + 0.703234*t2459 + 0.073913*t2560)*var2[5] + (t2767 + 0.703234*t2779 + 0.073913*t2826)*var2[24] + (0.073913*t3321 + 0.703234*t3353 + t3430)*var2[25] + (0.703234*t3001 + 0.073913*t3062 + t3178)*var2[26];
  p_output1[1]=(t4082 + 0.703234*t4109 + 0.073913*t4153)*var2[3] + (t3977 + 0.703234*t4009 + 0.073913*t4044)*var2[4] + (t4283 + 0.703234*t4341 + 0.073913*t4355)*var2[5] + (t4478 + 0.703234*t4503 + 0.073913*t4552)*var2[24] + (0.073913*t4727 + 0.703234*t4747 + t4770)*var2[25] + (0.703234*t4612 + 0.073913*t4626 + t4665)*var2[26];
  p_output1[2]=(t5717 + 0.703234*t5765 + 0.073913*t5789)*var2[4] + (t5010 + 0.703234*t5051 + 0.073913*t5092)*var2[5] + (t4878 + 0.703234*t4898 + 0.073913*t4934)*var2[24] + (0.073913*t5450 + 0.703234*t5490 + t5551)*var2[25] + (0.703234*t5218 + 0.073913*t5275 + t5320)*var2[26];
  p_output1[3]=0;
  p_output1[4]=(0.104528*t3734 - 0.994522*t3764)*var2[3] + (0.104528*t1837 - 0.994522*t2033)*var2[4] + (0.104528*t2459 - 0.994522*t2560)*var2[5] + (0.104528*t2779 - 0.994522*t2826)*var2[24] + (-0.994522*t3321 + 0.104528*t3353)*var2[25] + (0.104528*t3001 - 0.994522*t3062)*var2[26];
  p_output1[5]=(0.104528*t4109 - 0.994522*t4153)*var2[3] + (0.104528*t4009 - 0.994522*t4044)*var2[4] + (0.104528*t4341 - 0.994522*t4355)*var2[5] + (0.104528*t4503 - 0.994522*t4552)*var2[24] + (-0.994522*t4727 + 0.104528*t4747)*var2[25] + (0.104528*t4612 - 0.994522*t4626)*var2[26];
  p_output1[6]=(0.104528*t5765 - 0.994522*t5789)*var2[4] + (0.104528*t5051 - 0.994522*t5092)*var2[5] + (0.104528*t4898 - 0.994522*t4934)*var2[24] + (-0.994522*t5450 + 0.104528*t5490)*var2[25] + (0.104528*t5218 - 0.994522*t5275)*var2[26];
  p_output1[7]=0;
  p_output1[8]=(t3708 - 0.703234*t3734 - 0.073913*t3764)*var2[3] + (t1678 - 0.703234*t1837 - 0.073913*t2033)*var2[4] + (t2445 - 0.703234*t2459 - 0.073913*t2560)*var2[5] + (t2767 - 0.703234*t2779 - 0.073913*t2826)*var2[24] + (-0.073913*t3321 - 0.703234*t3353 + t3430)*var2[25] + (-0.703234*t3001 - 0.073913*t3062 + t3178)*var2[26];
  p_output1[9]=(t4082 - 0.703234*t4109 - 0.073913*t4153)*var2[3] + (t3977 - 0.703234*t4009 - 0.073913*t4044)*var2[4] + (t4283 - 0.703234*t4341 - 0.073913*t4355)*var2[5] + (t4478 - 0.703234*t4503 - 0.073913*t4552)*var2[24] + (-0.073913*t4727 - 0.703234*t4747 + t4770)*var2[25] + (-0.703234*t4612 - 0.073913*t4626 + t4665)*var2[26];
  p_output1[10]=(t5717 - 0.703234*t5765 - 0.073913*t5789)*var2[4] + (t5010 - 0.703234*t5051 - 0.073913*t5092)*var2[5] + (t4878 - 0.703234*t4898 - 0.073913*t4934)*var2[24] + (-0.073913*t5450 - 0.703234*t5490 + t5551)*var2[25] + (-0.703234*t5218 - 0.073913*t5275 + t5320)*var2[26];
  p_output1[11]=0;
  p_output1[12]=var2[0] + (0.060173*t3699 + 0.293218*t3734 - 0.220205*t3764 + t3531*t6606 + t3541*t6650 + t3576*t6678 + t3603*t6689 + t3608*t6758 - 1.*t2093*t6637*t681 + t3642*t6818 + t3687*t6844)*var2[3] + (0.060173*t1667 + 0.293218*t1837 - 0.220205*t2033 - 1.*t380*t528*t6637 + t1092*t6678 + t1218*t6689 + t1226*t6758 + t1082*t380*t6650*t681 + t1423*t6818 + t1661*t6844 + t380*t6606*t681*t783)*var2[4] + (0.060173*t2422 + 0.293218*t2459 - 0.220205*t2560 + t2172*t6606 + t2158*t6650 + t2251*t6678 + t2211*t6689 + t2343*t6758 + t2261*t6818 + t2393*t6844)*var2[5] + (0.060173*t2762 + 0.293218*t2779 - 0.220205*t2826 + t2648*t6678 + t2623*t6689 + t2672*t6758 + t2654*t6818 + t2723*t6844 + t2158*t6912 + t2616*t6922)*var2[24] + (-0.220205*t3321 + 0.293218*t3353 + 0.060173*t3412 + t3279*t6758 + t3312*t6818 + t3219*t6844 + t380*t681*t7119 + t2623*t7124 + t2871*t7138)*var2[25] + (0.293218*t3001 - 0.220205*t3062 + 0.060173*t3174 + t2884*t7023 + t2965*t7057 + t2997*t7079)*var2[26];
  p_output1[13]=var2[1] + (0.060173*t4077 + 0.293218*t4109 - 0.220205*t4153 + t2158*t6606 + t2616*t6650 + t2623*t6678 + t2871*t6689 + t2884*t6758 + t380*t6637*t681 + t2965*t6818 + t2997*t6844)*var2[3] + (0.060173*t3972 + 0.293218*t4009 - 0.220205*t4044 - 1.*t2093*t528*t6637 + t3848*t6678 + t3895*t6689 + t3909*t6758 + t1082*t2093*t6650*t681 + t3940*t6818 + t3962*t6844 + t2093*t6606*t681*t783)*var2[4] + (0.060173*t4281 + 0.293218*t4341 - 0.220205*t4355 + t3541*t6606 + t4189*t6650 + t4230*t6678 + t4197*t6689 + t4263*t6758 + t4239*t6818 + t4274*t6844)*var2[5] + (0.060173*t4470 + 0.293218*t4503 - 0.220205*t4552 + t4415*t6678 + t4395*t6689 + t4449*t6758 + t4426*t6818 + t4460*t6844 + t4189*t6912 + t4387*t6922)*var2[24] + (-0.220205*t4727 + 0.293218*t4747 + 0.060173*t4767 + t4686*t6758 + t4708*t6818 + t4675*t6844 + t2093*t681*t7119 + t4395*t7124 + t4568*t7138)*var2[25] + (0.293218*t4612 - 0.220205*t4626 + 0.060173*t4644 + t4570*t7023 + t4576*t7057 + t4600*t7079)*var2[26];
  p_output1[14]=var2[2] + (0.060173*t5716 + 0.293218*t5765 - 0.220205*t5789 - 1.*t1082*t528*t6650 + t5622*t6678 + t5645*t6689 + t5651*t6758 - 1.*t6637*t681 + t5676*t6818 + t5696*t6844 - 1.*t528*t6606*t783)*var2[4] + (0.060173*t5005 + 0.293218*t5051 - 0.220205*t5092 + t4957*t6678 + t4950*t6689 + t4973*t6758 - 1.*t1082*t6606*t681 + t4966*t6818 + t4982*t6844 + t6650*t681*t783)*var2[5] + (0.060173*t4877 + 0.293218*t4898 - 0.220205*t4934 + t4816*t6678 + t4791*t6689 + t4840*t6758 + t4826*t6818 + t4864*t6844 + t1082*t681*t6922 + t681*t6912*t783)*var2[24] + (-0.220205*t5450 + 0.293218*t5490 + 0.060173*t5548 + t5392*t6758 + t5411*t6818 + t5364*t6844 - 1.*t528*t7119 + t4791*t7124 + t4957*t7138)*var2[25] + (0.293218*t5218 - 0.220205*t5275 + 0.060173*t5303 + t5133*t7023 + t5163*t7057 + t5198*t7079)*var2[26];
  p_output1[15]=0;
}



void dT_shoulder_yaw_joint_right_src(double *p_output1, const double *var1,const double *var2)
{
  // Call Subroutines
  output1(p_output1, var1, var2);

}
