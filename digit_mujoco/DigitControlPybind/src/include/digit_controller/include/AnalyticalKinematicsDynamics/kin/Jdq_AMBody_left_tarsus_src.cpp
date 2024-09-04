/*
 * Automatically Generated from Mathematica.
 * Mon 4 Jul 2022 20:57:49 GMT-04:00
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Jdq_AMBody_left_tarsus_src.h"

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
  double t1372;
  double t1383;
  double t1422;
  double t1113;
  double t1146;
  double t1343;
  double t1859;
  double t841;
  double t993;
  double t1087;
  double t1545;
  double t1628;
  double t1633;
  double t1636;
  double t1645;
  double t1678;
  double t1692;
  double t1723;
  double t1730;
  double t1746;
  double t1840;
  double t1885;
  double t1922;
  double t2030;
  double t2032;
  double t2040;
  double t2050;
  double t2066;
  double t364;
  double t598;
  double t805;
  double t1650;
  double t1656;
  double t1671;
  double t1784;
  double t1844;
  double t1852;
  double t2072;
  double t2073;
  double t2093;
  double t2114;
  double t2119;
  double t2125;
  double t2131;
  double t2180;
  double t2191;
  double t2192;
  double t2204;
  double t2209;
  double t2215;
  double t2226;
  double t2253;
  double t2276;
  double t2370;
  double t2376;
  double t2381;
  double t2388;
  double t2397;
  double t2083;
  double t2089;
  double t2091;
  double t2193;
  double t2421;
  double t2428;
  double t2477;
  double t2481;
  double t2610;
  double t2611;
  double t2656;
  double t2663;
  double t2703;
  double t2704;
  double t2723;
  double t2744;
  double t2781;
  double t2874;
  double t2895;
  double t2933;
  double t2934;
  double t2946;
  double t2957;
  double t2978;
  double t3005;
  double t3037;
  double t3077;
  double t3206;
  double t2514;
  double t2550;
  double t2589;
  double t2778;
  double t2792;
  double t2865;
  double t3098;
  double t3105;
  double t181;
  double t3224;
  double t3234;
  double t3252;
  double t3291;
  double t3318;
  double t3362;
  double t3416;
  double t3422;
  double t17;
  double t127;
  double t3689;
  double t3692;
  double t3698;
  double t3699;
  double t3705;
  double t3707;
  double t3716;
  double t3720;
  double t3723;
  double t3729;
  double t3733;
  double t3740;
  double t3764;
  double t3770;
  double t3782;
  double t3784;
  double t3788;
  double t3791;
  double t3708;
  double t3762;
  double t3816;
  double t3862;
  double t3877;
  double t3884;
  double t3909;
  double t3942;
  double t3964;
  double t3972;
  double t3982;
  double t4010;
  double t3866;
  double t3961;
  double t4020;
  double t4022;
  double t4026;
  double t4034;
  double t4039;
  double t4048;
  double t4052;
  double t4065;
  double t4066;
  double t4073;
  double t3499;
  double t4025;
  double t4051;
  double t4108;
  double t4109;
  double t4126;
  double t4132;
  double t4133;
  double t4134;
  double t3610;
  double t3622;
  double t3624;
  double t3627;
  double t3629;
  double t3642;
  double t3647;
  double t4207;
  double t4213;
  double t4219;
  double t4222;
  double t4233;
  double t4238;
  double t4249;
  double t4283;
  double t4309;
  double t4333;
  double t4359;
  double t4362;
  double t4229;
  double t4292;
  double t4363;
  double t4372;
  double t4376;
  double t4377;
  double t4381;
  double t4385;
  double t4397;
  double t4437;
  double t4442;
  double t4448;
  double t4374;
  double t4388;
  double t4457;
  double t4460;
  double t4475;
  double t4481;
  double t4484;
  double t4490;
  double t4500;
  double t4508;
  double t4511;
  double t4515;
  double t4471;
  double t4492;
  double t4521;
  double t4523;
  double t4545;
  double t4550;
  double t4555;
  double t4562;
  double t3177;
  double t3427;
  double t3433;
  double t3478;
  double t3513;
  double t3517;
  double t3531;
  double t3578;
  double t3580;
  double t3592;
  double t3625;
  double t3632;
  double t3655;
  double t3659;
  double t3660;
  double t3683;
  double t4117;
  double t4138;
  double t4141;
  double t4146;
  double t4148;
  double t4152;
  double t4153;
  double t4160;
  double t4165;
  double t4172;
  double t4173;
  double t4176;
  double t4184;
  double t4191;
  double t4199;
  double t4200;
  double t4540;
  double t4563;
  double t4564;
  double t4566;
  double t4573;
  double t4606;
  double t4608;
  double t4609;
  double t4610;
  double t4614;
  double t4616;
  double t4631;
  double t4634;
  double t4635;
  double t4636;
  double t4656;
  double t4769;
  double t4774;
  double t4775;
  double t4777;
  double t4778;
  double t4788;
  double t4797;
  double t4805;
  double t4806;
  double t4879;
  double t4881;
  double t4896;
  double t4897;
  double t4902;
  double t4904;
  double t4905;
  double t4907;
  double t4912;
  double t4913;
  double t4915;
  double t4916;
  double t4983;
  double t4984;
  double t4987;
  double t4994;
  double t4995;
  double t4999;
  double t5007;
  double t5008;
  double t5018;
  double t5051;
  double t5052;
  double t5053;
  double t5086;
  double t5087;
  double t5091;
  double t5098;
  double t5101;
  double t5108;
  double t5144;
  double t5149;
  double t5153;
  double t5156;
  double t5164;
  double t5166;
  double t5176;
  double t5179;
  double t5180;
  t1372 = Cos(var1[10]);
  t1383 = -1.*t1372;
  t1422 = 1. + t1383;
  t1113 = Cos(var1[9]);
  t1146 = -1.*t1113;
  t1343 = 1. + t1146;
  t1859 = Sin(var1[10]);
  t841 = Cos(var1[8]);
  t993 = -1.*t841;
  t1087 = 1. + t993;
  t1545 = -0.8656776547239999*t1422;
  t1628 = 1. + t1545;
  t1633 = -0.366501*t1628;
  t1636 = -0.3172717261340007*t1422;
  t1645 = t1633 + t1636;
  t1678 = -0.134322983001*t1422;
  t1692 = 1. + t1678;
  t1723 = -0.930418*t1692;
  t1730 = -0.12497652119782442*t1422;
  t1746 = t1723 + t1730;
  t1840 = Sin(var1[9]);
  t1885 = -0.930418*t1859;
  t1922 = 0. + t1885;
  t2030 = -0.366501*t1922;
  t2032 = 0.366501*t1859;
  t2040 = 0. + t2032;
  t2050 = -0.930418*t2040;
  t2066 = t2030 + t2050;
  t364 = Cos(var1[7]);
  t598 = -1.*t364;
  t805 = 1. + t598;
  t1650 = 0.340999127418*t1343*t1645;
  t1656 = -0.134322983001*t1343;
  t1671 = 1. + t1656;
  t1784 = t1671*t1746;
  t1844 = -0.366501*t1840;
  t1852 = 0. + t1844;
  t2072 = t1852*t2066;
  t2073 = 0. + t1650 + t1784 + t2072;
  t2093 = -0.8656776547239999*t1343;
  t2114 = 1. + t2093;
  t2119 = t2114*t1645;
  t2125 = 0.340999127418*t1343*t1746;
  t2131 = 0.930418*t1840;
  t2180 = 0. + t2131;
  t2191 = t2180*t2066;
  t2192 = 0. + t2119 + t2125 + t2191;
  t2204 = -0.930418*t1840;
  t2209 = 0. + t2204;
  t2215 = t1645*t2209;
  t2226 = 0.366501*t1840;
  t2253 = 0. + t2226;
  t2276 = t1746*t2253;
  t2370 = -1.000000637725*t1343;
  t2376 = 1. + t2370;
  t2381 = t2376*t2066;
  t2388 = 0. + t2215 + t2276 + t2381;
  t2397 = Sin(var1[8]);
  t2083 = 0.340999127418*t1087*t2073;
  t2089 = -0.8656776547239999*t1087;
  t2091 = 1. + t2089;
  t2193 = t2091*t2192;
  t2421 = -0.930418*t2397;
  t2428 = 0. + t2421;
  t2477 = t2388*t2428;
  t2481 = 0. + t2083 + t2193 + t2477;
  t2610 = -0.134322983001*t1087;
  t2611 = 1. + t2610;
  t2656 = t2611*t2073;
  t2663 = 0.340999127418*t1087*t2192;
  t2703 = 0.366501*t2397;
  t2704 = 0. + t2703;
  t2723 = t2388*t2704;
  t2744 = 0. + t2656 + t2663 + t2723;
  t2781 = Sin(var1[7]);
  t2874 = -1.000000637725*t1087;
  t2895 = 1. + t2874;
  t2933 = t2895*t2388;
  t2934 = -0.366501*t2397;
  t2946 = 0. + t2934;
  t2957 = t2073*t2946;
  t2978 = 0.930418*t2397;
  t3005 = 0. + t2978;
  t3037 = t2192*t3005;
  t3077 = 0. + t2933 + t2957 + t3037;
  t3206 = Cos(var1[6]);
  t2514 = -0.340999127418*t805*t2481;
  t2550 = -0.8656776547239999*t805;
  t2589 = 1. + t2550;
  t2778 = t2589*t2744;
  t2792 = -0.930418*t2781;
  t2865 = 0. + t2792;
  t3098 = t2865*t3077;
  t3105 = 0. + t2514 + t2778 + t3098;
  t181 = Sin(var1[6]);
  t3224 = -0.134322983001*t805;
  t3234 = 1. + t3224;
  t3252 = t3234*t2481;
  t3291 = -0.340999127418*t805*t2744;
  t3318 = -0.366501*t2781;
  t3362 = 0. + t3318;
  t3416 = t3362*t3077;
  t3422 = 0. + t3252 + t3291 + t3416;
  t17 = Cos(var1[4]);
  t127 = Cos(var1[5]);
  t3689 = -0.310811*t1692;
  t3692 = 0.2690616104987713*t1422;
  t3698 = -0.366501*t1859;
  t3699 = 0. + t3698;
  t3705 = 0.529919*t3699;
  t3707 = t3689 + t3692 + t3705;
  t3716 = -1.000000637725*t1422;
  t3720 = 1. + t3716;
  t3723 = 0.529919*t3720;
  t3729 = 0.789039*t1922;
  t3733 = -0.310811*t2040;
  t3740 = t3723 + t3729 + t3733;
  t3764 = 0.789039*t1628;
  t3770 = -0.105986279791916*t1422;
  t3782 = 0.930418*t1859;
  t3784 = 0. + t3782;
  t3788 = 0.529919*t3784;
  t3791 = t3764 + t3770 + t3788;
  t3708 = 0.340999127418*t1343*t3707;
  t3762 = t2180*t3740;
  t3816 = t2114*t3791;
  t3862 = 0. + t3708 + t3762 + t3816;
  t3877 = t1671*t3707;
  t3884 = t1852*t3740;
  t3909 = 0.340999127418*t1343*t3791;
  t3942 = 0. + t3877 + t3884 + t3909;
  t3964 = t2253*t3707;
  t3972 = t2376*t3740;
  t3982 = t2209*t3791;
  t4010 = 0. + t3964 + t3972 + t3982;
  t3866 = t2091*t3862;
  t3961 = 0.340999127418*t1087*t3942;
  t4020 = t4010*t2428;
  t4022 = 0. + t3866 + t3961 + t4020;
  t4026 = 0.340999127418*t1087*t3862;
  t4034 = t2611*t3942;
  t4039 = t4010*t2704;
  t4048 = 0. + t4026 + t4034 + t4039;
  t4052 = t2895*t4010;
  t4065 = t3942*t2946;
  t4066 = t3862*t3005;
  t4073 = 0. + t4052 + t4065 + t4066;
  t3499 = Sin(var1[5]);
  t4025 = -0.340999127418*t805*t4022;
  t4051 = t2589*t4048;
  t4108 = t2865*t4073;
  t4109 = 0. + t4025 + t4051 + t4108;
  t4126 = t3234*t4022;
  t4132 = -0.340999127418*t805*t4048;
  t4133 = t3362*t4073;
  t4134 = 0. + t4126 + t4132 + t4133;
  t3610 = Sin(var1[4]);
  t3622 = 0.366501*t2781;
  t3624 = 0. + t3622;
  t3627 = 0.930418*t2781;
  t3629 = 0. + t3627;
  t3642 = -1.000000637725*t805;
  t3647 = 1. + t3642;
  t4207 = 0.194216*t1692;
  t4213 = -0.16812859677606265*t1422;
  t4219 = 0.848048*t3699;
  t4222 = t4207 + t4213 + t4219;
  t4233 = 0.848048*t3720;
  t4238 = -0.493047*t1922;
  t4249 = 0.194216*t2040;
  t4283 = t4233 + t4238 + t4249;
  t4309 = -0.493047*t1628;
  t4333 = 0.06622748653061429*t1422;
  t4359 = 0.848048*t3784;
  t4362 = t4309 + t4333 + t4359;
  t4229 = 0.340999127418*t1343*t4222;
  t4292 = t2180*t4283;
  t4363 = t2114*t4362;
  t4372 = 0. + t4229 + t4292 + t4363;
  t4376 = t1671*t4222;
  t4377 = t1852*t4283;
  t4381 = 0.340999127418*t1343*t4362;
  t4385 = 0. + t4376 + t4377 + t4381;
  t4397 = t2253*t4222;
  t4437 = t2376*t4283;
  t4442 = t2209*t4362;
  t4448 = 0. + t4397 + t4437 + t4442;
  t4374 = t2091*t4372;
  t4388 = 0.340999127418*t1087*t4385;
  t4457 = t4448*t2428;
  t4460 = 0. + t4374 + t4388 + t4457;
  t4475 = 0.340999127418*t1087*t4372;
  t4481 = t2611*t4385;
  t4484 = t4448*t2704;
  t4490 = 0. + t4475 + t4481 + t4484;
  t4500 = t2895*t4448;
  t4508 = t4385*t2946;
  t4511 = t4372*t3005;
  t4515 = 0. + t4500 + t4508 + t4511;
  t4471 = -0.340999127418*t805*t4460;
  t4492 = t2589*t4490;
  t4521 = t2865*t4515;
  t4523 = 0. + t4471 + t4492 + t4521;
  t4545 = t3234*t4460;
  t4550 = -0.340999127418*t805*t4490;
  t4555 = t3362*t4515;
  t4562 = 0. + t4545 + t4550 + t4555;
  t3177 = -1.*t181*t3105;
  t3427 = t3206*t3422;
  t3433 = 0. + t3177 + t3427;
  t3478 = t127*t3433;
  t3513 = t3206*t3105;
  t3517 = t181*t3422;
  t3531 = 0. + t3513 + t3517;
  t3578 = t3499*t3531;
  t3580 = 0. + t3478 + t3578;
  t3592 = t17*t3580;
  t3625 = t3624*t2481;
  t3632 = t3629*t2744;
  t3655 = t3647*t3077;
  t3659 = 0. + t3625 + t3632 + t3655;
  t3660 = -1.*t3610*t3659;
  t3683 = 0. + t3592 + t3660;
  t4117 = -1.*t181*t4109;
  t4138 = t3206*t4134;
  t4141 = 0. + t4117 + t4138;
  t4146 = t127*t4141;
  t4148 = t3206*t4109;
  t4152 = t181*t4134;
  t4153 = 0. + t4148 + t4152;
  t4160 = t3499*t4153;
  t4165 = 0. + t4146 + t4160;
  t4172 = t17*t4165;
  t4173 = t3624*t4022;
  t4176 = t3629*t4048;
  t4184 = t3647*t4073;
  t4191 = 0. + t4173 + t4176 + t4184;
  t4199 = -1.*t3610*t4191;
  t4200 = 0. + t4172 + t4199;
  t4540 = -1.*t181*t4523;
  t4563 = t3206*t4562;
  t4564 = 0. + t4540 + t4563;
  t4566 = t127*t4564;
  t4573 = t3206*t4523;
  t4606 = t181*t4562;
  t4608 = 0. + t4573 + t4606;
  t4609 = t3499*t4608;
  t4610 = 0. + t4566 + t4609;
  t4614 = t17*t4610;
  t4616 = t3624*t4460;
  t4631 = t3629*t4490;
  t4634 = t3647*t4515;
  t4635 = 0. + t4616 + t4631 + t4634;
  t4636 = -1.*t3610*t4635;
  t4656 = 0. + t4614 + t4636;
  t4769 = -1.*t3499*t3433;
  t4774 = t127*t3531;
  t4775 = 0. + t4769 + t4774;
  t4777 = -1.*t3499*t4141;
  t4778 = t127*t4153;
  t4788 = 0. + t4777 + t4778;
  t4797 = -1.*t3499*t4564;
  t4805 = t127*t4608;
  t4806 = 0. + t4797 + t4805;
  t4879 = -1.*t3624*t2481;
  t4881 = -1.*t3629*t2744;
  t4896 = -1.*t3647*t3077;
  t4897 = 0. + t4879 + t4881 + t4896;
  t4902 = -1.*t3624*t4022;
  t4904 = -1.*t3629*t4048;
  t4905 = -1.*t3647*t4073;
  t4907 = 0. + t4902 + t4904 + t4905;
  t4912 = -1.*t3624*t4460;
  t4913 = -1.*t3629*t4490;
  t4915 = -1.*t3647*t4515;
  t4916 = 0. + t4912 + t4913 + t4915;
  t4983 = -0.930418*t2481;
  t4984 = 0.366501*t2744;
  t4987 = 0. + t4983 + t4984;
  t4994 = -0.930418*t4022;
  t4995 = 0.366501*t4048;
  t4999 = 0. + t4994 + t4995;
  t5007 = -0.930418*t4460;
  t5008 = 0.366501*t4490;
  t5018 = 0. + t5007 + t5008;
  t5051 = 0.930418*t2073;
  t5052 = 0.366501*t2192;
  t5053 = 0. + t5051 + t5052;
  t5086 = 0.366501*t3862;
  t5087 = 0.930418*t3942;
  t5091 = 0. + t5086 + t5087;
  t5098 = 0.366501*t4372;
  t5101 = 0.930418*t4385;
  t5108 = 0. + t5098 + t5101;
  t5144 = -0.366501*t1645;
  t5149 = -0.930418*t1746;
  t5153 = 0. + t5144 + t5149;
  t5156 = -0.930418*t3707;
  t5164 = -0.366501*t3791;
  t5166 = 0. + t5156 + t5164;
  t5176 = -0.930418*t4222;
  t5179 = -0.366501*t4362;
  t5180 = 0. + t5176 + t5179;
  p_output1[0]=0;
  p_output1[1]=0;
  p_output1[2]=0;
  p_output1[3]=0;
  p_output1[4]=0;
  p_output1[5]=0;
  p_output1[6]=0;
  p_output1[7]=0;
  p_output1[8]=0;
  p_output1[9]=0.0001*t3683 + 0.00061*t4200 + 0.00093*t4656;
  p_output1[10]=0.00001*t3683 + 0.01641*t4200 + 0.00061*t4656;
  p_output1[11]=0.0165*t3683 + 0.00001*t4200 + 0.0001*t4656;
  p_output1[12]=0.0001*t4775 + 0.00061*t4788 + 0.00093*t4806;
  p_output1[13]=0.00001*t4775 + 0.01641*t4788 + 0.00061*t4806;
  p_output1[14]=0.0165*t4775 + 0.00001*t4788 + 0.0001*t4806;
  p_output1[15]=0.0001*t3659 + 0.00061*t4191 + 0.00093*t4635;
  p_output1[16]=0.00001*t3659 + 0.01641*t4191 + 0.00061*t4635;
  p_output1[17]=0.0165*t3659 + 0.00001*t4191 + 0.0001*t4635;
  p_output1[18]=0.0001*t4897 + 0.00061*t4907 + 0.00093*t4916;
  p_output1[19]=0.00001*t4897 + 0.01641*t4907 + 0.00061*t4916;
  p_output1[20]=0.0165*t4897 + 0.00001*t4907 + 0.0001*t4916;
  p_output1[21]=0.0001*t4987 + 0.00061*t4999 + 0.00093*t5018;
  p_output1[22]=0.00001*t4987 + 0.01641*t4999 + 0.00061*t5018;
  p_output1[23]=0.0165*t4987 + 0.00001*t4999 + 0.0001*t5018;
  p_output1[24]=0.0001*t5053 + 0.00061*t5091 + 0.00093*t5108;
  p_output1[25]=0.00001*t5053 + 0.01641*t5091 + 0.00061*t5108;
  p_output1[26]=0.0165*t5053 + 0.00001*t5091 + 0.0001*t5108;
  p_output1[27]=0.0001*t5153 + 0.00061*t5166 + 0.00093*t5180;
  p_output1[28]=0.00001*t5153 + 0.01641*t5166 + 0.00061*t5180;
  p_output1[29]=0.0165*t5153 + 0.00001*t5166 + 0.0001*t5180;
  p_output1[30]=0.00010000055463336002;
  p_output1[31]=0.00001000939728742977;
  p_output1[32]=0.01650001054375299;
  p_output1[33]=0;
  p_output1[34]=0;
  p_output1[35]=0;
  p_output1[36]=0;
  p_output1[37]=0;
  p_output1[38]=0;
  p_output1[39]=0;
  p_output1[40]=0;
  p_output1[41]=0;
  p_output1[42]=0;
  p_output1[43]=0;
  p_output1[44]=0;
  p_output1[45]=0;
  p_output1[46]=0;
  p_output1[47]=0;
  p_output1[48]=0;
  p_output1[49]=0;
  p_output1[50]=0;
  p_output1[51]=0;
  p_output1[52]=0;
  p_output1[53]=0;
  p_output1[54]=0;
  p_output1[55]=0;
  p_output1[56]=0;
  p_output1[57]=0;
  p_output1[58]=0;
  p_output1[59]=0;
  p_output1[60]=0;
  p_output1[61]=0;
  p_output1[62]=0;
  p_output1[63]=0;
  p_output1[64]=0;
  p_output1[65]=0;
  p_output1[66]=0;
  p_output1[67]=0;
  p_output1[68]=0;
  p_output1[69]=0;
  p_output1[70]=0;
  p_output1[71]=0;
  p_output1[72]=0;
  p_output1[73]=0;
  p_output1[74]=0;
  p_output1[75]=0;
  p_output1[76]=0;
  p_output1[77]=0;
  p_output1[78]=0;
  p_output1[79]=0;
  p_output1[80]=0;
  p_output1[81]=0;
  p_output1[82]=0;
  p_output1[83]=0;
}



void Jdq_AMBody_left_tarsus_src(double *p_output1, const double *var1,const double *var2)
{
  // Call Subroutines
  output1(p_output1, var1, var2);

}
