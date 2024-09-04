/*
 * Automatically Generated from Mathematica.
 * Sun 16 Oct 2022 21:35:14 GMT-04:00
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Jb_LeftToeBottomBack_src.h"

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
  double t423;
  double t568;
  double t570;
  double t995;
  double t374;
  double t376;
  double t415;
  double t844;
  double t283;
  double t614;
  double t727;
  double t760;
  double t1090;
  double t1094;
  double t1163;
  double t1189;
  double t1192;
  double t1373;
  double t1460;
  double t1471;
  double t1506;
  double t1509;
  double t1527;
  double t1546;
  double t1561;
  double t1563;
  double t1703;
  double t1718;
  double t1727;
  double t1732;
  double t1734;
  double t1790;
  double t1823;
  double t2334;
  double t2341;
  double t2357;
  double t1190;
  double t1238;
  double t1307;
  double t1665;
  double t1684;
  double t1694;
  double t1840;
  double t1885;
  double t2022;
  double t2057;
  double t2058;
  double t2086;
  double t2098;
  double t2106;
  double t2127;
  double t2241;
  double t2376;
  double t2390;
  double t2410;
  double t2436;
  double t2466;
  double t2487;
  double t2500;
  double t2547;
  double t2568;
  double t2578;
  double t2679;
  double t2747;
  double t2765;
  double t208;
  double t234;
  double t239;
  double t265;
  double t302;
  double t313;
  double t1886;
  double t1929;
  double t1947;
  double t2260;
  double t2358;
  double t2373;
  double t2604;
  double t2637;
  double t2767;
  double t2779;
  double t2793;
  double t2798;
  double t2822;
  double t2825;
  double t2893;
  double t2895;
  double t2959;
  double t2993;
  double t3002;
  double t3014;
  double t3020;
  double t3021;
  double t3124;
  double t3156;
  double t156;
  double t204;
  double t205;
  double t273;
  double t274;
  double t2640;
  double t2896;
  double t2921;
  double t2930;
  double t3160;
  double t3162;
  double t3164;
  double t3190;
  double t3211;
  double t3284;
  double t3299;
  double t3330;
  double t3366;
  double t3400;
  double t3408;
  double t3412;
  double t3426;
  double t3427;
  double t3449;
  double t3451;
  double t3503;
  double t3508;
  double t3516;
  double t3517;
  double t3518;
  double t242;
  double t245;
  double t3163;
  double t3402;
  double t3551;
  double t3553;
  double t3615;
  double t3635;
  double t3701;
  double t3730;
  double t3733;
  double t3778;
  double t3789;
  double t3832;
  double t3853;
  double t3883;
  double t3894;
  double t3931;
  double t3935;
  double t3972;
  double t3980;
  double t3981;
  double t3983;
  double t3991;
  double t3996;
  double t3997;
  double t4013;
  double t4044;
  double t3647;
  double t3699;
  double t3700;
  double t3891;
  double t3901;
  double t3915;
  double t4020;
  double t4023;
  double t124;
  double t4048;
  double t4059;
  double t4061;
  double t4064;
  double t4075;
  double t4094;
  double t4102;
  double t4103;
  double t4116;
  double t4037;
  double t4105;
  double t4107;
  double t87;
  double t4117;
  double t4120;
  double t4126;
  double t86;
  double t4220;
  double t4225;
  double t4226;
  double t4227;
  double t4231;
  double t4234;
  double t4240;
  double t4242;
  double t4247;
  double t4250;
  double t4255;
  double t4262;
  double t4264;
  double t4270;
  double t4280;
  double t4291;
  double t4295;
  double t4298;
  double t4238;
  double t4263;
  double t4305;
  double t4306;
  double t4312;
  double t4319;
  double t4320;
  double t4321;
  double t4323;
  double t4330;
  double t4343;
  double t4345;
  double t4310;
  double t4322;
  double t4347;
  double t4357;
  double t4361;
  double t4362;
  double t4365;
  double t4368;
  double t4380;
  double t4393;
  double t4394;
  double t4397;
  double t4360;
  double t4371;
  double t4399;
  double t4401;
  double t4403;
  double t4405;
  double t4407;
  double t4408;
  double t4414;
  double t4424;
  double t4425;
  double t4426;
  double t4402;
  double t4409;
  double t4428;
  double t4433;
  double t4441;
  double t4444;
  double t4447;
  double t4448;
  double t4453;
  double t4457;
  double t4458;
  double t4460;
  double t4434;
  double t4452;
  double t4463;
  double t4464;
  double t4467;
  double t4468;
  double t4473;
  double t4475;
  double t4142;
  double t4145;
  double t4466;
  double t4477;
  double t4478;
  double t4484;
  double t4486;
  double t4487;
  double t4159;
  double t4161;
  double t4170;
  double t4179;
  double t4183;
  double t4190;
  double t4193;
  double t4560;
  double t4565;
  double t4567;
  double t4572;
  double t4576;
  double t4578;
  double t4581;
  double t4583;
  double t4588;
  double t4591;
  double t4593;
  double t4595;
  double t4575;
  double t4587;
  double t4596;
  double t4597;
  double t4603;
  double t4606;
  double t4607;
  double t4608;
  double t4627;
  double t4630;
  double t4633;
  double t4634;
  double t4601;
  double t4613;
  double t4638;
  double t4645;
  double t4655;
  double t4670;
  double t4679;
  double t4683;
  double t4693;
  double t4706;
  double t4714;
  double t4720;
  double t4649;
  double t4688;
  double t4734;
  double t4737;
  double t4748;
  double t4749;
  double t4750;
  double t4753;
  double t4757;
  double t4758;
  double t4762;
  double t4763;
  double t4746;
  double t4755;
  double t4764;
  double t4771;
  double t4780;
  double t4781;
  double t4786;
  double t4787;
  double t4794;
  double t4795;
  double t4798;
  double t4799;
  double t4774;
  double t4792;
  double t4801;
  double t4802;
  double t4807;
  double t4809;
  double t4811;
  double t4812;
  double t4804;
  double t4814;
  double t4816;
  double t4818;
  double t4820;
  double t4825;
  double t4109;
  double t4130;
  double t4135;
  double t4148;
  double t4151;
  double t4153;
  double t4154;
  double t4171;
  double t4187;
  double t4197;
  double t4206;
  double t4209;
  double t4212;
  double t4482;
  double t4491;
  double t4494;
  double t4502;
  double t4505;
  double t4507;
  double t4522;
  double t4524;
  double t4526;
  double t4535;
  double t4540;
  double t4541;
  double t4545;
  double t4817;
  double t4828;
  double t4831;
  double t4833;
  double t4835;
  double t4838;
  double t4839;
  double t4840;
  double t4841;
  double t4843;
  double t4844;
  double t4846;
  double t4853;
  double t4973;
  double t4976;
  double t4997;
  double t4959;
  double t4960;
  double t4964;
  double t4971;
  double t4972;
  double t4974;
  double t4975;
  double t4977;
  double t4980;
  double t4985;
  double t4988;
  double t4989;
  double t4990;
  double t4998;
  double t5002;
  double t5003;
  double t5004;
  double t5005;
  double t5008;
  double t5009;
  double t5015;
  double t5026;
  double t5032;
  double t5033;
  double t5036;
  double t5037;
  double t5039;
  double t5041;
  double t5042;
  double t5046;
  double t4956;
  double t4957;
  double t4958;
  double t4987;
  double t5014;
  double t5047;
  double t5049;
  double t5053;
  double t5056;
  double t5057;
  double t5059;
  double t5062;
  double t5071;
  double t5077;
  double t5081;
  double t5085;
  double t5086;
  double t5090;
  double t5096;
  double t5102;
  double t5109;
  double t4951;
  double t4952;
  double t4954;
  double t5050;
  double t5080;
  double t5110;
  double t5113;
  double t5136;
  double t5139;
  double t5140;
  double t5141;
  double t5143;
  double t5144;
  double t5145;
  double t5154;
  double t5158;
  double t5160;
  double t5163;
  double t5164;
  double t5170;
  double t5173;
  double t4944;
  double t4945;
  double t4946;
  double t5134;
  double t5151;
  double t5174;
  double t5175;
  double t5181;
  double t5184;
  double t5191;
  double t5194;
  double t5196;
  double t5200;
  double t5201;
  double t5207;
  double t5211;
  double t5214;
  double t5218;
  double t5222;
  double t5224;
  double t5226;
  double t4938;
  double t4940;
  double t5177;
  double t5203;
  double t5204;
  double t5232;
  double t5236;
  double t5242;
  double t5244;
  double t5245;
  double t5248;
  double t5252;
  double t5253;
  double t5255;
  double t5273;
  double t5275;
  double t5277;
  double t5280;
  double t5285;
  double t5287;
  double t5288;
  double t4929;
  double t4932;
  double t4937;
  double t5240;
  double t5263;
  double t5290;
  double t5292;
  double t5306;
  double t5309;
  double t5314;
  double t5315;
  double t5318;
  double t5320;
  double t5323;
  double t4925;
  double t5299;
  double t5325;
  double t5326;
  double t5330;
  double t5331;
  double t5335;
  double t5339;
  double t5340;
  double t5341;
  double t4917;
  double t4918;
  double t4919;
  double t5354;
  double t5356;
  double t5357;
  double t5359;
  double t5360;
  double t5362;
  double t5366;
  double t5368;
  double t5370;
  double t5373;
  double t4907;
  double t4908;
  double t4911;
  double t5327;
  double t5348;
  double t5349;
  double t5358;
  double t5374;
  double t5379;
  double t5386;
  double t5387;
  double t5388;
  double t4890;
  double t4894;
  double t4900;
  double t5399;
  double t5351;
  double t5435;
  double t5414;
  double t5460;
  double t5449;
  double t5482;
  double t5472;
  double t5499;
  double t5488;
  double t5528;
  double t5516;
  double t5741;
  double t5742;
  double t5743;
  double t5744;
  double t5747;
  double t5749;
  double t5753;
  double t5758;
  double t5795;
  double t5796;
  double t5802;
  double t5803;
  double t5806;
  double t5809;
  double t5811;
  double t5812;
  double t5833;
  double t5834;
  double t5835;
  double t5839;
  double t5845;
  double t5846;
  double t5847;
  double t5848;
  double t5892;
  double t5893;
  double t5895;
  double t5898;
  double t5903;
  double t5909;
  double t5917;
  double t5926;
  double t5949;
  double t5950;
  double t5951;
  double t5952;
  double t5954;
  double t5956;
  double t5957;
  double t5958;
  double t5991;
  double t5994;
  double t5995;
  double t6004;
  double t6008;
  double t6009;
  double t6010;
  double t6012;
  double t6055;
  double t6059;
  double t6070;
  double t6074;
  double t6076;
  double t6080;
  double t6081;
  double t6083;
  double t6107;
  double t6108;
  double t6110;
  double t6112;
  double t6114;
  double t6115;
  double t6116;
  double t6119;
  double t6145;
  double t6148;
  double t6151;
  double t6160;
  double t6175;
  double t6179;
  double t6180;
  double t6181;
  double t6230;
  double t6231;
  double t6233;
  double t6236;
  double t6238;
  double t6239;
  double t6241;
  double t6242;
  double t6265;
  double t6266;
  double t6267;
  double t6268;
  double t6272;
  double t6275;
  double t6278;
  double t6281;
  double t6306;
  double t6308;
  double t6310;
  double t6314;
  double t6318;
  double t6319;
  double t6322;
  double t6324;
  double t6385;
  double t6386;
  double t6389;
  double t6391;
  double t6394;
  double t6397;
  double t6402;
  double t6407;
  double t6450;
  double t6451;
  double t6452;
  double t6453;
  double t6434;
  double t6439;
  double t6441;
  double t6442;
  double t6492;
  double t6493;
  double t6494;
  double t6495;
  double t6486;
  double t6487;
  double t6488;
  double t6490;
  t423 = Cos(var1[12]);
  t568 = -1.*t423;
  t570 = 1. + t568;
  t995 = Sin(var1[12]);
  t374 = Cos(var1[11]);
  t376 = -1.*t374;
  t415 = 1. + t376;
  t844 = -0.120666640478*t570;
  t283 = Sin(var1[10]);
  t614 = -0.952469601425*t570;
  t727 = 1. + t614;
  t760 = 0.930418*t727;
  t1090 = -0.803828*t995;
  t1094 = t844 + t1090;
  t1163 = 0.366501*t1094;
  t1189 = t760 + t1163;
  t1192 = Sin(var1[11]);
  t1373 = -0.175248972904*t570;
  t1460 = -0.553471*t995;
  t1471 = t1373 + t1460;
  t1506 = 0.930418*t1471;
  t1509 = 0.444895486988*t570;
  t1527 = -0.218018*t995;
  t1546 = t1509 + t1527;
  t1561 = 0.366501*t1546;
  t1563 = t1506 + t1561;
  t1703 = -0.693671301908*t570;
  t1718 = 1. + t1703;
  t1727 = 0.366501*t1718;
  t1732 = 0.803828*t995;
  t1734 = t844 + t1732;
  t1790 = 0.930418*t1734;
  t1823 = t1727 + t1790;
  t2334 = Cos(var1[10]);
  t2341 = -1.*t2334;
  t2357 = 1. + t2341;
  t1190 = 0.340999127418*t415*t1189;
  t1238 = 0.930418*t1192;
  t1307 = 0. + t1238;
  t1665 = t1307*t1563;
  t1684 = -0.8656776547239999*t415;
  t1694 = 1. + t1684;
  t1840 = t1694*t1823;
  t1885 = 0. + t1190 + t1665 + t1840;
  t2022 = -0.134322983001*t415;
  t2057 = 1. + t2022;
  t2058 = t2057*t1189;
  t2086 = -0.366501*t1192;
  t2098 = 0. + t2086;
  t2106 = t2098*t1563;
  t2127 = 0.340999127418*t415*t1823;
  t2241 = 0. + t2058 + t2106 + t2127;
  t2376 = 0.366501*t1192;
  t2390 = 0. + t2376;
  t2410 = t2390*t1189;
  t2436 = -1.000000637725*t415;
  t2466 = 1. + t2436;
  t2487 = t2466*t1563;
  t2500 = -0.930418*t1192;
  t2547 = 0. + t2500;
  t2568 = t2547*t1823;
  t2578 = 0. + t2410 + t2487 + t2568;
  t2679 = Cos(var1[9]);
  t2747 = -1.*t2679;
  t2765 = 1. + t2747;
  t208 = Cos(var1[8]);
  t234 = -1.*t208;
  t239 = 1. + t234;
  t265 = Sin(var1[9]);
  t302 = -0.930418*t283;
  t313 = 0. + t302;
  t1886 = t313*t1885;
  t1929 = 0.366501*t283;
  t1947 = 0. + t1929;
  t2260 = t1947*t2241;
  t2358 = -1.000000637725*t2357;
  t2373 = 1. + t2358;
  t2604 = t2373*t2578;
  t2637 = 0. + t1886 + t2260 + t2604;
  t2767 = 0.340999127418*t2357*t1885;
  t2779 = -0.134322983001*t2357;
  t2793 = 1. + t2779;
  t2798 = t2793*t2241;
  t2822 = -0.366501*t283;
  t2825 = 0. + t2822;
  t2893 = t2825*t2578;
  t2895 = 0. + t2767 + t2798 + t2893;
  t2959 = -0.8656776547239999*t2357;
  t2993 = 1. + t2959;
  t3002 = t2993*t1885;
  t3014 = 0.340999127418*t2357*t2241;
  t3020 = 0.930418*t283;
  t3021 = 0. + t3020;
  t3124 = t3021*t2578;
  t3156 = 0. + t3002 + t3014 + t3124;
  t156 = Cos(var1[7]);
  t204 = -1.*t156;
  t205 = 1. + t204;
  t273 = 0.930418*t265;
  t274 = 0. + t273;
  t2640 = t274*t2637;
  t2896 = 0.340999127418*t2765*t2895;
  t2921 = -0.8656776547239999*t2765;
  t2930 = 1. + t2921;
  t3160 = t2930*t3156;
  t3162 = 0. + t2640 + t2896 + t3160;
  t3164 = -0.366501*t265;
  t3190 = 0. + t3164;
  t3211 = t3190*t2637;
  t3284 = -0.134322983001*t2765;
  t3299 = 1. + t3284;
  t3330 = t3299*t2895;
  t3366 = 0.340999127418*t2765*t3156;
  t3400 = 0. + t3211 + t3330 + t3366;
  t3408 = -1.000000637725*t2765;
  t3412 = 1. + t3408;
  t3426 = t3412*t2637;
  t3427 = 0.366501*t265;
  t3449 = 0. + t3427;
  t3451 = t3449*t2895;
  t3503 = -0.930418*t265;
  t3508 = 0. + t3503;
  t3516 = t3508*t3156;
  t3517 = 0. + t3426 + t3451 + t3516;
  t3518 = Sin(var1[8]);
  t242 = -0.8656776547239999*t239;
  t245 = 1. + t242;
  t3163 = t245*t3162;
  t3402 = 0.340999127418*t239*t3400;
  t3551 = -0.930418*t3518;
  t3553 = 0. + t3551;
  t3615 = t3517*t3553;
  t3635 = 0. + t3163 + t3402 + t3615;
  t3701 = 0.340999127418*t239*t3162;
  t3730 = -0.134322983001*t239;
  t3733 = 1. + t3730;
  t3778 = t3733*t3400;
  t3789 = 0.366501*t3518;
  t3832 = 0. + t3789;
  t3853 = t3517*t3832;
  t3883 = 0. + t3701 + t3778 + t3853;
  t3894 = Sin(var1[7]);
  t3931 = -1.000000637725*t239;
  t3935 = 1. + t3931;
  t3972 = t3935*t3517;
  t3980 = -0.366501*t3518;
  t3981 = 0. + t3980;
  t3983 = t3400*t3981;
  t3991 = 0.930418*t3518;
  t3996 = 0. + t3991;
  t3997 = t3162*t3996;
  t4013 = 0. + t3972 + t3983 + t3997;
  t4044 = Cos(var1[6]);
  t3647 = -0.340999127418*t205*t3635;
  t3699 = -0.8656776547239999*t205;
  t3700 = 1. + t3699;
  t3891 = t3700*t3883;
  t3901 = -0.930418*t3894;
  t3915 = 0. + t3901;
  t4020 = t3915*t4013;
  t4023 = 0. + t3647 + t3891 + t4020;
  t124 = Sin(var1[6]);
  t4048 = -0.134322983001*t205;
  t4059 = 1. + t4048;
  t4061 = t4059*t3635;
  t4064 = -0.340999127418*t205*t3883;
  t4075 = -0.366501*t3894;
  t4094 = 0. + t4075;
  t4102 = t4094*t4013;
  t4103 = 0. + t4061 + t4064 + t4102;
  t4116 = Cos(var1[5]);
  t4037 = -1.*t124*t4023;
  t4105 = t4044*t4103;
  t4107 = 0. + t4037 + t4105;
  t87 = Sin(var1[5]);
  t4117 = t4044*t4023;
  t4120 = t124*t4103;
  t4126 = 0. + t4117 + t4120;
  t86 = Sin(var1[3]);
  t4220 = -0.353861996165*t570;
  t4225 = 1. + t4220;
  t4226 = -0.594863*t4225;
  t4227 = -0.294604*t1471;
  t4231 = 0.747896*t1546;
  t4234 = t4226 + t4227 + t4231;
  t4240 = -0.294604*t727;
  t4242 = 0.747896*t1094;
  t4247 = 0.553471*t995;
  t4250 = t1373 + t4247;
  t4255 = -0.594863*t4250;
  t4262 = t4240 + t4242 + t4255;
  t4264 = 0.747896*t1718;
  t4270 = 0.218018*t995;
  t4280 = t1509 + t4270;
  t4291 = -0.594863*t4280;
  t4295 = -0.294604*t1734;
  t4298 = t4264 + t4291 + t4295;
  t4238 = t1307*t4234;
  t4263 = 0.340999127418*t415*t4262;
  t4305 = t1694*t4298;
  t4306 = 0. + t4238 + t4263 + t4305;
  t4312 = t2098*t4234;
  t4319 = t2057*t4262;
  t4320 = 0.340999127418*t415*t4298;
  t4321 = 0. + t4312 + t4319 + t4320;
  t4323 = t2466*t4234;
  t4330 = t2390*t4262;
  t4343 = t2547*t4298;
  t4345 = 0. + t4323 + t4330 + t4343;
  t4310 = t313*t4306;
  t4322 = t1947*t4321;
  t4347 = t2373*t4345;
  t4357 = 0. + t4310 + t4322 + t4347;
  t4361 = 0.340999127418*t2357*t4306;
  t4362 = t2793*t4321;
  t4365 = t2825*t4345;
  t4368 = 0. + t4361 + t4362 + t4365;
  t4380 = t2993*t4306;
  t4393 = 0.340999127418*t2357*t4321;
  t4394 = t3021*t4345;
  t4397 = 0. + t4380 + t4393 + t4394;
  t4360 = t274*t4357;
  t4371 = 0.340999127418*t2765*t4368;
  t4399 = t2930*t4397;
  t4401 = 0. + t4360 + t4371 + t4399;
  t4403 = t3190*t4357;
  t4405 = t3299*t4368;
  t4407 = 0.340999127418*t2765*t4397;
  t4408 = 0. + t4403 + t4405 + t4407;
  t4414 = t3412*t4357;
  t4424 = t3449*t4368;
  t4425 = t3508*t4397;
  t4426 = 0. + t4414 + t4424 + t4425;
  t4402 = t245*t4401;
  t4409 = 0.340999127418*t239*t4408;
  t4428 = t4426*t3553;
  t4433 = 0. + t4402 + t4409 + t4428;
  t4441 = 0.340999127418*t239*t4401;
  t4444 = t3733*t4408;
  t4447 = t4426*t3832;
  t4448 = 0. + t4441 + t4444 + t4447;
  t4453 = t3935*t4426;
  t4457 = t4408*t3981;
  t4458 = t4401*t3996;
  t4460 = 0. + t4453 + t4457 + t4458;
  t4434 = -0.340999127418*t205*t4433;
  t4452 = t3700*t4448;
  t4463 = t3915*t4460;
  t4464 = 0. + t4434 + t4452 + t4463;
  t4467 = t4059*t4433;
  t4468 = -0.340999127418*t205*t4448;
  t4473 = t4094*t4460;
  t4475 = 0. + t4467 + t4468 + t4473;
  t4142 = Cos(var1[3]);
  t4145 = Sin(var1[4]);
  t4466 = -1.*t124*t4464;
  t4477 = t4044*t4475;
  t4478 = 0. + t4466 + t4477;
  t4484 = t4044*t4464;
  t4486 = t124*t4475;
  t4487 = 0. + t4484 + t4486;
  t4159 = Cos(var1[4]);
  t4161 = 0.366501*t3894;
  t4170 = 0. + t4161;
  t4179 = 0.930418*t3894;
  t4183 = 0. + t4179;
  t4190 = -1.000000637725*t205;
  t4193 = 1. + t4190;
  t4560 = 0.803828*t4225;
  t4565 = -0.218018*t1471;
  t4567 = 0.553471*t1546;
  t4572 = t4560 + t4565 + t4567;
  t4576 = -0.218018*t727;
  t4578 = 0.553471*t1094;
  t4581 = 0.803828*t4250;
  t4583 = t4576 + t4578 + t4581;
  t4588 = 0.553471*t1718;
  t4591 = 0.803828*t4280;
  t4593 = -0.218018*t1734;
  t4595 = t4588 + t4591 + t4593;
  t4575 = t1307*t4572;
  t4587 = 0.340999127418*t415*t4583;
  t4596 = t1694*t4595;
  t4597 = 0. + t4575 + t4587 + t4596;
  t4603 = t2098*t4572;
  t4606 = t2057*t4583;
  t4607 = 0.340999127418*t415*t4595;
  t4608 = 0. + t4603 + t4606 + t4607;
  t4627 = t2466*t4572;
  t4630 = t2390*t4583;
  t4633 = t2547*t4595;
  t4634 = 0. + t4627 + t4630 + t4633;
  t4601 = t313*t4597;
  t4613 = t1947*t4608;
  t4638 = t2373*t4634;
  t4645 = 0. + t4601 + t4613 + t4638;
  t4655 = 0.340999127418*t2357*t4597;
  t4670 = t2793*t4608;
  t4679 = t2825*t4634;
  t4683 = 0. + t4655 + t4670 + t4679;
  t4693 = t2993*t4597;
  t4706 = 0.340999127418*t2357*t4608;
  t4714 = t3021*t4634;
  t4720 = 0. + t4693 + t4706 + t4714;
  t4649 = t274*t4645;
  t4688 = 0.340999127418*t2765*t4683;
  t4734 = t2930*t4720;
  t4737 = 0. + t4649 + t4688 + t4734;
  t4748 = t3190*t4645;
  t4749 = t3299*t4683;
  t4750 = 0.340999127418*t2765*t4720;
  t4753 = 0. + t4748 + t4749 + t4750;
  t4757 = t3412*t4645;
  t4758 = t3449*t4683;
  t4762 = t3508*t4720;
  t4763 = 0. + t4757 + t4758 + t4762;
  t4746 = t245*t4737;
  t4755 = 0.340999127418*t239*t4753;
  t4764 = t4763*t3553;
  t4771 = 0. + t4746 + t4755 + t4764;
  t4780 = 0.340999127418*t239*t4737;
  t4781 = t3733*t4753;
  t4786 = t4763*t3832;
  t4787 = 0. + t4780 + t4781 + t4786;
  t4794 = t3935*t4763;
  t4795 = t4753*t3981;
  t4798 = t4737*t3996;
  t4799 = 0. + t4794 + t4795 + t4798;
  t4774 = -0.340999127418*t205*t4771;
  t4792 = t3700*t4787;
  t4801 = t3915*t4799;
  t4802 = 0. + t4774 + t4792 + t4801;
  t4807 = t4059*t4771;
  t4809 = -0.340999127418*t205*t4787;
  t4811 = t4094*t4799;
  t4812 = 0. + t4807 + t4809 + t4811;
  t4804 = -1.*t124*t4802;
  t4814 = t4044*t4812;
  t4816 = 0. + t4804 + t4814;
  t4818 = t4044*t4802;
  t4820 = t124*t4812;
  t4825 = 0. + t4818 + t4820;
  t4109 = -1.*t87*t4107;
  t4130 = t4116*t4126;
  t4135 = 0. + t4109 + t4130;
  t4148 = t4116*t4107;
  t4151 = t87*t4126;
  t4153 = 0. + t4148 + t4151;
  t4154 = t4145*t4153;
  t4171 = t4170*t3635;
  t4187 = t4183*t3883;
  t4197 = t4193*t4013;
  t4206 = 0. + t4171 + t4187 + t4197;
  t4209 = t4159*t4206;
  t4212 = 0. + t4154 + t4209;
  t4482 = -1.*t87*t4478;
  t4491 = t4116*t4487;
  t4494 = 0. + t4482 + t4491;
  t4502 = t4116*t4478;
  t4505 = t87*t4487;
  t4507 = 0. + t4502 + t4505;
  t4522 = t4145*t4507;
  t4524 = t4170*t4433;
  t4526 = t4183*t4448;
  t4535 = t4193*t4460;
  t4540 = 0. + t4524 + t4526 + t4535;
  t4541 = t4159*t4540;
  t4545 = 0. + t4522 + t4541;
  t4817 = -1.*t87*t4816;
  t4828 = t4116*t4825;
  t4831 = 0. + t4817 + t4828;
  t4833 = t4116*t4816;
  t4835 = t87*t4825;
  t4838 = 0. + t4833 + t4835;
  t4839 = t4145*t4838;
  t4840 = t4170*t4771;
  t4841 = t4183*t4787;
  t4843 = t4193*t4799;
  t4844 = 0. + t4840 + t4841 + t4843;
  t4846 = t4159*t4844;
  t4853 = 0. + t4839 + t4846;
  t4973 = -0.444895486988*t570;
  t4976 = 0.175248972904*t570;
  t4997 = 0.120666640478*t570;
  t4959 = -1.0464152525368286e-7*var1[12];
  t4960 = 0.061997*t4225;
  t4964 = 0.15748087543254813*t570;
  t4971 = 0.323516*t1471;
  t4972 = -0.823565*t1546;
  t4974 = t4973 + t4270;
  t4975 = -0.5286755231320001*t4974;
  t4977 = t4976 + t4247;
  t4980 = 0.29871295412*t4977;
  t4985 = t4959 + t4960 + t4964 + t4971 + t4972 + t4975 + t4980;
  t4988 = -7.20503013377005e-8*var1[12];
  t4989 = -0.823565*t1718;
  t4990 = -0.3667270384178674*t570;
  t4998 = t4997 + t1090;
  t5002 = 0.29871295412*t4998;
  t5003 = t4973 + t1527;
  t5004 = 0.44503472296900004*t5003;
  t5005 = 0.061997*t4280;
  t5008 = 0.323516*t1734;
  t5009 = t4988 + t4989 + t4990 + t5002 + t5004 + t5005 + t5008;
  t5015 = 2.838136523330542e-8*var1[12];
  t5026 = 0.323516*t727;
  t5032 = 0.2845150083511607*t570;
  t5033 = -0.823565*t1094;
  t5036 = t4976 + t1460;
  t5037 = 0.44503472296900004*t5036;
  t5039 = 0.061997*t4250;
  t5041 = t4997 + t1732;
  t5042 = -0.5286755231320001*t5041;
  t5046 = t5015 + t5026 + t5032 + t5033 + t5037 + t5039 + t5042;
  t4956 = 7.500378623168247e-8*var1[11];
  t4957 = 0.32429092013729516*t415;
  t4958 = 0.061996937216*t2390;
  t4987 = t2098*t4985;
  t5014 = 0.340999127418*t415*t5009;
  t5047 = t2057*t5046;
  t5049 = t4956 + t4957 + t4958 + t4987 + t5014 + t5047;
  t5053 = 2.95447451120871e-8*var1[11];
  t5056 = -0.8232613535360118*t415;
  t5057 = 0.061996937216*t2547;
  t5059 = t1307*t4985;
  t5062 = t1694*t5009;
  t5071 = 0.340999127418*t415*t5046;
  t5077 = t5053 + t5056 + t5057 + t5059 + t5062 + t5071;
  t5081 = 0.06199697675299678*t415;
  t5085 = 0.324290713329*t2098;
  t5086 = -0.823260828522*t1307;
  t5090 = t2466*t4985;
  t5096 = t2547*t5009;
  t5102 = t2390*t5046;
  t5109 = 0. + t5081 + t5085 + t5086 + t5090 + t5096 + t5102;
  t4951 = -0.26281014453449253*t2357;
  t4952 = 0.23261818470000004*t2825;
  t4954 = -0.5905363046000001*t3021;
  t5050 = t1947*t5049;
  t5080 = t313*t5077;
  t5110 = t2373*t5109;
  t5113 = 0. + t4951 + t4952 + t4954 + t5050 + t5080 + t5110;
  t5136 = 5.7930615939377813e-8*var1[10];
  t5139 = 0.23261833304643187*t2357;
  t5140 = -0.262809976934*t1947;
  t5141 = t2793*t5049;
  t5143 = 0.340999127418*t2357*t5077;
  t5144 = t2825*t5109;
  t5145 = t5136 + t5139 + t5140 + t5141 + t5143 + t5144;
  t5154 = 2.281945176511838e-8*var1[10];
  t5158 = -0.5905366811997648*t2357;
  t5160 = -0.262809976934*t313;
  t5163 = 0.340999127418*t2357*t5049;
  t5164 = t2993*t5077;
  t5170 = t3021*t5109;
  t5173 = t5154 + t5158 + t5160 + t5163 + t5164 + t5170;
  t4944 = -6.295460977284962e-8*var1[9];
  t4945 = -0.22023473313910558*t2765;
  t4946 = 0.039853013046*t3508;
  t5134 = t274*t5113;
  t5151 = 0.340999127418*t2765*t5145;
  t5174 = t2930*t5173;
  t5175 = t4944 + t4945 + t4946 + t5134 + t5151 + t5174;
  t5181 = -1.5981976069815686e-7*var1[9];
  t5184 = 0.08675267452931407*t2765;
  t5191 = 0.039853013046*t3449;
  t5194 = t3190*t5113;
  t5196 = t3299*t5145;
  t5200 = 0.340999127418*t2765*t5173;
  t5201 = t5181 + t5184 + t5191 + t5194 + t5196 + t5200;
  t5207 = 0.039853038461262744*t2765;
  t5211 = 0.086752619205*t3190;
  t5214 = -0.22023459268999998*t274;
  t5218 = t3412*t5113;
  t5222 = t3449*t5145;
  t5224 = t3508*t5173;
  t5226 = 0. + t5207 + t5211 + t5214 + t5218 + t5222 + t5224;
  t4938 = 3.2909349868922137e-7*var1[8];
  t4940 = 0.055653945343889656*t239;
  t5177 = 0.340999127418*t239*t5175;
  t5203 = t3733*t5201;
  t5204 = -0.045000372235*t3981;
  t5232 = t5226*t3832;
  t5236 = t4938 + t4940 + t5177 + t5203 + t5204 + t5232;
  t5242 = 1.296332362046933e-7*var1[8];
  t5244 = -0.14128592423750855*t239;
  t5245 = t245*t5175;
  t5248 = 0.340999127418*t239*t5201;
  t5252 = t5226*t3553;
  t5253 = -0.045000372235*t3996;
  t5255 = t5242 + t5244 + t5245 + t5248 + t5252 + t5253;
  t5273 = -0.04500040093286238*t239;
  t5275 = t3935*t5226;
  t5277 = -0.141285834136*t3553;
  t5280 = t5201*t3981;
  t5285 = 0.055653909852*t3832;
  t5287 = t5175*t3996;
  t5288 = 0. + t5273 + t5275 + t5277 + t5280 + t5285 + t5287;
  t4929 = 1.296332362046933e-7*var1[7];
  t4932 = 0.07877668146182712*t205;
  t4937 = -0.045000372235*t4183;
  t5240 = t3700*t5236;
  t5263 = -0.340999127418*t205*t5255;
  t5290 = t3915*t5288;
  t5292 = t4929 + t4932 + t4937 + t5240 + t5263 + t5290;
  t5306 = -3.2909349868922137e-7*var1[7];
  t5309 = 0.03103092645718495*t205;
  t5314 = -0.045000372235*t4170;
  t5315 = -0.340999127418*t205*t5236;
  t5318 = t4059*t5255;
  t5320 = t4094*t5288;
  t5323 = t5306 + t5309 + t5314 + t5315 + t5318 + t5320;
  t4925 = 0.091*t124;
  t5299 = -1.*t124*t5292;
  t5325 = t4044*t5323;
  t5326 = 0. + t4925 + t5299 + t5325;
  t5330 = -1.*t4044;
  t5331 = 1. + t5330;
  t5335 = 0.091*t5331;
  t5339 = t4044*t5292;
  t5340 = t124*t5323;
  t5341 = 0. + t5335 + t5339 + t5340;
  t4917 = t4159*t4838;
  t4918 = -1.*t4145*t4844;
  t4919 = 0. + t4917 + t4918;
  t5354 = t4116*t5326;
  t5356 = t87*t5341;
  t5357 = 0. + t5354 + t5356;
  t5359 = -0.04500040093286238*t205;
  t5360 = 0.07877663122399998*t3915;
  t5362 = 0.031030906668*t4094;
  t5366 = t4183*t5236;
  t5368 = t4170*t5255;
  t5370 = t4193*t5288;
  t5373 = 0. + t5359 + t5360 + t5362 + t5366 + t5368 + t5370;
  t4907 = t4159*t4507;
  t4908 = -1.*t4145*t4540;
  t4911 = 0. + t4907 + t4908;
  t5327 = -1.*t87*t5326;
  t5348 = t4116*t5341;
  t5349 = 0. + t5327 + t5348;
  t5358 = t4145*t5357;
  t5374 = t4159*t5373;
  t5379 = 0. + t5358 + t5374;
  t5386 = t4159*t5357;
  t5387 = -1.*t4145*t5373;
  t5388 = 0. + t5386 + t5387;
  t4890 = t4159*t4153;
  t4894 = -1.*t4145*t4206;
  t4900 = 0. + t4890 + t4894;
  t5399 = -1.*t4494*t5349;
  t5351 = t4831*t5349;
  t5435 = -1.*t4831*t5349;
  t5414 = t4135*t5349;
  t5460 = t4494*t5349;
  t5449 = -1.*t4135*t5349;
  t5482 = t4844*t5373;
  t5472 = -1.*t4540*t5373;
  t5499 = t4206*t5373;
  t5488 = -1.*t4844*t5373;
  t5528 = -1.*t4206*t5373;
  t5516 = t4540*t5373;
  t5741 = t4787*t5236;
  t5742 = t4771*t5255;
  t5743 = t4799*t5288;
  t5744 = t5741 + t5742 + t5743;
  t5747 = -1.*t4448*t5236;
  t5749 = -1.*t4433*t5255;
  t5753 = -1.*t4460*t5288;
  t5758 = t5747 + t5749 + t5753;
  t5795 = t3883*t5236;
  t5796 = t3635*t5255;
  t5802 = t4013*t5288;
  t5803 = t5795 + t5796 + t5802;
  t5806 = -1.*t4787*t5236;
  t5809 = -1.*t4771*t5255;
  t5811 = -1.*t4799*t5288;
  t5812 = t5806 + t5809 + t5811;
  t5833 = -1.*t3883*t5236;
  t5834 = -1.*t3635*t5255;
  t5835 = -1.*t4013*t5288;
  t5839 = t5833 + t5834 + t5835;
  t5845 = t4448*t5236;
  t5846 = t4433*t5255;
  t5847 = t4460*t5288;
  t5848 = t5845 + t5846 + t5847;
  t5892 = t4737*t5175;
  t5893 = t4753*t5201;
  t5895 = t4763*t5226;
  t5898 = t5892 + t5893 + t5895;
  t5903 = -1.*t4401*t5175;
  t5909 = -1.*t4408*t5201;
  t5917 = -1.*t4426*t5226;
  t5926 = t5903 + t5909 + t5917;
  t5949 = t3162*t5175;
  t5950 = t3400*t5201;
  t5951 = t3517*t5226;
  t5952 = t5949 + t5950 + t5951;
  t5954 = -1.*t4737*t5175;
  t5956 = -1.*t4753*t5201;
  t5957 = -1.*t4763*t5226;
  t5958 = t5954 + t5956 + t5957;
  t5991 = -1.*t3162*t5175;
  t5994 = -1.*t3400*t5201;
  t5995 = -1.*t3517*t5226;
  t6004 = t5991 + t5994 + t5995;
  t6008 = t4401*t5175;
  t6009 = t4408*t5201;
  t6010 = t4426*t5226;
  t6012 = t6008 + t6009 + t6010;
  t6055 = t4645*t5113;
  t6059 = t4683*t5145;
  t6070 = t4720*t5173;
  t6074 = t6055 + t6059 + t6070;
  t6076 = -1.*t4357*t5113;
  t6080 = -1.*t4368*t5145;
  t6081 = -1.*t4397*t5173;
  t6083 = t6076 + t6080 + t6081;
  t6107 = t2637*t5113;
  t6108 = t2895*t5145;
  t6110 = t3156*t5173;
  t6112 = t6107 + t6108 + t6110;
  t6114 = -1.*t4645*t5113;
  t6115 = -1.*t4683*t5145;
  t6116 = -1.*t4720*t5173;
  t6119 = t6114 + t6115 + t6116;
  t6145 = -1.*t2637*t5113;
  t6148 = -1.*t2895*t5145;
  t6151 = -1.*t3156*t5173;
  t6160 = t6145 + t6148 + t6151;
  t6175 = t4357*t5113;
  t6179 = t4368*t5145;
  t6180 = t4397*t5173;
  t6181 = t6175 + t6179 + t6180;
  t6230 = t4608*t5049;
  t6231 = t4597*t5077;
  t6233 = t4634*t5109;
  t6236 = t6230 + t6231 + t6233;
  t6238 = -1.*t4321*t5049;
  t6239 = -1.*t4306*t5077;
  t6241 = -1.*t4345*t5109;
  t6242 = t6238 + t6239 + t6241;
  t6265 = t2241*t5049;
  t6266 = t1885*t5077;
  t6267 = t2578*t5109;
  t6268 = t6265 + t6266 + t6267;
  t6272 = -1.*t4608*t5049;
  t6275 = -1.*t4597*t5077;
  t6278 = -1.*t4634*t5109;
  t6281 = t6272 + t6275 + t6278;
  t6306 = -1.*t2241*t5049;
  t6308 = -1.*t1885*t5077;
  t6310 = -1.*t2578*t5109;
  t6314 = t6306 + t6308 + t6310;
  t6318 = t4321*t5049;
  t6319 = t4306*t5077;
  t6322 = t4345*t5109;
  t6324 = t6318 + t6319 + t6322;
  t6385 = t4572*t4985;
  t6386 = t4595*t5009;
  t6389 = t4583*t5046;
  t6391 = t6385 + t6386 + t6389;
  t6394 = -1.*t4234*t4985;
  t6397 = -1.*t4298*t5009;
  t6402 = -1.*t4262*t5046;
  t6407 = t6394 + t6397 + t6402;
  t6450 = t1563*t4985;
  t6451 = t5009*t1823;
  t6452 = t1189*t5046;
  t6453 = t6450 + t6451 + t6452;
  t6434 = -1.*t4572*t4985;
  t6439 = -1.*t4595*t5009;
  t6441 = -1.*t4583*t5046;
  t6442 = t6434 + t6439 + t6441;
  t6492 = -1.*t1563*t4985;
  t6493 = -1.*t5009*t1823;
  t6494 = -1.*t1189*t5046;
  t6495 = t6492 + t6493 + t6494;
  t6486 = t4234*t4985;
  t6487 = t4298*t5009;
  t6488 = t4262*t5046;
  t6490 = t6486 + t6487 + t6488;
  p_output1[0]=0. + t4142*t4212 - 1.*t4135*t86;
  p_output1[1]=0. + t4142*t4545 - 1.*t4494*t86;
  p_output1[2]=0. + t4142*t4853 - 1.*t4831*t86;
  p_output1[3]=0;
  p_output1[4]=0;
  p_output1[5]=0;
  p_output1[6]=0. + t4135*t4142 + t4212*t86;
  p_output1[7]=0. + t4142*t4494 + t4545*t86;
  p_output1[8]=0. + t4142*t4831 + t4853*t86;
  p_output1[9]=0;
  p_output1[10]=0;
  p_output1[11]=0;
  p_output1[12]=t4900;
  p_output1[13]=t4911;
  p_output1[14]=t4919;
  p_output1[15]=0;
  p_output1[16]=0;
  p_output1[17]=0;
  p_output1[18]=t4911*(t5351 + t4853*t5379 + t4919*t5388) + t4919*(-1.*t4545*t5379 - 1.*t4911*t5388 + t5399);
  p_output1[19]=t4919*(t4212*t5379 + t4900*t5388 + t5414) + t4900*(-1.*t4853*t5379 - 1.*t4919*t5388 + t5435);
  p_output1[20]=t4911*(-1.*t4212*t5379 - 1.*t4900*t5388 + t5449) + t4900*(t4545*t5379 + t4911*t5388 + t5460);
  p_output1[21]=t4900;
  p_output1[22]=t4911;
  p_output1[23]=t4919;
  p_output1[24]=t4831*(-1.*t4507*t5357 + t5399 + t5472) + t4494*(t5351 + t4838*t5357 + t5482);
  p_output1[25]=t4135*(-1.*t4838*t5357 + t5435 + t5488) + t4831*(t4153*t5357 + t5414 + t5499);
  p_output1[26]=t4135*(t4507*t5357 + t5460 + t5516) + t4494*(-1.*t4153*t5357 + t5449 + t5528);
  p_output1[27]=t4135;
  p_output1[28]=t4494;
  p_output1[29]=t4831;
  p_output1[30]=t4844*(-1.*t4478*t5326 - 1.*t4487*t5341 + t5472) + t4540*(t4816*t5326 + t4825*t5341 + t5482);
  p_output1[31]=t4206*(-1.*t4816*t5326 - 1.*t4825*t5341 + t5488) + t4844*(t4107*t5326 + t4126*t5341 + t5499);
  p_output1[32]=t4206*(t4478*t5326 + t4487*t5341 + t5516) + t4540*(-1.*t4107*t5326 - 1.*t4126*t5341 + t5528);
  p_output1[33]=t4206;
  p_output1[34]=t4540;
  p_output1[35]=t4844;
  p_output1[36]=0.091*t4103 - 1.*t4844*(-1.*t4464*t5292 - 1.*t4475*t5323 + t5472) - 1.*t4540*(t4802*t5292 + t4812*t5323 + t5482);
  p_output1[37]=0.091*t4475 - 1.*t4206*(-1.*t4802*t5292 - 1.*t4812*t5323 + t5488) - 1.*t4844*(t4023*t5292 + t4103*t5323 + t5499);
  p_output1[38]=0.091*t4812 - 1.*t4206*(t4464*t5292 + t4475*t5323 + t5516) - 1.*t4540*(-1.*t4023*t5292 - 1.*t4103*t5323 + t5528);
  p_output1[39]=0. - 1.*t3635*t4170 - 1.*t3883*t4183 - 1.*t4013*t4193;
  p_output1[40]=0. - 1.*t4170*t4433 - 1.*t4183*t4448 - 1.*t4193*t4460;
  p_output1[41]=0. - 1.*t4170*t4771 - 1.*t4183*t4787 - 1.*t4193*t4799;
  p_output1[42]=-0.016493*t3635 - 0.041869*t3883 - 0.084668*t4013 - 0.930418*(t4433*t5744 + t4771*t5758) + 0.366501*(t4448*t5744 + t4787*t5758);
  p_output1[43]=-0.016493*t4433 - 0.041869*t4448 - 0.084668*t4460 - 0.930418*(t4771*t5803 + t3635*t5812) + 0.366501*(t4787*t5803 + t3883*t5812);
  p_output1[44]=-0.016493*t4771 - 0.041869*t4787 - 0.084668*t4799 - 0.930418*(t4433*t5839 + t3635*t5848) + 0.366501*(t4448*t5839 + t3883*t5848);
  p_output1[45]=0. - 0.930418*t3635 + 0.366501*t3883;
  p_output1[46]=0. - 0.930418*t4433 + 0.366501*t4448;
  p_output1[47]=0. - 0.930418*t4771 + 0.366501*t4787;
  p_output1[48]=-0.041869*t3162 + 0.016493*t3400 + 0.151852*t3517 + 0.366501*(t4401*t5898 + t4737*t5926) + 0.930418*(t4408*t5898 + t4753*t5926);
  p_output1[49]=-0.041869*t4401 + 0.016493*t4408 + 0.151852*t4426 + 0.366501*(t4737*t5952 + t3162*t5958) + 0.930418*(t4753*t5952 + t3400*t5958);
  p_output1[50]=-0.041869*t4737 + 0.016493*t4753 + 0.151852*t4763 + 0.366501*(t4401*t6004 + t3162*t6012) + 0.930418*(t4408*t6004 + t3400*t6012);
  p_output1[51]=0. + 0.366501*t3162 + 0.930418*t3400;
  p_output1[52]=0. + 0.366501*t4401 + 0.930418*t4408;
  p_output1[53]=0. + 0.366501*t4737 + 0.930418*t4753;
  p_output1[54]=-0.236705*t2637 + 0.014606*t2895 - 0.03708*t3156 - 0.930418*(t4368*t6074 + t4683*t6083) - 0.366501*(t4397*t6074 + t4720*t6083);
  p_output1[55]=-0.236705*t4357 + 0.014606*t4368 - 0.03708*t4397 - 0.930418*(t4683*t6112 + t2895*t6119) - 0.366501*(t4720*t6112 + t3156*t6119);
  p_output1[56]=-0.236705*t4645 + 0.014606*t4683 - 0.03708*t4720 - 0.930418*(t4368*t6160 + t2895*t6181) - 0.366501*(t4397*t6160 + t3156*t6181);
  p_output1[57]=0. - 0.930418*t2895 - 0.366501*t3156;
  p_output1[58]=0. - 0.930418*t4368 - 0.366501*t4397;
  p_output1[59]=0. - 0.930418*t4683 - 0.366501*t4720;
  p_output1[60]=0.244523*t1885 - 0.09632*t2241 - 0.6347*t2578 - 0.366501*(t4306*t6236 + t4597*t6242) - 0.930418*(t4321*t6236 + t4608*t6242);
  p_output1[61]=0.244523*t4306 - 0.09632*t4321 - 0.6347*t4345 - 0.366501*(t4597*t6268 + t1885*t6281) - 0.930418*(t4608*t6268 + t2241*t6281);
  p_output1[62]=0.244523*t4597 - 0.09632*t4608 - 0.6347*t4634 - 0.366501*(t4306*t6314 + t1885*t6324) - 0.930418*(t4321*t6314 + t2241*t6324);
  p_output1[63]=0. - 0.366501*t1885 - 0.930418*t2241;
  p_output1[64]=0. - 0.366501*t4306 - 0.930418*t4321;
  p_output1[65]=0. - 0.366501*t4597 - 0.930418*t4608;
  p_output1[66]=0.022722*t1189 - 0.884829*t1563 - 0.057683*t1823 - 0.930418*(t4262*t6391 + t4583*t6407) - 0.366501*(t4298*t6391 + t4595*t6407);
  p_output1[67]=-0.884829*t4234 + 0.022722*t4262 - 0.057683*t4298 - 0.930418*(t1189*t6442 + t4583*t6453) - 0.366501*(t1823*t6442 + t4595*t6453);
  p_output1[68]=-0.884829*t4572 + 0.022722*t4583 - 0.057683*t4595 - 0.930418*(t1189*t6490 + t4262*t6495) - 0.366501*(t1823*t6490 + t4298*t6495);
  p_output1[69]=0. - 0.930418*t1189 - 0.366501*t1823;
  p_output1[70]=0. - 0.930418*t4262 - 0.366501*t4298;
  p_output1[71]=0. - 0.930418*t4583 - 0.366501*t4595;
  p_output1[72]=8.097039235488435e-7;
  p_output1[73]=-0.08500000216439943;
  p_output1[74]=1.70232311025309e-8;
  p_output1[75]=-1.9655299995924302e-7;
  p_output1[76]=1.8632400006213246e-7;
  p_output1[77]=1.000001449749;
  p_output1[78]=0;
  p_output1[79]=0;
  p_output1[80]=0;
  p_output1[81]=0;
  p_output1[82]=0;
  p_output1[83]=0;
  p_output1[84]=0;
  p_output1[85]=0;
  p_output1[86]=0;
  p_output1[87]=0;
  p_output1[88]=0;
  p_output1[89]=0;
  p_output1[90]=0;
  p_output1[91]=0;
  p_output1[92]=0;
  p_output1[93]=0;
  p_output1[94]=0;
  p_output1[95]=0;
  p_output1[96]=0;
  p_output1[97]=0;
  p_output1[98]=0;
  p_output1[99]=0;
  p_output1[100]=0;
  p_output1[101]=0;
  p_output1[102]=0;
  p_output1[103]=0;
  p_output1[104]=0;
  p_output1[105]=0;
  p_output1[106]=0;
  p_output1[107]=0;
  p_output1[108]=0;
  p_output1[109]=0;
  p_output1[110]=0;
  p_output1[111]=0;
  p_output1[112]=0;
  p_output1[113]=0;
  p_output1[114]=0;
  p_output1[115]=0;
  p_output1[116]=0;
  p_output1[117]=0;
  p_output1[118]=0;
  p_output1[119]=0;
}



void Jb_LeftToeBottomBack_src(double *p_output1, const double *var1)
{
  // Call Subroutines
  output1(p_output1, var1);

}
