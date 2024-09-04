/*
 * Automatically Generated from Mathematica.
 * Mon 4 Jul 2022 20:33:22 GMT-04:00
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Jb_hip_flexion_left_src.h"

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
  double t1498;
  double t1713;
  double t1889;
  double t913;
  double t1185;
  double t1465;
  double t4120;
  double t2166;
  double t4115;
  double t4117;
  double t4119;
  double t4121;
  double t4128;
  double t4131;
  double t4153;
  double t4167;
  double t4173;
  double t4174;
  double t4176;
  double t4184;
  double t4189;
  double t4194;
  double t4197;
  double t4203;
  double t4221;
  double t4232;
  double t4235;
  double t4237;
  double t4243;
  double t4246;
  double t4248;
  double t4262;
  double t4263;
  double t4266;
  double t4282;
  double t4156;
  double t4157;
  double t4159;
  double t4201;
  double t4213;
  double t4215;
  double t4275;
  double t4277;
  double t869;
  double t4283;
  double t4289;
  double t4290;
  double t4291;
  double t4295;
  double t4296;
  double t4298;
  double t4302;
  double t4310;
  double t4279;
  double t4306;
  double t4308;
  double t839;
  double t4311;
  double t4328;
  double t4337;
  double t500;
  double t4403;
  double t4404;
  double t4407;
  double t4411;
  double t4415;
  double t4417;
  double t4424;
  double t4425;
  double t4428;
  double t4410;
  double t4423;
  double t4433;
  double t4434;
  double t4439;
  double t4440;
  double t4442;
  double t4444;
  double t4350;
  double t4354;
  double t4437;
  double t4445;
  double t4446;
  double t4453;
  double t4455;
  double t4457;
  double t4366;
  double t4367;
  double t4370;
  double t4373;
  double t4374;
  double t4380;
  double t4381;
  double t4503;
  double t4505;
  double t4506;
  double t4509;
  double t4511;
  double t4513;
  double t4515;
  double t4517;
  double t4518;
  double t4507;
  double t4514;
  double t4519;
  double t4520;
  double t4523;
  double t4524;
  double t4525;
  double t4529;
  double t4522;
  double t4530;
  double t4537;
  double t4544;
  double t4545;
  double t4548;
  double t4309;
  double t4338;
  double t4344;
  double t4357;
  double t4358;
  double t4359;
  double t4363;
  double t4372;
  double t4377;
  double t4382;
  double t4387;
  double t4396;
  double t4397;
  double t4447;
  double t4458;
  double t4460;
  double t4470;
  double t4472;
  double t4473;
  double t4474;
  double t4475;
  double t4476;
  double t4478;
  double t4481;
  double t4486;
  double t4497;
  double t4542;
  double t4549;
  double t4550;
  double t4553;
  double t4555;
  double t4556;
  double t4557;
  double t4564;
  double t4565;
  double t4567;
  double t4568;
  double t4570;
  double t4571;
  double t4624;
  double t4626;
  double t4627;
  double t4628;
  double t4629;
  double t4630;
  double t4633;
  double t4635;
  double t4636;
  double t4639;
  double t4640;
  double t4641;
  double t4642;
  double t4647;
  double t4648;
  double t4650;
  double t4652;
  double t4653;
  double t4654;
  double t4617;
  double t4619;
  double t4620;
  double t4631;
  double t4646;
  double t4657;
  double t4658;
  double t4662;
  double t4663;
  double t4665;
  double t4667;
  double t4668;
  double t4669;
  double t4670;
  double t4609;
  double t4612;
  double t4614;
  double t4659;
  double t4671;
  double t4672;
  double t4674;
  double t4675;
  double t4676;
  double t4677;
  double t4681;
  double t4682;
  double t4683;
  double t4685;
  double t4686;
  double t4687;
  double t4688;
  double t4690;
  double t4692;
  double t4694;
  double t4601;
  double t4602;
  double t4603;
  double t4604;
  double t4606;
  double t4607;
  double t4673;
  double t4678;
  double t4679;
  double t4684;
  double t4695;
  double t4697;
  double t4702;
  double t4703;
  double t4704;
  double t4596;
  double t4597;
  double t4599;
  double t4709;
  double t4680;
  double t4722;
  double t4716;
  double t4735;
  double t4730;
  double t4747;
  double t4742;
  double t4761;
  double t4755;
  double t4770;
  double t4766;
  double t4861;
  double t4862;
  double t4863;
  double t4864;
  double t4866;
  double t4867;
  double t4868;
  double t4869;
  double t4884;
  double t4885;
  double t4886;
  double t4887;
  double t4891;
  double t4892;
  double t4894;
  double t4895;
  double t4914;
  double t4916;
  double t4917;
  double t4918;
  double t4909;
  double t4910;
  double t4911;
  double t4912;
  t1498 = Cos(var1[8]);
  t1713 = -1.*t1498;
  t1889 = 1. + t1713;
  t913 = Cos(var1[7]);
  t1185 = -1.*t913;
  t1465 = 1. + t1185;
  t4120 = Sin(var1[8]);
  t2166 = -0.8656776547239999*t1889;
  t4115 = 1. + t2166;
  t4117 = -0.657905*t4115;
  t4119 = 0.0883716288660118*t1889;
  t4121 = -0.930418*t4120;
  t4128 = 0. + t4121;
  t4131 = 0.707107*t4128;
  t4153 = t4117 + t4119 + t4131;
  t4167 = -0.134322983001*t1889;
  t4173 = 1. + t4167;
  t4174 = 0.259155*t4173;
  t4176 = -0.22434503092393926*t1889;
  t4184 = 0.366501*t4120;
  t4189 = 0. + t4184;
  t4194 = 0.707107*t4189;
  t4197 = t4174 + t4176 + t4194;
  t4203 = Sin(var1[7]);
  t4221 = -1.000000637725*t1889;
  t4232 = 1. + t4221;
  t4235 = 0.707107*t4232;
  t4237 = -0.366501*t4120;
  t4243 = 0. + t4237;
  t4246 = 0.259155*t4243;
  t4248 = 0.930418*t4120;
  t4262 = 0. + t4248;
  t4263 = -0.657905*t4262;
  t4266 = t4235 + t4246 + t4263;
  t4282 = Cos(var1[6]);
  t4156 = -0.340999127418*t1465*t4153;
  t4157 = -0.8656776547239999*t1465;
  t4159 = 1. + t4157;
  t4201 = t4159*t4197;
  t4213 = -0.930418*t4203;
  t4215 = 0. + t4213;
  t4275 = t4215*t4266;
  t4277 = 0. + t4156 + t4201 + t4275;
  t869 = Sin(var1[6]);
  t4283 = -0.134322983001*t1465;
  t4289 = 1. + t4283;
  t4290 = t4289*t4153;
  t4291 = -0.340999127418*t1465*t4197;
  t4295 = -0.366501*t4203;
  t4296 = 0. + t4295;
  t4298 = t4296*t4266;
  t4302 = 0. + t4290 + t4291 + t4298;
  t4310 = Cos(var1[5]);
  t4279 = -1.*t869*t4277;
  t4306 = t4282*t4302;
  t4308 = 0. + t4279 + t4306;
  t839 = Sin(var1[5]);
  t4311 = t4282*t4277;
  t4328 = t869*t4302;
  t4337 = 0. + t4311 + t4328;
  t500 = Sin(var1[3]);
  t4403 = 0.657905*t4115;
  t4404 = -0.0883716288660118*t1889;
  t4407 = t4403 + t4404 + t4131;
  t4411 = -0.259155*t4173;
  t4415 = 0.22434503092393926*t1889;
  t4417 = t4411 + t4415 + t4194;
  t4424 = -0.259155*t4243;
  t4425 = 0.657905*t4262;
  t4428 = t4235 + t4424 + t4425;
  t4410 = -0.340999127418*t1465*t4407;
  t4423 = t4159*t4417;
  t4433 = t4215*t4428;
  t4434 = 0. + t4410 + t4423 + t4433;
  t4439 = t4289*t4407;
  t4440 = -0.340999127418*t1465*t4417;
  t4442 = t4296*t4428;
  t4444 = 0. + t4439 + t4440 + t4442;
  t4350 = Cos(var1[3]);
  t4354 = Sin(var1[4]);
  t4437 = -1.*t869*t4434;
  t4445 = t4282*t4444;
  t4446 = 0. + t4437 + t4445;
  t4453 = t4282*t4434;
  t4455 = t869*t4444;
  t4457 = 0. + t4453 + t4455;
  t4366 = Cos(var1[4]);
  t4367 = 0.366501*t4203;
  t4370 = 0. + t4367;
  t4373 = 0.930418*t4203;
  t4374 = 0. + t4373;
  t4380 = -1.000000637725*t1465;
  t4381 = 1. + t4380;
  t4503 = -0.366501*t4115;
  t4505 = -0.3172717261340007*t1889;
  t4506 = t4503 + t4505;
  t4509 = -0.930418*t4173;
  t4511 = -0.12497652119782442*t1889;
  t4513 = t4509 + t4511;
  t4515 = -0.930418*t4243;
  t4517 = -0.366501*t4262;
  t4518 = t4515 + t4517;
  t4507 = -0.340999127418*t1465*t4506;
  t4514 = t4159*t4513;
  t4519 = t4215*t4518;
  t4520 = 0. + t4507 + t4514 + t4519;
  t4523 = t4289*t4506;
  t4524 = -0.340999127418*t1465*t4513;
  t4525 = t4296*t4518;
  t4529 = 0. + t4523 + t4524 + t4525;
  t4522 = -1.*t869*t4520;
  t4530 = t4282*t4529;
  t4537 = 0. + t4522 + t4530;
  t4544 = t4282*t4520;
  t4545 = t869*t4529;
  t4548 = 0. + t4544 + t4545;
  t4309 = -1.*t839*t4308;
  t4338 = t4310*t4337;
  t4344 = 0. + t4309 + t4338;
  t4357 = t4310*t4308;
  t4358 = t839*t4337;
  t4359 = 0. + t4357 + t4358;
  t4363 = t4354*t4359;
  t4372 = t4370*t4153;
  t4377 = t4374*t4197;
  t4382 = t4381*t4266;
  t4387 = 0. + t4372 + t4377 + t4382;
  t4396 = t4366*t4387;
  t4397 = 0. + t4363 + t4396;
  t4447 = -1.*t839*t4446;
  t4458 = t4310*t4457;
  t4460 = 0. + t4447 + t4458;
  t4470 = t4310*t4446;
  t4472 = t839*t4457;
  t4473 = 0. + t4470 + t4472;
  t4474 = t4354*t4473;
  t4475 = t4370*t4407;
  t4476 = t4374*t4417;
  t4478 = t4381*t4428;
  t4481 = 0. + t4475 + t4476 + t4478;
  t4486 = t4366*t4481;
  t4497 = 0. + t4474 + t4486;
  t4542 = -1.*t839*t4537;
  t4549 = t4310*t4548;
  t4550 = 0. + t4542 + t4549;
  t4553 = t4310*t4537;
  t4555 = t839*t4548;
  t4556 = 0. + t4553 + t4555;
  t4557 = t4354*t4556;
  t4564 = t4506*t4370;
  t4565 = t4513*t4374;
  t4567 = t4381*t4518;
  t4568 = 0. + t4564 + t4565 + t4567;
  t4570 = t4366*t4568;
  t4571 = 0. + t4557 + t4570;
  t4624 = 3.2909349868922137e-7*var1[8];
  t4626 = 0.138152*t4173;
  t4627 = 0.01855699127121286*t1889;
  t4628 = -0.045000372235*t4243;
  t4629 = -0.045*t4189;
  t4630 = t4624 + t4626 + t4627 + t4628 + t4629;
  t4633 = -0.045*t4232;
  t4635 = -0.04500040093286238*t1889;
  t4636 = -0.141285834136*t4128;
  t4639 = 0.138152*t4243;
  t4640 = 0.055653909852*t4189;
  t4641 = -0.108789*t4262;
  t4642 = 0. + t4633 + t4635 + t4636 + t4639 + t4640 + t4641;
  t4647 = 1.296332362046933e-7*var1[8];
  t4648 = -0.108789*t4115;
  t4650 = -0.09417621278645702*t1889;
  t4652 = -0.045*t4128;
  t4653 = -0.045000372235*t4262;
  t4654 = t4647 + t4648 + t4650 + t4652 + t4653;
  t4617 = -3.2909349868922137e-7*var1[7];
  t4619 = 0.03103092645718495*t1465;
  t4620 = -0.045000372235*t4370;
  t4631 = -0.340999127418*t1465*t4630;
  t4646 = t4296*t4642;
  t4657 = t4289*t4654;
  t4658 = t4617 + t4619 + t4620 + t4631 + t4646 + t4657;
  t4662 = 1.296332362046933e-7*var1[7];
  t4663 = 0.07877668146182712*t1465;
  t4665 = -0.045000372235*t4374;
  t4667 = t4159*t4630;
  t4668 = t4215*t4642;
  t4669 = -0.340999127418*t1465*t4654;
  t4670 = t4662 + t4663 + t4665 + t4667 + t4668 + t4669;
  t4609 = -1.*t4282;
  t4612 = 1. + t4609;
  t4614 = 0.091*t4612;
  t4659 = t869*t4658;
  t4671 = t4282*t4670;
  t4672 = 0. + t4614 + t4659 + t4671;
  t4674 = 0.091*t869;
  t4675 = t4282*t4658;
  t4676 = -1.*t869*t4670;
  t4677 = 0. + t4674 + t4675 + t4676;
  t4681 = t839*t4672;
  t4682 = t4310*t4677;
  t4683 = 0. + t4681 + t4682;
  t4685 = -0.04500040093286238*t1465;
  t4686 = 0.07877663122399998*t4215;
  t4687 = 0.031030906668*t4296;
  t4688 = t4374*t4630;
  t4690 = t4381*t4642;
  t4692 = t4370*t4654;
  t4694 = 0. + t4685 + t4686 + t4687 + t4688 + t4690 + t4692;
  t4601 = t4366*t4473;
  t4602 = -1.*t4354*t4481;
  t4603 = 0. + t4601 + t4602;
  t4604 = t4366*t4556;
  t4606 = -1.*t4354*t4568;
  t4607 = 0. + t4604 + t4606;
  t4673 = t4310*t4672;
  t4678 = -1.*t839*t4677;
  t4679 = 0. + t4673 + t4678;
  t4684 = t4354*t4683;
  t4695 = t4366*t4694;
  t4697 = 0. + t4684 + t4695;
  t4702 = t4366*t4683;
  t4703 = -1.*t4354*t4694;
  t4704 = 0. + t4702 + t4703;
  t4596 = t4366*t4359;
  t4597 = -1.*t4354*t4387;
  t4599 = 0. + t4596 + t4597;
  t4709 = t4550*t4679;
  t4680 = -1.*t4679*t4460;
  t4722 = t4344*t4679;
  t4716 = -1.*t4550*t4679;
  t4735 = -1.*t4344*t4679;
  t4730 = t4679*t4460;
  t4747 = -1.*t4694*t4481;
  t4742 = t4568*t4694;
  t4761 = -1.*t4568*t4694;
  t4755 = t4387*t4694;
  t4770 = t4694*t4481;
  t4766 = -1.*t4387*t4694;
  t4861 = -1.*t4630*t4417;
  t4862 = -1.*t4407*t4654;
  t4863 = -1.*t4642*t4428;
  t4864 = t4861 + t4862 + t4863;
  t4866 = t4513*t4630;
  t4867 = t4518*t4642;
  t4868 = t4506*t4654;
  t4869 = t4866 + t4867 + t4868;
  t4884 = t4630*t4197;
  t4885 = t4266*t4642;
  t4886 = t4153*t4654;
  t4887 = t4884 + t4885 + t4886;
  t4891 = -1.*t4513*t4630;
  t4892 = -1.*t4518*t4642;
  t4894 = -1.*t4506*t4654;
  t4895 = t4891 + t4892 + t4894;
  t4914 = t4630*t4417;
  t4916 = t4407*t4654;
  t4917 = t4642*t4428;
  t4918 = t4914 + t4916 + t4917;
  t4909 = -1.*t4630*t4197;
  t4910 = -1.*t4266*t4642;
  t4911 = -1.*t4153*t4654;
  t4912 = t4909 + t4910 + t4911;
  p_output1[0]=0. + t4350*t4397 - 1.*t4344*t500;
  p_output1[1]=0. + t4350*t4497 - 1.*t4460*t500;
  p_output1[2]=0. + t4350*t4571 - 1.*t4550*t500;
  p_output1[3]=0;
  p_output1[4]=0;
  p_output1[5]=0;
  p_output1[6]=0. + t4344*t4350 + t4397*t500;
  p_output1[7]=0. + t4350*t4460 + t4497*t500;
  p_output1[8]=0. + t4350*t4550 + t4571*t500;
  p_output1[9]=0;
  p_output1[10]=0;
  p_output1[11]=0;
  p_output1[12]=t4599;
  p_output1[13]=t4603;
  p_output1[14]=t4607;
  p_output1[15]=0;
  p_output1[16]=0;
  p_output1[17]=0;
  p_output1[18]=t4607*(t4680 - 1.*t4497*t4697 - 1.*t4603*t4704) + t4603*(t4571*t4697 + t4607*t4704 + t4709);
  p_output1[19]=t4599*(-1.*t4571*t4697 - 1.*t4607*t4704 + t4716) + t4607*(t4397*t4697 + t4599*t4704 + t4722);
  p_output1[20]=t4599*(t4497*t4697 + t4603*t4704 + t4730) + t4603*(-1.*t4397*t4697 - 1.*t4599*t4704 + t4735);
  p_output1[21]=t4599;
  p_output1[22]=t4603;
  p_output1[23]=t4607;
  p_output1[24]=t4460*(t4556*t4683 + t4709 + t4742) + t4550*(t4680 - 1.*t4473*t4683 + t4747);
  p_output1[25]=t4550*(t4359*t4683 + t4722 + t4755) + t4344*(-1.*t4556*t4683 + t4716 + t4761);
  p_output1[26]=t4460*(-1.*t4359*t4683 + t4735 + t4766) + t4344*(t4473*t4683 + t4730 + t4770);
  p_output1[27]=t4344;
  p_output1[28]=t4460;
  p_output1[29]=t4550;
  p_output1[30]=t4481*(t4548*t4672 + t4537*t4677 + t4742) + t4568*(-1.*t4457*t4672 - 1.*t4446*t4677 + t4747);
  p_output1[31]=t4568*(t4337*t4672 + t4308*t4677 + t4755) + t4387*(-1.*t4548*t4672 - 1.*t4537*t4677 + t4761);
  p_output1[32]=t4481*(-1.*t4337*t4672 - 1.*t4308*t4677 + t4766) + t4387*(t4457*t4672 + t4446*t4677 + t4770);
  p_output1[33]=t4387;
  p_output1[34]=t4481;
  p_output1[35]=t4568;
  p_output1[36]=0.091*t4302 - 1.*t4481*(t4529*t4658 + t4520*t4670 + t4742) - 1.*t4568*(-1.*t4444*t4658 - 1.*t4434*t4670 + t4747);
  p_output1[37]=0.091*t4444 - 1.*t4568*(t4302*t4658 + t4277*t4670 + t4755) - 1.*t4387*(-1.*t4529*t4658 - 1.*t4520*t4670 + t4761);
  p_output1[38]=0.091*t4529 - 1.*t4481*(-1.*t4302*t4658 - 1.*t4277*t4670 + t4766) - 1.*t4387*(t4444*t4658 + t4434*t4670 + t4770);
  p_output1[39]=0. - 1.*t4153*t4370 - 1.*t4197*t4374 - 1.*t4266*t4381;
  p_output1[40]=0. - 1.*t4370*t4407 - 1.*t4374*t4417 - 1.*t4381*t4428;
  p_output1[41]=0. - 1.*t4370*t4506 - 1.*t4374*t4513 - 1.*t4381*t4518;
  p_output1[42]=-0.016493*t4153 - 0.041869*t4197 - 0.084668*t4266 - 0.930418*(t4506*t4864 + t4407*t4869) + 0.366501*(t4513*t4864 + t4417*t4869);
  p_output1[43]=-0.016493*t4407 - 0.041869*t4417 - 0.084668*t4428 - 0.930418*(t4506*t4887 + t4153*t4895) + 0.366501*(t4513*t4887 + t4197*t4895);
  p_output1[44]=-0.016493*t4506 - 0.041869*t4513 - 0.084668*t4518 - 0.930418*(t4407*t4912 + t4153*t4918) + 0.366501*(t4417*t4912 + t4197*t4918);
  p_output1[45]=0. - 0.930418*t4153 + 0.366501*t4197;
  p_output1[46]=0. - 0.930418*t4407 + 0.366501*t4417;
  p_output1[47]=0. - 0.930418*t4506 + 0.366501*t4513;
  p_output1[48]=1.3540964544089817e-7;
  p_output1[49]=-3.2972577157508454e-7;
  p_output1[50]=-3.305646759411973e-7;
  p_output1[51]=-3.6361499999859603e-7;
  p_output1[52]=3.6361499999859603e-7;
  p_output1[53]=-1.000000637725;
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
  p_output1[120]=0;
  p_output1[121]=0;
  p_output1[122]=0;
  p_output1[123]=0;
  p_output1[124]=0;
  p_output1[125]=0;
  p_output1[126]=0;
  p_output1[127]=0;
  p_output1[128]=0;
  p_output1[129]=0;
  p_output1[130]=0;
  p_output1[131]=0;
  p_output1[132]=0;
  p_output1[133]=0;
  p_output1[134]=0;
  p_output1[135]=0;
  p_output1[136]=0;
  p_output1[137]=0;
  p_output1[138]=0;
  p_output1[139]=0;
  p_output1[140]=0;
  p_output1[141]=0;
  p_output1[142]=0;
  p_output1[143]=0;
  p_output1[144]=0;
  p_output1[145]=0;
  p_output1[146]=0;
  p_output1[147]=0;
  p_output1[148]=0;
  p_output1[149]=0;
  p_output1[150]=0;
  p_output1[151]=0;
  p_output1[152]=0;
  p_output1[153]=0;
  p_output1[154]=0;
  p_output1[155]=0;
  p_output1[156]=0;
  p_output1[157]=0;
  p_output1[158]=0;
  p_output1[159]=0;
  p_output1[160]=0;
  p_output1[161]=0;
  p_output1[162]=0;
  p_output1[163]=0;
  p_output1[164]=0;
  p_output1[165]=0;
  p_output1[166]=0;
  p_output1[167]=0;
}



void Jb_hip_flexion_left_src(double *p_output1, const double *var1)
{
  // Call Subroutines
  output1(p_output1, var1);

}
