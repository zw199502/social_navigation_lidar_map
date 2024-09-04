/*
 * Automatically Generated from Mathematica.
 * Mon 4 Jul 2022 20:44:24 GMT-04:00
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "dJs_hip_abduction_right_src.h"

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
  double t313;
  double t629;
  double t1220;
  double t1382;
  double t2815;
  double t2975;
  t313 = Cos(var1[3]);
  t629 = Sin(var1[3]);
  t1220 = Cos(var1[4]);
  t1382 = Sin(var1[4]);
  t2815 = Cos(var1[5]);
  t2975 = Sin(var1[5]);
  p_output1[0]=0;
  p_output1[1]=0;
  p_output1[2]=0;
  p_output1[3]=0;
  p_output1[4]=0;
  p_output1[5]=0;
  p_output1[6]=0;
  p_output1[7]=0;
  p_output1[8]=0;
  p_output1[9]=0;
  p_output1[10]=0;
  p_output1[11]=0;
  p_output1[12]=0;
  p_output1[13]=0;
  p_output1[14]=0;
  p_output1[15]=0;
  p_output1[16]=0;
  p_output1[17]=0;
  p_output1[18]=var2[1];
  p_output1[19]=-1.*var2[0];
  p_output1[20]=0;
  p_output1[21]=0;
  p_output1[22]=0;
  p_output1[23]=0;
  p_output1[24]=-1.*t313*var2[2] + t629*var1[2]*var2[3];
  p_output1[25]=-1.*t629*var2[2] - 1.*t313*var1[2]*var2[3];
  p_output1[26]=t313*var2[0] + t629*var2[1] + (-1.*t629*var1[0] + t313*var1[1])*var2[3];
  p_output1[27]=-1.*t313*var2[3];
  p_output1[28]=-1.*t629*var2[3];
  p_output1[29]=0;
  p_output1[30]=-1.*t1382*var2[1] - 1.*t1220*t629*var2[2] - 1.*t1220*t313*var1[2]*var2[3] + (-1.*t1220*var1[1] + t1382*t629*var1[2])*var2[4];
  p_output1[31]=t1382*var2[0] + t1220*t313*var2[2] - 1.*t1220*t629*var1[2]*var2[3] + (t1220*var1[0] - 1.*t1382*t313*var1[2])*var2[4];
  p_output1[32]=t1220*t629*var2[0] - 1.*t1220*t313*var2[1] + (t1220*t313*var1[0] + t1220*t629*var1[1])*var2[3] + (-1.*t1382*t629*var1[0] + t1382*t313*var1[1])*var2[4];
  p_output1[33]=-1.*t1220*t629*var2[3] - 1.*t1382*t313*var2[4];
  p_output1[34]=t1220*t313*var2[3] - 1.*t1382*t629*var2[4];
  p_output1[35]=-1.*t1220*var2[4];
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
  p_output1[102]=t1382*var2[1] + t1220*t629*var2[2] + (-0.091*(t2975*t313 - 1.*t1382*t2815*t629) + t1220*t313*var1[2])*var2[3] + (-0.091*t1220*t2815*t313 + t1220*var1[1] - 1.*t1382*t629*var1[2])*var2[4] - 0.091*(-1.*t1382*t2975*t313 + t2815*t629)*var2[5];
  p_output1[103]=-1.*t1382*var2[0] - 1.*t1220*t313*var2[2] + (-0.091*(t1382*t2815*t313 + t2975*t629) + t1220*t629*var1[2])*var2[3] + (-0.091*t1220*t2815*t629 - 1.*t1220*var1[0] + t1382*t313*var1[2])*var2[4] - 0.091*(-1.*t2815*t313 - 1.*t1382*t2975*t629)*var2[5];
  p_output1[104]=-1.*t1220*t629*var2[0] + t1220*t313*var2[1] + (-1.*t1220*t313*var1[0] - 1.*t1220*t629*var1[1])*var2[3] + (0.091*t1382*t2815 + t1382*t629*var1[0] - 1.*t1382*t313*var1[1])*var2[4] + 0.091*t1220*t2975*var2[5];
  p_output1[105]=t1220*t629*var2[3] + t1382*t313*var2[4];
  p_output1[106]=-1.*t1220*t313*var2[3] + t1382*t629*var2[4];
  p_output1[107]=t1220*var2[4];
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



void dJs_hip_abduction_right_src(double *p_output1, const double *var1,const double *var2)
{
  // Call Subroutines
  output1(p_output1, var1, var2);

}
