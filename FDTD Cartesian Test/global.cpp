#include "global.h"
#include <stdio.h>
#include <string.h>


float *dev_ez_float;
float *dev_hx_float;
float *dev_hy_float;
double *dev_ez;
double *dev_hx;
double *dev_hy;
double *dev_chxh;
double *dev_chxe;
double *dev_chyh;
double *dev_chye;
double *dev_cezh;
double *dev_ceze;
float *dvminimum_field_value;
float *dvmaximum_field_value;
Grid *g = new Grid;
int plotting_step;
double dx = 1e-2;
double dt;
double N_lambda;

