// This file is intended for external variable declarations so that OpenGL functions 
// have access to program variables without directly passing them.
#pragma once

#ifndef _global_h
#define _global_h

#include "grid_2d.h"
#include <GL/glew.h>

extern Grid *g;
extern unsigned int *image_data;		// the container for the normalized color field data
extern double *dev_hx;					// Now the global device pointer for field Hx
extern double *dev_chxh;				// Global device pointer for Chxh
extern double *dev_chxe;				// Same
extern double *dev_hy;					// Now the global device pointer for field Hy
extern double *dev_chyh;				// Same
extern double *dev_chye;				// Same
extern double *dev_ez;					// Now the global device pointer for field Ez
extern double *dev_cezh;				// Same
extern double *dev_ceze;				// Same
extern float *dev_ez_float;				// Copy of dev_ez but in single precision
extern float *dev_hx_float;				// Copy of dev_hx but in single precision
extern float *dev_hy_float;				// Copy of dev_hy but in single precision

// Note for all the externs declared below:  they have no location in memory until defined somewhere else (or here).  
// Extern <variable type> just declares the variable globally to the program, but it does not exist until
// it has been defined.
extern double dt;						// differential time operator
extern double dx;						// differential x-operator
extern double N_lambda;
extern float dy;						// differential y-operator
extern int nx;							// ?
extern int ny;							//	?
extern int number_of_cells;				// Total cells M*N
extern int number_of_cells_with_pads;
extern float domain_min_x;				// ?
extern float domain_min_y;				// ?
extern float domain_max_x;				// ?
extern float domain_max_y;				// ?
extern float global_min_field;			// calculated by find_min_max_on_gpu
extern float global_max_field;			// calculated by find_min_max_on_gpu
extern unsigned int rgb[];				// used in createImageOnGpu() in graphics.cpp
extern float* field_data;				// presumably the argument to createImageOnGpu() or something
extern unsigned int* dvimage_data;
extern float *dvminimum_field_value;			// Both of these are passed to the find-min-max-gpu functions
extern float *dvmaximum_field_value;		    // to get proper min/max field values for color-scaling
extern bool show_Ez;					// Used as a flag in visualization
extern int plotting_step;					// Used in IterationAndDisplay; every plotting_step steps arrays will 
// be displayed via OpenGL


extern GLuint pbo_destination;
extern struct cudaGraphicsResource *cuda_pbo_destination_resource;
extern GLuint cuda_result_texture;
extern int iGLUTWindowHandle;          // handle to the GLUT window
#endif