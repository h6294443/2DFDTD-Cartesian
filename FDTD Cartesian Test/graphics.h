#include "parameters.h"
#include <GL/glew.h>
#include <GL/GL.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "grid_2d.h"
#include "global.h"
#include "cuda_functions.h"

// Forward declarations
void Cleanup(int iExitCode);

// GL functionality
bool initializeGL(int argc, char** argv);

void createPixelBufferObject(GLuint* pbo, struct cudaGraphicsResource **pbo_resource);
void deletePBO(GLuint* pbo);

void createTextureDestination(GLuint* cuda_result_texture, unsigned int size_x, unsigned int size_y);
void deleteTexture(GLuint* tex);

// rendering callbacks
void idle();
void keyboard(unsigned char key, int x, int y);
void reshape(int w, int h);

void setImageAndWindowSize();			// Self-explanatory.  Why am I writing this.
bool deallocateArrays();				// used to free() all host arrays
void runIterationsAndDisplay();			// Used as display callback function in the glutMainLoop()
bool runFdtdWithFieldDisplay(int argc, char** argv);

// Function prototypes for stuff in cuda-opengl_functions.cu
__global__ void find_min_and_max_on_gpu(int nblocks, float* field,
										float* minimum_field_value,
										float* maximum_field_value);

void createColormapOnGpu();
__global__ void create_image_on_gpu(unsigned int* g_odata, float* Ez, int M, float minval, float maxval);
void createImageOnGpu(unsigned int* g_odata);

void mouse(int button, int state, int x, int y);
void motion(int x, int y);


