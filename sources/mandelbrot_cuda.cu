#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <gl/freeglut.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>

#define WIDTH 800
#define HEIGHT 800
#define ITERATIONS 5000


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

typedef struct rgb_color {
	float r;
	float g;
	float b;
}color;

color colors[16];
color pixels[WIDTH*HEIGHT];

double real_min = -2, real_max = 1;
double img_min = -1.5, img_max = 1.5;
int MAX_ITER = ITERATIONS; int iter_step = 10; 

int window;

int size = HEIGHT * WIDTH * sizeof(int); //device aloc size
int *iters = (int *)malloc(size);
int *iters_d;

int arr_size = WIDTH * HEIGHT;
int num_threads = 512;
int block_size = 1250;
double t_start, t_end;

void initColors() {
	colors[0].r = 66; colors[0].g = 30; colors[0].b = 15;
	colors[1].r = 25; colors[1].g = 7; colors[1].b = 26;
	colors[2].r = 9; colors[2].g = 1; colors[2].b = 47;
	colors[3].r = 4; colors[3].g = 4; colors[3].b = 73;
	colors[4].r = 0; colors[4].g = 7; colors[4].b = 100;
	colors[5].r = 12; colors[5].g = 44; colors[5].b = 138;
	colors[6].r = 24; colors[6].g = 82; colors[6].b = 177;
	colors[7].r = 57; colors[7].g = 125; colors[7].b = 209;
	colors[8].r = 134; colors[8].g = 181; colors[8].b = 229;
	colors[9].r = 211; colors[9].g = 236; colors[9].b = 248;
	colors[10].r = 241; colors[10].g = 233; colors[10].b = 191;
	colors[11].r = 248; colors[11].g = 201; colors[11].b = 95;
	colors[12].r = 255; colors[12].g = 170; colors[12].b = 0;
	colors[13].r = 204; colors[13].g = 128; colors[13].b = 0;
	colors[14].r = 153; colors[14].g = 87; colors[14].b = 0;
	colors[15].r = 106; colors[15].g = 52; colors[15].b = 3;
}

void setUpColor() {
	const int NX = WIDTH;
	const int NY = HEIGHT;
	int i, j, VAL;

	for (i = 0; i < NX; i++) {
		for (j = 0; j < NY; j++) {
			VAL = iters[i*WIDTH + j];

			if (VAL < MAX_ITER && VAL > 0) {
				int cid = VAL % 16;

				pixels[i + j*HEIGHT].r = colors[cid].r / 255;
				pixels[i + j*HEIGHT].g = colors[cid].g / 255;
				pixels[i + j*HEIGHT].b = colors[cid].b / 255;
			}
			else {
				pixels[i + j*HEIGHT].r = 0;
				pixels[i + j*HEIGHT].g = 0;
				pixels[i + j*HEIGHT].b = 0;
			}
		}
	}
}

__global__ void mandelbrotset(int *iter_arr, double xmin, double xmax, double ymin, double ymax, int w, int h, int max_iters, int N) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index >= N)
		return;

	const int NX = w;
	const int NY = h;

	double dx = (xmax - xmin) / NX;
	double dy = (ymax - ymin) / NY;

	int i = index / w;
	int j = index % h;

	double x = xmin + i*dx;
	double y = ymin + j*dy;

	double c_i = y;
	double c_r = x;
	double z_r = 0, z_i = 0;
	int n;

	for ( n = 0; n < max_iters; ++n) {
		double z_r2 = z_r*z_r, z_i2 = z_i*z_i;
		if (z_r2 + z_i2 > 4) {
			break;
		}
		z_i = 2 * z_r*z_i + c_i;
		z_r = z_r2 - z_i2 + c_r;
	}

	iter_arr[index] = n;
}

void SpecialKeys(int key, int x, int y) {

	double xStep = fabs(real_min - real_max) * 0.08;
	double yStep = fabs(img_min - img_max) * 0.08;


	switch (key) {
	case GLUT_KEY_LEFT:
		real_min -= xStep;
		real_max -= xStep;
		break;

	case GLUT_KEY_RIGHT:
		real_min += xStep;
		real_max += xStep;
		break;

	case GLUT_KEY_UP:
		img_min += yStep;
		img_max += yStep;
		break;

	case GLUT_KEY_DOWN:
		img_min -= yStep;
		img_max -= yStep;
		break;
	}
	glutPostRedisplay();
}

void KeyB(unsigned char key, int x, int y) {
	double xStep = fabs(real_min - real_max) * 0.08;
	double yStep = fabs(img_min - img_max) * 0.08;


	switch (key) {
	case '+':
		real_min += xStep * 2;
		real_max -= xStep * 2;
		img_min += yStep * 2;
		img_max -= yStep * 2;
		break;

	case '-':
		real_min -= xStep * 2;
		real_max += xStep * 2;
		img_min -= yStep * 2;
		img_max += yStep * 2;
		break;

	case 27: // Escape key
		delete(iters);
		glutDestroyWindow(window);
		cudaFree(iters_d);
		exit(0);
		break;

	case 'r':
		MAX_ITER += iter_step;
		printf("Iterations: %d -> %d\n", MAX_ITER - 10, MAX_ITER);
		break;

	case 't':
		MAX_ITER -= iter_step;
		printf("Iterations: %d -> %d\n", MAX_ITER + 10, MAX_ITER);
		break;
	}
	glutPostRedisplay();
}

void onDisplay() {
	t_start = omp_get_wtime();

	mandelbrotset<<<(block_size*num_threads + num_threads - 1) / num_threads,num_threads>>>(iters_d, real_min, real_max, img_min, img_max, WIDTH, HEIGHT, MAX_ITER, arr_size);
	
	gpuErrchk(cudaMemcpy(iters, iters_d, size, cudaMemcpyDeviceToHost));
	
	t_end = omp_get_wtime();
	printf("Render time: %lf\n", t_end - t_start);
	
	setUpColor();

	glClearColor(1, 1, 1, 0);
	glClear(GL_COLOR_BUFFER_BIT);

	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_FLOAT, pixels);
	glutSwapBuffers();
}

void Init() {
	glutInitWindowSize(WIDTH, HEIGHT);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowPosition(100, 100);
	window = glutCreateWindow("Mandelbrotset");

	glutKeyboardFunc(KeyB);
	glutSpecialFunc(SpecialKeys);
	glViewport(0, 0, HEIGHT, WIDTH);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, WIDTH, 0, HEIGHT);


	gpuErrchk(cudaMalloc((void **)&iters_d, size));

	initColors();
}

int main(int argc, char** argv) {
		glutInit(&argc, argv);
		Init();

		glutDisplayFunc(onDisplay);
		glutMainLoop();

	return 0;
}
