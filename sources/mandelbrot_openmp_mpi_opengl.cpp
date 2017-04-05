#include "mpi.h"
#include <GL/gl.h>
#include <GL/glut.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>

#define WIDTH 800
#define HEIGHT 800
#define NUM_THREADS 8
#define OPENMP
#define MPI
#define ITERATIONS 5000

typedef struct rgb_color {
       float r;
       float g;
       float b;
}color;

color colors[16];
color pixels[WIDTH*HEIGHT];
int *iters = new int[WIDTH*HEIGHT];

double real_min = -2, real_max = 1;
double img_min = -1.5, img_max = 1.5;
int MAX_ITER = ITERATIONS; int iter_step = 10; int window;
int world_rank = 0; int world_size = 1; int chunks = WIDTH*HEIGHT;


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

    for( i = 0; i < NX; i++) {
        for( j = 0; j < NY; j++) {
            VAL = iters[i*WIDTH + j];

            if(VAL < MAX_ITER && VAL > 0) {
               int cid = VAL % 16;

               pixels[i + j*HEIGHT].r = colors[cid].r/255;
               pixels[i + j*HEIGHT].g = colors[cid].g/255;
               pixels[i + j*HEIGHT].b = colors[cid].b/255;
            }
            else {
               pixels[i + j*HEIGHT].r = 0;
               pixels[i + j*HEIGHT].g = 0;
               pixels[i + j*HEIGHT].b = 0;
            }
       }
   }
}

int Mandelbrot_Member(double x, double y) {
    double c_i = y;
    double c_r = x;
    double z_r = 0, z_i = 0;
    int n;

    for(n=0; n<MAX_ITER; ++n) {
        double z_r2 = z_r*z_r, z_i2 = z_i*z_i;
        if(z_r2 + z_i2 > 4) {
            break;
        }
        z_i = 2*z_r*z_i + c_i;
        z_r = z_r2 - z_i2 + c_r;
    }

    return n;

}

void mandelbrotset(double xmin, double xmax, double ymin, double ymax) {

    int *iters_s = new int[chunks];
    const int NX = WIDTH;
    const int NY = HEIGHT;

    double dx = (xmax - xmin)/NX;
    double dy = (ymax - ymin)/NY;

    double start, endt;
    double x, y;
    int i, j;

    const int starti = world_rank * WIDTH/world_size;
    const int rows  = WIDTH/world_size + world_rank * (WIDTH/world_size);

    #ifdef MPI
    if (world_rank == 0) {
        double data[6] = {xmin, xmax, ymin, ymax, 0, (double) MAX_ITER};
        MPI_Bcast(data, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    #endif

    start = omp_get_wtime();

    #ifdef OPENMP
        omp_set_dynamic(0);
    #pragma omp parallel for num_threads(NUM_THREADS) \
                            default(shared) private(j, x, y, i)
    #endif
    for(/*i = 0*/ i = starti; i < rows /*NX*/; i++) {
        for(j = 0; j < NY; j++) {
            x = xmin + i*dx;
            y = ymin + j*dy;

            if (world_rank == 0) {
                iters_s[(WIDTH/world_size-(rows - i)) * WIDTH + j] = Mandelbrot_Member(x,y);
            }
            else {
                iters[(WIDTH/world_size-(rows - i))/*i*/* WIDTH + j] = Mandelbrot_Member(x,y);
            }
        }
    }
    //end of parallel region



    #ifdef MPI
    if (world_rank == 0) {
        MPI_Gather(iters_s, chunks/world_size, MPI_INT,  iters, chunks/world_size, MPI_INT, 0, MPI_COMM_WORLD );
        endt = omp_get_wtime();
        printf("Render-phase time: %lf\n", (endt - start));
        setUpColor();

    } else {
        MPI_Gather(iters, chunks/world_size, MPI_INT,  NULL, chunks/world_size, MPI_INT, 0, MPI_COMM_WORLD );
    }
    #else
    endt = omp_get_wtime();
        printf("Render-phase time: %lf\n", (endt - start));
        iters = iters_s;
        setUpColor();
    #endif
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
    #ifdef MPI
    double end[6] = {1, 1, 1, 1, 1, (double) MAX_ITER};
    #endif

    switch(key) {
        case '+':
            real_min += xStep*2;
            real_max -= xStep*2;
            img_min += yStep*2;
            img_max -= yStep*2;
        break;

        case '-':
            real_min -= xStep*2;
            real_max += xStep*2;
            img_min -= yStep*2;
            img_max += yStep*2;
        break;

        case 27: // Escape key
            delete(iters);
            glutDestroyWindow (window);
            #ifdef MPI
            MPI_Bcast(&end, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            #endif
            exit (0);
        break;

        case 'r':
             MAX_ITER += iter_step;
             printf("Iterations: %d -> %d\n", MAX_ITER - 10 ,MAX_ITER);
        break;

        case 't':
             MAX_ITER -= iter_step;
             printf("Iterations: %d -> %d\n", MAX_ITER + 10 ,MAX_ITER);
        break;
    }
    glutPostRedisplay();
}

void onDisplay() {
    mandelbrotset(real_min, real_max, img_min, img_max);

    glClearColor(1, 1, 1, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_FLOAT, pixels);
    glutSwapBuffers();
}

void Init() {
    glutInitWindowSize (WIDTH, HEIGHT);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition (100, 100);
    window = glutCreateWindow ("Mandelbrotset");

    glutKeyboardFunc(KeyB);
    glutSpecialFunc(SpecialKeys);
    glViewport(0, 0, HEIGHT, WIDTH);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, WIDTH, 0, HEIGHT);

    initColors();
}


int main(int argc, char** argv) {
    // Get the rank of the process
    #ifdef MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
    #endif
        glutInit(&argc, argv);
        Init ();

        glutDisplayFunc(onDisplay);
        glutMainLoop();
    #ifdef MPI
    }
    else {
            while (1) {
                double data[6];
                MPI_Bcast(data, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                if (data[4] == 1) break;

                MAX_ITER = (int) data[5];
                mandelbrotset(data[0], data[1], data[2], data[3]);
            }

    }
    MPI_Finalize();
    #endif
    return 0;
}
