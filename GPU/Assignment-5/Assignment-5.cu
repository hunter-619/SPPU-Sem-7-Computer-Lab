#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <glew.h>
#include <freeglut.h>

#include <math.h>


const int width = 800;
const int height = 600;
const int numVertices = 3;

float *vertices;

GLfloat colors[] = {
    1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f,
};

// CUDA function to update vertex data
__global__ void updateVertices(float* vertices, float time) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float angle = idx * 2 * 3.14159265359 / numVertices;
    float radius = 0.5;
    vertices[idx * 3] = radius * cos(angle + time);
    vertices[idx * 3 + 1] = radius * sin(angle + time);
    vertices[idx * 3 + 2] = 0.0f;
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, vertices);

    glEnableClientState(GL_COLOR_ARRAY); // Enable color array
    glColorPointer(3, GL_FLOAT, 0, colors);


    glDrawArrays(GL_TRIANGLES, 0, numVertices);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
}

void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void updateAnimation(int value) {
    // Call the CUDA kernel to update the vertex data
    updateVertices <<< 1, numVertices >>> (vertices, glutGet(GLUT_ELAPSED_TIME) / 1000.0);
    cudaDeviceSynchronize();

    glutPostRedisplay();
    glutTimerFunc(1, updateAnimation, 0); // Update every 16ms (approximately 60 FPS)
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(width, height);
    glutCreateWindow("OpenGL with CUDA");

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutTimerFunc(0, updateAnimation, 0); // Start animation loop

    glewInit();

    // Allocate memory for vertices
    cudaMallocManaged(&vertices, numVertices * 3 * sizeof(float));

    glutMainLoop();

    // Free CUDA memory
    cudaFree(vertices);

    return 0;
}
