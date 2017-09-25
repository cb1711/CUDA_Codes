#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <stdio.h>
#define MAX_ITER 1000
using namespace std;
short *buf;//Stores the values for the buffer
int colors[1000][3];//Colors for different values of buffer
double zoom=1;
double lft,rght,top,bottom;
void init(){
	glClearColor(1.0,1.0,1.0,1.0);
	glOrtho(-512,512,-512,512,-1.0,1.0);
}
/*GPU kernel to determine whether a point lies inside mandelbrot set or not*/
__global__ void mdbUtil(short* buf,double x_min,double x_max,double y_min,double y_max){
	int bid=blockIdx.x;
	int tid_local=threadIdx.x;
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	double x=x_min+((x_max-x_min)*tid_local)/1023.0;
	double y=y_min+((y_max-y_min)*bid)/1023.0;
	double x0=x;
	double y0=y;
	int iter=0;
	double xt;
	while(iter<MAX_ITER && x*x+y*y<4)
	{
		xt=x*x-y*y+x0;
		y=2*x*y+y0;
		x=xt;
		iter++;
	}
	buf[tid]=MAX_ITER-iter;
}
void gpuFuncCaller()
{
	mdbUtil<<<1024,1024>>>(buf,lft,rght,bottom,top);
	cudaDeviceSynchronize();
}
void plotter()
{
	gpuFuncCaller();

	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_POINTS);

	for(int i=0;i<1024;i++)
	{
		for(int j=0;j<1024;j++)
		{
			glColor3f(colors[0][buf[i*1024+j]]/1000.0,colors[1][buf[i*1024+j]]/1000.0,colors[2][buf[i*1024+j]]/1000.0);
			glVertex2s(j-512,i-512);
		}
	}

	glEnd();
	glColor3f(1.0,1.0,1.0);
	glBegin(GL_LINES);
	glVertex2s(0,-512);
	glVertex2s(0,512);
	glVertex2s(-512,0);
	glVertex2s(512,0);
	cout<<"Zoom "<<zoom<<"X"<<endl;
	glEnd();
	glFlush();
}
void mouse(int button,int state,int mx,int my)
{
	//Left click to zoom in and right click to zoom out
	if(button==GLUT_LEFT_BUTTON && state==GLUT_DOWN)
	{
		double one=1;
		my=500-my;
		double diff=rght-lft;
		double mxd=lft+((mx*one)/500.0)*diff;;
		rght=mxd+diff/4;
		lft=mxd-diff/4;
		diff=top-bottom;
		double myd=bottom+((my*one)/500.0)*diff;
		top=myd+diff/4;
		bottom=myd-diff/4;\
		zoom*=2;
	}
	else if(button==GLUT_RIGHT_BUTTON && state==GLUT_DOWN)
	{
		double one=1;
		my=500-my;
		double diff=rght-lft;
		double mxd=lft+((mx*one)/500.0)*diff;;
		rght=mxd+diff;
		lft=mxd-diff;
		diff=top-bottom;
		double myd=bottom+((my*one)/500.0)*diff;
		top=myd+diff;
		bottom=myd-diff;
		zoom/=2;
	}
	glutPostRedisplay();
}

int main(int argc,char** argv)
{
	cudaMallocManaged(&buf,1024*1024*sizeof(short));
	for(int i=0;i<1000;i++)
	{
		colors[0][i]=rand()%MAX_ITER+1;
		colors[1][i]=rand()%MAX_ITER+1;
		colors[2][i]=rand()%MAX_ITER+1;
	}
	colors[0][0]=0;
	colors[1][0]=0;
	colors[2][0]=0;
	lft=-2.5;
	rght=1.0;
	top=1.0;
	bottom=-1.0;
	cudaDeviceSynchronize();
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_SINGLE| GLUT_RGB);
    glutInitWindowSize(500,500);
    glutInitWindowPosition(100,100);
    glutCreateWindow("Window");
    init();
    glutMouseFunc(mouse);
    glutDisplayFunc(plotter);
    glutMainLoop();
    return 0;
}
