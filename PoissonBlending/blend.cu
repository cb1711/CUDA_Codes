#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <thrust/host_vector.h>
#define THREADS_PER_BLOCK 256

using namespace std;
__global__ void maskCompute(uchar4 *sourceImg,bool *mask,int cols,int rows)
{
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  int size=cols*rows;
  mask[id]=0;
  if(id<size)
  {
    if(sourceImg[id].x!=255 && sourceImg[id].y!=255 && sourceImg[id].z!=255)
    {
          mask[id]=1;
    }
  }
}
__global__ void borderMark(unsigned char *border,bool *mask,int cols,int rows)
{
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  //int size=cols*rows;
  int r=id/cols;
  int c=id%cols;
  int cnt=0;
  int x=r*cols+c;
  if(r>0 && r<rows && c>0 && c<cols){
    if(mask[x-cols]==1)
      cnt++;
    if(mask[x+cols]==1)
      cnt++;
    if(mask[x+1]==1)
      cnt++;
    if(mask[x-1]==1)
      cnt++;
  }
  if(cnt==0)
    border[id]=0;
  else if(cnt==4)
    border[id]=2;
  else
    border[id]=1;
}
__global__ void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id<numRows*numCols)
  {
    redChannel[id]=inputImageRGBA[id].x;
    blueChannel[id]=inputImageRGBA[id].z;
    greenChannel[id]=inputImageRGBA[id].y;
  }
}
__global__ void init_guess(unsigned char *channel,float* buf1,float* buf2,int sz)
{
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  unsigned char x=channel[id];
  if(id<sz)
  {
    buf1[id]=x;
    buf2[id]=x;
  }
}

__global__ void blender(unsigned char *sourceChannel,
                        unsigned char* destChannel,
                        float *pbuf,
                        float *cbuf,
                        int numCols,
                        int numRows,
                        unsigned char* border)
{
   int id=blockIdx.x*blockDim.x+threadIdx.x;
   float sum1=0,sum2=0;
   int row=id/numCols;
   int col=id%numCols;
   if(row<numRows && col<numCols)
   {
    if(border[id]==2)
    {
        unsigned char sc=sourceChannel[id];
        int n1=(row-1)*numCols+col;
        if(border[n1]==2)
          sum1+=pbuf[n1];
        else
          sum1+=destChannel[n1];
        sum2+=sc-sourceChannel[n1];

        n1=(row)*numCols+col-1;
        if(border[n1]==2)
          sum1+=pbuf[n1];
        else
          sum1+=destChannel[n1];
        sum2+=sc-sourceChannel[n1];
        
        n1=(row)*numCols+col+1;
        if(border[n1]==2)
          sum1+=pbuf[n1];
        else
          sum1+=destChannel[n1];
        sum2+=sc-sourceChannel[n1];

        n1=(row+1)*numCols+col;
        if(border[n1]==2)
          sum1+=pbuf[n1];
        else
          sum1+=destChannel[n1];
        sum2+=sc-sourceChannel[n1];
        float newVal=(sum1+sum2)/4.f;
        cbuf[id]=min(255.f, max(0.f, newVal));
   }
  }
}
__global__ void final_merge(uchar4 *blendedImg,
                            int numCols,
                            int numRows,
                            float *green,
                            float *blue,
                            float *red,
                            unsigned char *border)
{
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id<numRows*numCols)
  {
    if(border[id]==2)
    {
      blendedImg[id].x=red[id];
      blendedImg[id].z=blue[id];
      blendedImg[id].y=green[id];
    }
  }
} 

int main()
{
  int numRowsSource, numColsSource;
  numRowsSource=540;//Hard Coded values
  numColsSource=960;
  uchar4* h_sourceImg,*h_destImg,*h_blendedImg;  //IN
  int totalSize=numColsSource*numRowsSource;
  freopen("s_im.txt","r",stdin);
  h_sourceImg=(uchar4*)malloc(sizeof(uchar4)*totalSize);
  h_destImg=(uchar4*)malloc(sizeof(uchar4)*totalSize);
  h_blendedImg=(uchar4*)malloc(sizeof(uchar4)*totalSize);
  int t1;
  for(int i=0;i<numRowsSource;i++)
  {
    for(int j=0;j<numColsSource;j++)
    {
      cin>>t1;
      h_sourceImg[i*numColsSource+j].z=t1;
      cin>>t1;
      h_sourceImg[i*numColsSource+j].y=t1;
      cin>>t1;
      h_sourceImg[i*numColsSource+j].x=t1;
    }
  }
  freopen("d_im.txt","r",stdin);
  for(int i=0;i<numRowsSource;i++)
  {
    for(int j=0;j<numColsSource;j++)
    {
      cin>>t1;
      h_destImg[i*numColsSource+j].z=t1;
      cin>>t1;
      h_destImg[i*numColsSource+j].y=t1;
      cin>>t1;
      h_destImg[i*numColsSource+j].x=t1;
    }
  }
  cout<<"input taken"<<endl;
  //freopen("d_out.txt","w",stdout);
  uchar4 *d_sourceImg,*d_destImg;
  bool *d_mask;
  cudaStream_t rstream,gstream,bstream;
  cudaStreamCreate(&rstream);
  cudaStreamCreate(&gstream);
  cudaStreamCreate(&bstream);  
  unsigned char *source_r,*source_g,*source_b,*dest_r,*dest_b,*dest_g,*d_border;
  
  cudaMalloc((uchar4**)&d_sourceImg,sizeof(uchar4)*totalSize);
  cudaMalloc((uchar4**)&d_destImg,sizeof(uchar4)*totalSize);
  
  cudaMemcpy(d_destImg,h_destImg,sizeof(uchar4)*totalSize,cudaMemcpyHostToDevice);
  cudaMemcpy(d_sourceImg,h_sourceImg,sizeof(uchar4)*totalSize,cudaMemcpyHostToDevice);
  
  cudaMalloc((bool**)&d_mask,sizeof(bool)*totalSize);
  cudaMalloc((unsigned char**)&d_border,sizeof(char)*totalSize);
  
  int blocks=(totalSize)/THREADS_PER_BLOCK;

  //1
  maskCompute<<<blocks,THREADS_PER_BLOCK>>>(d_sourceImg,d_mask,numColsSource,numRowsSource);
  
  //2
  borderMark<<<blocks,THREADS_PER_BLOCK>>>(d_border,d_mask,numColsSource,numRowsSource);
  
  //3
  //Channels for source image
  cudaMalloc((unsigned char**)&source_r,sizeof(char)*totalSize);
  cudaMalloc((unsigned char**)&source_b,sizeof(char)*totalSize);
  cudaMalloc((unsigned char**)&source_g,sizeof(char)*totalSize);
  
  //channels for destination image
  cudaMalloc((unsigned char**)&dest_r,sizeof(char)*totalSize);
  cudaMalloc((unsigned char**)&dest_b,sizeof(char)*totalSize);
  cudaMalloc((unsigned char**)&dest_g,sizeof(char)*totalSize);

  separateChannels<<<blocks,THREADS_PER_BLOCK,0,rstream>>>(d_sourceImg,numRowsSource,numColsSource,source_r,source_g,source_b);
  separateChannels<<<blocks,THREADS_PER_BLOCK,0,gstream>>>(d_destImg,numRowsSource,numColsSource,dest_r,dest_g,dest_b);

  //4
  float *buf1_r,*buf1_g,*buf1_b,*buf2_r,*buf2_g,*buf2_b;
  cudaMalloc((float**)&buf1_r,sizeof(float)*totalSize);
  cudaMalloc((float**)&buf1_g,sizeof(float)*totalSize);
  cudaMalloc((float**)&buf1_b,sizeof(float)*totalSize);
  cudaMalloc((float**)&buf2_r,sizeof(float)*totalSize);
  cudaMalloc((float**)&buf2_g,sizeof(float)*totalSize);
  cudaMalloc((float**)&buf2_b,sizeof(float)*totalSize);
  


  init_guess<<<blocks,THREADS_PER_BLOCK,0,rstream>>>(source_r,buf1_r,buf2_r,totalSize);
  init_guess<<<blocks,THREADS_PER_BLOCK,0,gstream>>>(source_g,buf1_g,buf2_g,totalSize);
  init_guess<<<blocks,THREADS_PER_BLOCK,0,bstream>>>(source_b,buf1_b,buf2_b,totalSize);

  //5
  //Call the kernel 800 times for each color channel
    for(int i=0;i<400;i++)
  {
    blender<<<blocks,THREADS_PER_BLOCK,0,rstream>>>(source_r,dest_r,buf1_r,buf2_r,numColsSource,numRowsSource,d_border);
    blender<<<blocks,THREADS_PER_BLOCK,0,bstream>>>(source_b,dest_b,buf1_b,buf2_b,numColsSource,numRowsSource,d_border);
    blender<<<blocks,THREADS_PER_BLOCK,0,gstream>>>(source_g,dest_g,buf1_g,buf2_g,numColsSource,numRowsSource,d_border);
    blender<<<blocks,THREADS_PER_BLOCK,0,rstream>>>(source_r,dest_r,buf2_r,buf1_r,numColsSource,numRowsSource,d_border);
    blender<<<blocks,THREADS_PER_BLOCK,0,bstream>>>(source_b,dest_b,buf2_b,buf1_b,numColsSource,numRowsSource,d_border);
    blender<<<blocks,THREADS_PER_BLOCK,0,gstream>>>(source_g,dest_g,buf2_g,buf1_g,numColsSource,numRowsSource,d_border);
  }
  
  final_merge<<<blocks,THREADS_PER_BLOCK>>>(d_destImg,numColsSource,numRowsSource,buf1_g,buf1_b,buf1_r,d_border);
  cudaMemcpy(h_blendedImg,d_destImg,sizeof(uchar4)*totalSize,cudaMemcpyDeviceToHost);
  int sz=numColsSource*numRowsSource;
  freopen("b_out.txt","w",stdout);
  for(int i=0;i<sz;i++)
  {
    cout<<int(h_blendedImg[i].z)<<" "<<int(h_blendedImg[i].y)<<" "<<int(h_blendedImg[i].x)<<" ";
  }
  cudaDeviceReset();
  return 0;
}
