#include <iostream>
#include <stdio.h>
#include <opencv/cv.h>
#include "opencv2/highgui/highgui.hpp"
using namespace std;
using namespace cv;
//Blur a grayscale image using
int main()
{
    freopen("b_out.txt","r",stdin);

	Mat img1;
	img1=imread("s2.png");
    resize(img1,img1,cvSize(960,540));
	int c=img1.cols;
	int r=img1.rows;
	//cout<<c<<" "<<r<<endl;
	int sz=r*c;
	int *img;
	for(int i=0;i<r;i++)
		for(int j=0;j<c;j++)
        {
            int x;
            cin>>x;
            img1.at<Vec3b>(i,j)[0]=x;
            cin>>x;
            img1.at<Vec3b>(i,j)[1]=x;
            cin>>x;
            img1.at<Vec3b>(i,j)[2]=x;
        }
    imshow("im",img1);
    imwrite("out.jpg",img1);
    waitKey(5000);
	return 0;
}


