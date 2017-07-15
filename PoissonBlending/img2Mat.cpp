#include <iostream>
#include <opencv/cv.h>
#include "opencv2/highgui/highgui.hpp"
using namespace std;
using namespace cv;
int main()
{
    string source="s2.png";
    string dest="d2.jpg";
    freopen("s_im.txt","w",stdout);
    Mat img1=imread(source);
    resize(img1,img1,cvSize(960,540));
    int x=img1.cols;
    int y=img1.rows;
    for(int i=0;i<y;i++)
    {
        for(int j=0;j<x;j++)
        {
            cout<<int(img1.at<Vec3b>(i,j)[0])<<" "<<int(img1.at<Vec3b>(i,j)[1])<<" "<<int(img1.at<Vec3b>(i,j)[2])<<" ";
        }
    }
    freopen("d_im.txt","w",stdout);
    Mat img2=imread(dest);
    resize(img2,img2,cvSize(960,540));
    x=img2.cols;
    y=img2.rows;

    for(int i=0;i<y;i++)
    {
        for(int j=0;j<x;j++)
        {
            cout<<int(img2.at<Vec3b>(i,j)[0])<<" "<<int(img2.at<Vec3b>(i,j)[1])<<" "<<int(img2.at<Vec3b>(i,j)[2])<<" ";
        }
    }
    return 0;
}

