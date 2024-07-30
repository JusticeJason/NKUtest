#include<iostream>
#include<opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat img1;
	img1 = imread("test2.jpg");
	imshow("原图", img1);
	Rect rect(84, 84, 406, 318);
	Mat img2, bg, fg;
	grabCut(img1, img2, rect, bg, fg,1,GC_INIT_WITH_RECT);
	compare(img2, GC_PR_FGD, img2, CMP_EQ);
	imshow("img2", img2);
	Mat img3(img1.size(), CV_8UC3, Scalar(255, 255, 255));
	img1.copyTo(img3, img2);
	imshow("img3", img3);
	waitKey(0);
}
