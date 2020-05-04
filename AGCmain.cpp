#include "AGammaCorrection.h"

int main()
{
	IplImage* img = cvLoadImage("C:/Users/54053/Desktop/InterBox/face.jpg");

	AGammaCorrection agamma;
	IplImage* hsv = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
	cvCvtColor(img, hsv, CV_RGB2HSV);

	IplImage* hue = cvCreateImage(cvGetSize(hsv), IPL_DEPTH_32F, 1);
	IplImage* sat = cvCreateImage(cvGetSize(hsv), IPL_DEPTH_32F, 1);
	IplImage* val = cvCreateImage(cvGetSize(hsv), IPL_DEPTH_32F, 1);

	agamma.AdaptiveGammaCorrection(img, hue, sat, val);
	IplImage* hsvF = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, img->nChannels);
	cvMerge(hue, sat, val, NULL, hsvF);
	cvConvert(hsvF,hsv);
	
	IplImage* temp = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
	cvCvtColor(hsv, temp, CV_HSV2BGR);
	cvSetZero(hsv);
	cvScaleAdd(temp, cvScalarAll(1), hsv, img);

	cvSaveImage("C:/Users/54053/Desktop/InterBox/final.jpg", img);

	return 0;
}