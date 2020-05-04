#include "AGammaCorrection.h"

using namespace std;

const int m_sig1 = 15;
const int m_sig2 = 80;
const int m_sig3 = 250;
const int m_pValue = 2;
const double m_qValue = sqrt(2.0);

AGammaCorrection::AGammaCorrection()
{

}

AGammaCorrection::~AGammaCorrection()
{

}

void AGammaCorrection::AdaptiveGammaCorrection(IplImage *pImg, IplImage *pHue, IplImage *pSat, IplImage *pVal)
{

	int w = pImg->width;
	int h = pImg->height;

	IplImage *imgF = cvCreateImage(cvSize(w, h), IPL_DEPTH_32F, 3);
	IplImage *imgFN = cvCreateImage(cvSize(w, h), IPL_DEPTH_32F, 3);
	IplImage *scalar = cvCreateImage(cvSize(w, h), IPL_DEPTH_32F, 3);
	cvConvert(pImg, imgF);
	cvSet(scalar, cvScalarAll(1));
	cvDiv(imgF, scalar, imgFN);
	cvReleaseImage(&imgF);
	cvReleaseImage(&scalar);
	imgF = imgFN;

	cvSplit(imgF, pHue, pSat, pVal, NULL);

	int hsize = min(w, h);

	IplImage *gauss1 = cvCreateImage(cvSize(w, h), imgF->depth, 1);
	IplImage *gauss2 = cvCreateImage(cvSize(w, h), imgF->depth, 1);
	IplImage *gauss3 = cvCreateImage(cvSize(w, h), imgF->depth, 1);

	CvMat *F1 = cvCreateMat(hsize, hsize, CV_32F);
	CvMat *F2 = cvCreateMat(hsize, hsize, CV_32F);
	CvMat *F3 = cvCreateMat(hsize, hsize, CV_32F);

	GenerateGaussianKernel(hsize, m_sig1 / m_qValue, F1);
	GenerateGaussianKernel(hsize, m_sig2 / m_qValue, F2);
	GenerateGaussianKernel(hsize, m_sig3 / m_qValue, F3);

	//cvSaveImage("C:/Users/54053/Desktop/InterBox/val.jpg", val);

	cvFilter2D(pVal, gauss1, F1);
	cvFilter2D(pVal, gauss2, F2);
	cvFilter2D(pVal, gauss3, F3);

	//cvSaveImage("C:/Users/54053/Desktop/InterBox/gauss3.jpg", gauss3);

	IplImage *temp = cvCreateImage(cvSize(w, h), imgF->depth, 1);

	cvAdd(gauss2, gauss3, temp);
	cvAdd(gauss1, temp, gauss3);

	cvSubS(gauss3, cvScalar(1 / 3.0), gauss2);

	cvSaveImage("C:/Users/54053/Desktop/InterBox/gamma.jpg", gauss2);
	CvScalar meanScalar = cvAvg(gauss2);
	double m = meanScalar.val[0];
	//double m = cvMean(gauss2);


	for (int i = 0; i < gauss2->height; i++)
	{
		float *ptr = (float*)(gauss2->imageData + i * gauss2->widthStep);
		float *ptrVal = (float*)(pVal->imageData + i * pVal->widthStep);
		for (int j = 0; j < gauss2->width; j++)
		{
			float pownum = pow(m_pValue, (m - ptr[j]) / m);
			ptrVal[j] = pow(ptrVal[j], pownum);

		}
	}
	cvSaveImage("C:/Users/54053/Desktop/InterBox/val.jpg", pVal);

}

void AGammaCorrection::GenerateGaussianKernel(int kernelSize, double sigma0, CvMat *pKernel)
{
	int m = (kernelSize - 1) / 2;
	CvMat *pPattern = cvCreateMat(kernelSize, kernelSize, CV_32F);

	//generate 2d gaussian kernel
	double s2 = 2.0 * sigma0 * sigma0;

	for (int i = 0; i < kernelSize; i++)
	{
		for (int j = 0; j < kernelSize; j++)
		{
			int x = i - m;
			int y = j - m;
			float v = exp(-(1.0 * x * x + 1.0 * y * y) / s2);
			CV_MAT_ELEM(*pPattern, float, i, j) = v;
		}
	}

	CvScalar all = cvSum(pPattern);
	CvMat *tTemp = cvCreateMat(kernelSize, kernelSize, CV_32F);
	cvSet(tTemp, all);
	cvDiv(pPattern, tTemp, pKernel);

}