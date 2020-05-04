#include "..\..\Include\OpenSource\OpenCV\OpenCV2.4.11\opencv.hpp"

class AGammaCorrection
{
public:
	AGammaCorrection();
	~AGammaCorrection();

	void AdaptiveGammaCorrection(IplImage *pImg, IplImage *pHue, IplImage *pSat, IplImage *pVal);

private:
	void GenerateGaussianKernel(int kernelSize, double sigma0, CvMat *pKernel);
};
