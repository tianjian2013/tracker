#include "GPUSURFFeatureMatcher.h"

#include <iostream>
#include <set>#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;
using namespace cv::gpu;

//c'tor
GPUSURFFeatureMatcher::GPUSURFFeatureMatcher()
{
	//cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
}	
