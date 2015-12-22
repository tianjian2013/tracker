#ifndef COMMON_H
#define COMMON_H

/* From standard C library */
#include <math.h>
#include <stdio.h>
#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
//#include "unistd.h"

/* From OpenCV library */
#include <opencv2\opencv.hpp>


/* From GSL */
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/* From standard C++ library */
#include <vector>
#include <iostream>
#include <list>
#include <set>
#include <queue>
#include <string>

/* cv color*/
#define YELLOW Scalar(0,255,255)
#define BLACK Scalar(0,0,0)
#define BLUE Scalar(255,0,0)
#define GREEN Scalar(0,255,0)
#define RED Scalar(0,0,255)
#define PURPLE Scalar(240,32,160)
#define ORANGE Scalar(0,97,255)
#define BROWN  Scalar(42,42,128)


#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef MIN
#define MIN(x,y) ( ( x < y )? x : y )
#endif
#ifndef MAX
#define MAX(x,y) ( ( x > y )? x : y )
#endif
#ifndef ABS
#define ABS(x) ( ( x < 0 )? -x : x )
#endif

using namespace cv;
using namespace std;



void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps);

void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
							   const std::vector<cv::KeyPoint>& imgpts2,
							   const std::vector<cv::DMatch>& matches,
							   std::vector<cv::KeyPoint>& pt_set1,
							   std::vector<cv::KeyPoint>& pt_set2);

void ShowMatchResult(Mat &img1, 
					 Mat &img2,
					 const std::vector<cv::KeyPoint>& imgpts1,
					 const std::vector<cv::KeyPoint>& imgpts2,
					 const std::vector<cv::DMatch>& matches);

void addOne(string &str);

struct CloudPoint 
{
	cv::Point3d pt;
	std::vector<int> imgpt_for_img;
	double reprojection_error;
};


std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts);
vector <string> makeImgsVector(const string &path, int len);


const string libraryPath = "D:\\样本集\\跟踪\\vot2015\\";
const string vedioName = "iceskater1";
const int vedioLength = 661;

//const string imgPath = "D:\\样本集\\跟踪\\vot2015\\bag\\";



#endif