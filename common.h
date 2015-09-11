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

#include <vector>
#include <iostream>
#include <list>
#include <set>


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

const string imgPath = "D:\\Ñù±¾¼¯\\¸ú×Ù\\vot2015\\iceskater2\\";

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

#endif