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
void showVector(vector <int> & v);
void weightAdd(vector <int> & v1, vector <int> & v2);
vector < double> NormHis(vector <int> & v);
void weightAdd(vector <double> & v1, vector <double> & v2);

const string libraryPath = "D:\\样本集\\跟踪\\vot2013\\";
//const string vedioName = "iceskater";
const string vedioName = "face";
//const string vedioName = "tiger";
//const string vedioName = "wiper";
//const string vedioName = "soccer1";
//const string vedioName = "car1";
//const string vedioName = "girl";

const int vedioLength = 400;
const string outputImgPath = "E:\\论文结果图片\\2\\";
//const string imgPath = "D:\\样本集\\跟踪\\vot2015\\bag\\";

const bool FLAG1 = true;  // 显示匹配结果数值

#endif