#include "common.h"

void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps) {
	ps.clear();
	for (unsigned int i=0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}

void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
							   const std::vector<cv::KeyPoint>& imgpts2,
							   const std::vector<cv::DMatch>& matches,
							   std::vector<cv::KeyPoint>& pt_set1,
							   std::vector<cv::KeyPoint>& pt_set2) 
{
	for (unsigned int i=0; i<matches.size(); i++) {
//		cout << "matches[i].queryIdx " << matches[i].queryIdx << " matches[i].trainIdx " << matches[i].trainIdx << endl;
		pt_set1.push_back(imgpts1[matches[i].queryIdx]);
		pt_set2.push_back(imgpts2[matches[i].trainIdx]);
	}	
}

void ShowMatchResult(Mat &img1, 
					 Mat &img2,
					 const std::vector<cv::KeyPoint>& imgpts1,
					 const std::vector<cv::KeyPoint>& imgpts2,
					 const std::vector<cv::DMatch>& matches)
{
	//Mat show;
	//show.create(img1.rows,img1.cols*2,CV_8U);
	for (unsigned int i=0; i<matches.size(); i++) 
	{
		circle(img1, imgpts1[matches[i].queryIdx].pt, 10, Scalar(0));
		circle(img2, imgpts2[matches[i].trainIdx].pt, 10, Scalar(0));
		imshow("img1", img1);
		imshow("img2", img2);
		waitKey(0);
	}
}