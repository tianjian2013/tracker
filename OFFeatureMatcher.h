#include "common.h"

class OFFeatureMatcher 
{
	std::vector<cv::Mat>& imgs; 
	std::vector<std::vector<cv::KeyPoint> >& imgpts;
	
public:
	OFFeatureMatcher(std::vector<cv::Mat>& imgs_, std::vector<std::vector<cv::KeyPoint> >& imgpts_);
	void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches);
	std::vector<cv::KeyPoint> GetImagePoints(int idx) { return imgpts[idx]; }
};