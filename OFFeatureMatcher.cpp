#include "OFFeatureMatcher.h"

#ifdef __SFM__DEBUG__
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>
#include <sstream>
#endif

#include <set>

using namespace std;
using namespace cv;

OFFeatureMatcher::OFFeatureMatcher(std::vector<cv::Mat>& imgs_, std::vector<std::vector<cv::KeyPoint> >& imgpts_):imgpts(imgpts_),
																												  imgs(imgs_)
{
	//detect keypoints for all images
	FastFeatureDetector ffd;
//	DenseFeatureDetector ffd;
	ffd.detect(imgs, imgpts);
}

void OFFeatureMatcher::MatchFeatures(int idx_i, int idx_j, vector<DMatch>* matches) {	
	
	vector<Point2f> i_pts; 
	KeyPointsToPoints(imgpts[idx_i],i_pts);
	
	vector<Point2f> j_pts(i_pts.size());
	
	// making sure images are grayscale
	Mat prevgray,gray;
	if (imgs[idx_i].channels() == 3) {
		cvtColor(imgs[idx_i],prevgray,CV_RGB2GRAY);
		cvtColor(imgs[idx_j],gray,CV_RGB2GRAY);
	} else {
		prevgray = imgs[idx_i];
		gray = imgs[idx_j];
	}
	vector<uchar> vstatus(i_pts.size()); 
	vector<float> verror(i_pts.size());

    calcOpticalFlowPyrLK(prevgray, gray, i_pts, j_pts, vstatus, verror);
	
	double thresh = 1.0;
	vector<Point2f> to_find;
	vector<int> to_find_back_idx;
	for (unsigned int i = 0; i < vstatus.size(); i++)
	{
		if (vstatus[i] && verror[i] < 1.0)
		{
			to_find_back_idx.push_back(i);
			to_find.push_back(j_pts[i]);
		}
		else
		{
			vstatus[i] = 0;
		}
	}

	std::set<int> found_in_imgpts_j;
	Mat to_find_flat = Mat(to_find).reshape(1,to_find.size());
	
	vector<Point2f> j_pts_to_find;
	KeyPointsToPoints(imgpts[idx_j],j_pts_to_find);
	Mat j_pts_flat = Mat(j_pts_to_find).reshape(1,j_pts_to_find.size());

	vector<vector<DMatch> > knn_matches;
	//FlannBasedMatcher matcher;
	BFMatcher matcher(CV_L2);
	matcher.radiusMatch(to_find_flat,j_pts_flat,knn_matches,2.0f);
	for(int i = 0; i < knn_matches.size(); i++)
	{
		DMatch _m;
		if(knn_matches[i].size()==1) 
		{
			_m = knn_matches[i][0];
		}
		else if(knn_matches[i].size()>1) 
		{
			if(knn_matches[i][0].distance / knn_matches[i][1].distance < 0.7) 
			{
				_m = knn_matches[i][0];
			}
			else 
			{
				continue; // did not pass ratio test
			}
		} 
		else {
			continue; // no match
		}
		if (found_in_imgpts_j.find(_m.trainIdx) == found_in_imgpts_j.end()) 
		{ // prevent duplicates
			_m.queryIdx = to_find_back_idx[_m.queryIdx]; //back to original indexing of points for <i_idx>
			matches->push_back(_m);
			found_in_imgpts_j.insert(_m.trainIdx);
		}
	}
	cout << "pruned " << matches->size() << " / " << knn_matches.size() << " matches" << endl;
}