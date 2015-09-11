#include "common.h"

Mat GetFundamentalMat(const vector<KeyPoint>& imgpts1,
					   const vector<KeyPoint>& imgpts2,
					   vector<KeyPoint>& imgpts1_good,
					   vector<KeyPoint>& imgpts2_good,
					   vector<DMatch>& matches
#ifdef __SFM__DEBUG__
					  ,const Mat& img_1,
					  const Mat& img_2
#endif
					  ) 
{
	//Try to eliminate keypoints based on the fundamental matrix
	//(although this is not the proper way to do this)
	vector<uchar> status(imgpts1.size());
	
#ifdef __SFM__DEBUG__
	std::vector< DMatch > good_matches_;
	std::vector<KeyPoint> keypoints_1, keypoints_2;
#endif		
	//	undistortPoints(imgpts1, imgpts1, cam_matrix, distortion_coeff);
	//	undistortPoints(imgpts2, imgpts2, cam_matrix, distortion_coeff);
	//
	imgpts1_good.clear(); imgpts2_good.clear();
	
	vector<KeyPoint> imgpts1_tmp;
	vector<KeyPoint> imgpts2_tmp;
	if (matches.size() <= 0) {
		imgpts1_tmp = imgpts1;
		imgpts2_tmp = imgpts2;
	} else {
		GetAlignedPointsFromMatch(imgpts1, imgpts2, matches, imgpts1_tmp, imgpts2_tmp);
	}
	
	Mat F;
	{
		vector<Point2f> pts1,pts2;
		KeyPointsToPoints(imgpts1_tmp, pts1);
		KeyPointsToPoints(imgpts2_tmp, pts2);
#ifdef __SFM__DEBUG__
		cout << "pts1 " << pts1.size() << " (orig pts " << imgpts1_tmp.size() << ")" << endl;
		cout << "pts2 " << pts2.size() << " (orig pts " << imgpts2_tmp.size() << ")" << endl;
#endif
		double minVal,maxVal;
		cv::minMaxIdx(pts1,&minVal,&maxVal);
		F = findFundamentalMat(pts1, pts2, FM_RANSAC, 0.006 * maxVal, 0.99, status); //threshold from [Snavely07 4.1]
	}
	
	vector<DMatch> new_matches;
	cout << "F keeping " << countNonZero(status) << " / " << status.size() << endl;	
	for (unsigned int i=0; i<status.size(); i++) {
		if (status[i]) 
		{
			imgpts1_good.push_back(imgpts1_tmp[i]);
			imgpts2_good.push_back(imgpts2_tmp[i]);

			//new_matches.push_back(DMatch(matches[i].queryIdx,matches[i].trainIdx,matches[i].distance));
			new_matches.push_back(matches[i]);
#ifdef __SFM__DEBUG__
			good_matches_.push_back(DMatch(imgpts1_good.size()-1,imgpts1_good.size()-1,1.0));
			keypoints_1.push_back(imgpts1_tmp[i]);
			keypoints_2.push_back(imgpts2_tmp[i]);
#endif
		}
	}	
	
	cout << matches.size() << " matches before, " << new_matches.size() << " new matches after Fundamental Matrix\n";
	matches = new_matches; //keep only those points who survived the fundamental matrix
	
#if 0
	//-- Draw only "good" matches
#ifdef __SFM__DEBUG__
	if(!img_1.empty() && !img_2.empty()) {		
		vector<Point2f> i_pts,j_pts;
		Mat img_orig_matches;
		{ //draw original features in red
			vector<uchar> vstatus(imgpts1_tmp.size(),1);
			vector<float> verror(imgpts1_tmp.size(),1.0);
			img_1.copyTo(img_orig_matches);
			KeyPointsToPoints(imgpts1_tmp, i_pts);
			KeyPointsToPoints(imgpts2_tmp, j_pts);
			drawArrows(img_orig_matches, i_pts, j_pts, vstatus, verror, Scalar(0,0,255));
		}
		{ //superimpose filtered features in green
			vector<uchar> vstatus(imgpts1_good.size(),1);
			vector<float> verror(imgpts1_good.size(),1.0);
			i_pts.resize(imgpts1_good.size());
			j_pts.resize(imgpts2_good.size());
			KeyPointsToPoints(imgpts1_good, i_pts);
			KeyPointsToPoints(imgpts2_good, j_pts);
			drawArrows(img_orig_matches, i_pts, j_pts, vstatus, verror, Scalar(0,255,0));
			imshow( "Filtered Matches", img_orig_matches );
		}
		int c = waitKey(0);
		if (c=='s') {
			imwrite("fundamental_mat_matches.png", img_orig_matches);
		}
		destroyWindow("Filtered Matches");
	}
#endif		
#endif
	
	return F;
}