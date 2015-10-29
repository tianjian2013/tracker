#include "FindCameraMatrix.h"

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
	imgpts1_good.clear(); 
	imgpts2_good.clear();
	
	vector<KeyPoint> imgpts1_tmp;
	vector<KeyPoint> imgpts2_tmp;
	if (matches.size() <= 0) 
	{
		imgpts1_tmp = imgpts1;
		imgpts2_tmp = imgpts2;
	} 
	else 
	{
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
	for (unsigned int i=0; i<status.size(); i++) 
	{
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
	return F;
}

void TakeSVDOfE(Mat_<double>& E, Mat& svd_u, Mat& svd_vt, Mat& svd_w) {
#if 1
	//Using OpenCV's SVD
	SVD svd(E,SVD::MODIFY_A);
	svd_u = svd.u;
	svd_vt = svd.vt;
	svd_w = svd.w;
#else
	//Using Eigen's SVD
	cout << "Eigen3 SVD..\n";
	Eigen::Matrix3f  e = Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor> >((double*)E.data).cast<float>();
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(e, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXf Esvd_u = svd.matrixU();
	Eigen::MatrixXf Esvd_v = svd.matrixV();
	svd_u = (Mat_<double>(3,3) << Esvd_u(0,0), Esvd_u(0,1), Esvd_u(0,2),
						  Esvd_u(1,0), Esvd_u(1,1), Esvd_u(1,2), 
						  Esvd_u(2,0), Esvd_u(2,1), Esvd_u(2,2)); 
	Mat_<double> svd_v = (Mat_<double>(3,3) << Esvd_v(0,0), Esvd_v(0,1), Esvd_v(0,2),
						  Esvd_v(1,0), Esvd_v(1,1), Esvd_v(1,2), 
						  Esvd_v(2,0), Esvd_v(2,1), Esvd_v(2,2));
	svd_vt = svd_v.t();
	svd_w = (Mat_<double>(1,3) << svd.singularValues()[0] , svd.singularValues()[1] , svd.singularValues()[2]);
#endif
	//cout << "----------------------- SVD ------------------------\n";
	//cout << "U:\n"<<svd_u<<"\nW:\n"<<svd_w<<"\nVt:\n"<<svd_vt<<endl;
	//cout << "----------------------------------------------------\n";
}

bool DecomposeEtoRandT(
					   Mat_<double>& E,
					   Mat_<double>& R1,
					   Mat_<double>& R2,
					   Mat_<double>& t1,
					   Mat_<double>& t2
					  ) 
{
	//Using HZ E decomposition
	Mat svd_u, svd_vt, svd_w;
	TakeSVDOfE(E,svd_u,svd_vt,svd_w);

	//check if first and second singular values are the same (as they should be)
	double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
	if(singular_values_ratio>1.0) 
		singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
	if (singular_values_ratio < 0.7) 
	{
		cout << "singular values are too far apart\n";
		return false;
	}

	Matx33d W(0,-1,0,	//HZ 9.13
		1,0,0,
		0,0,1);
	Matx33d Wt(0,1,0,
		-1,0,0,
		0,0,1);
	R1 = svd_u * Mat(W) * svd_vt; //HZ 9.19
	R2 = svd_u * Mat(Wt) * svd_vt; //HZ 9.19
	t1 = svd_u.col(2); //u3
	t2 = -svd_u.col(2); //u3
	return true;
}

bool TestTriangulation(const vector<CloudPoint>& pcloud, const Matx34d& P, vector<uchar>& status) 
{
	vector<Point3d> pcloud_pt3d = CloudPointsToPoints(pcloud);
	vector<Point3d> pcloud_pt3d_projected(pcloud_pt3d.size());
	
	Matx44d P4x4 = Matx44d::eye(); 
	for(int i=0;i<12;i++) P4x4.val[i] = P.val[i];
	
	perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);
	
	status.resize(pcloud.size(),0);
	for (int i=0; i<pcloud.size(); i++) {
		status[i] = (pcloud_pt3d_projected[i].z > 0) ? 1 : 0;
	}
	int count = countNonZero(status);

	double percentage = ((double)count / (double)pcloud.size());
	cout << count << "/" << pcloud.size() << " = " << percentage*100.0 << "% are in front of camera" << endl;
	if(percentage < 0.75)
		return false; //less than 75% of the points are in front of the camera

	//check for coplanarity of points
	if(false) //not
	{
		cv::Mat_<double> cldm(pcloud.size(),3);
		for(unsigned int i=0;i<pcloud.size();i++) {
			cldm.row(i)(0) = pcloud[i].pt.x;
			cldm.row(i)(1) = pcloud[i].pt.y;
			cldm.row(i)(2) = pcloud[i].pt.z;
		}
		cv::Mat_<double> mean;
		cv::PCA pca(cldm,mean,CV_PCA_DATA_AS_ROW);

		int num_inliers = 0;
		cv::Vec3d nrm = pca.eigenvectors.row(2); nrm = nrm / norm(nrm);
		cv::Vec3d x0 = pca.mean;
		double p_to_plane_thresh = sqrt(pca.eigenvalues.at<double>(2));

		for (int i=0; i<pcloud.size(); i++) {
			Vec3d w = Vec3d(pcloud[i].pt) - x0;
			double D = fabs(nrm.dot(w));
			if(D < p_to_plane_thresh) num_inliers++;
		}

		cout << num_inliers << "/" << pcloud.size() << " are coplanar" << endl;
		if((double)num_inliers / (double)(pcloud.size()) > 0.85)
			return false;
	}

	return true;
}