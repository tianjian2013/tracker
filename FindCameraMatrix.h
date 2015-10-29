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
					  ) ;

bool DecomposeEtoRandT(
					   Mat_<double>& E,
					   Mat_<double>& R1,
					   Mat_<double>& R2,
					   Mat_<double>& t1,
					   Mat_<double>& t2
					  ) ;

bool TestTriangulation(const vector<CloudPoint>& pcloud, const Matx34d& P, vector<uchar>& status);