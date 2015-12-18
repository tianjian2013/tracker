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

void addOne(string &str)
{
	int n = str.size() - 1;
	while (n >= 0)
	{
		if (str[n] == '9')
		{
			str[n] = '0';
			n--;
		}
		else
		{
			str[n] = str[n] + 1;
			break;
		}
	}
}

std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts) 
{
	std::vector<cv::Point3d> out;
	for (unsigned int i=0; i<cpts.size(); i++) {
		out.push_back(cpts[i].pt);
	}
	return out;
}

vector <string> makeImgsVector(const string &path, int len)
{
    string firstImg = "00000000";
	vector <string> ret;
	for (int i = 0; i < len; i++)
	{
		addOne(firstImg);
		ret.push_back(path+firstImg+".jpg");
	}
	return ret;
}