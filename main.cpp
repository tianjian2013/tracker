#include "common.h"
//#include "OFFeatureMatcher.h"
#include "PF_Tracker.h"

int main()
{

	VideoProcessor vedio;
	PF_Tracker tracker;
    //vedio.setInput("D:\\������\\�����ڵ�\\ʵ������\\rotate.avi");
	vedio.setInput("D:\\������\\����\\vot2013\\iceskater.avi");
	//vedio.setInput("D:\\������\\�����ڵ�\\occ\\20.avi");
	vedio.displayOutput("Output");
	vedio.setFrameProcessor(&tracker);
	vedio.setDelay(30);
	vedio.run();

	/*
	Mat src1 = imread(imgPath+"00000001.jpg");
	Mat src2 = imread(imgPath+"00000002.jpg");

	vector<cv::Mat> imgs; 
	vector<vector<cv::KeyPoint> > imgpts;
	vector<DMatch> matches;

	imgs.push_back(src1);
	imgs.push_back(src2);

	OFFeatureMatcher offMatcher(imgs, imgpts);
	offMatcher.MatchFeatures(0, 1, &matches);

	ShowMatchResult(src1, src2, imgpts[0], imgpts[1], matches);*/

	return 0;
}