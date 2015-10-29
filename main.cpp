#include <string>
#include <iostream>

using namespace std;

#include "common.h"
#include "PF_Tracker.h"



int main()
{
	
	vector<string> imgs;

	/*
	string firstImg="D:\\样本集\\stisample\\seq02-img-right\\image_00000930";
	for (int i = 0; i < 50; i++)
	{
		imgs.push_back(firstImg+"_1.png");
		addOne(firstImg);
	}*/

    imgs.push_back("D:\\样本集\\3d\\35m3\\14472237-2015-05-14-135632.jpg");
	imgs.push_back("D:\\样本集\\3d\\35m3\\14472237-2015-05-14-135707.jpg");
	imgs.push_back("D:\\样本集\\3d\\35m3\\14472237-2015-05-14-135723.jpg");
	VideoProcessor vedio;
	PF_Tracker tracker;
	vedio.setInput(imgs);
	//vedio.setInput("D:\\样本集\\3d\\Capture_20150915_1.mp4");
	
	vedio.displayOutput("Output");
	vedio.setFrameProcessor(&tracker);
	vedio.setDelay(30);
	vedio.run();
	return 0;
}
