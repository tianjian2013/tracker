#include "PF_Tracker.h"

int main()
{
	string imgPath = libraryPath+vedioName+"\\";
	vector<string> imgs = makeImgsVector(imgPath, vedioLength);
	VideoProcessor vedio;
	PF_Tracker tracker;
	vedio.setInput(imgs);
	vedio.displayOutput("Output");
	vedio.setFrameProcessor(&tracker);
	vedio.setDelay(1);
	vedio.run();
	return 0;
}
