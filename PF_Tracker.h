#include "videoprocessor.h"
#include "common.h"

using namespace cv;
using namespace std;


/******************************* Definitions *********************************/

/* number of bins of HSV in histogram */
#define NH 10
#define NS 10
#define NV 10

/* max HSV values */
#define H_MAX 360.0
#define S_MAX 1.0
#define V_MAX 1.0

/* low thresholds on saturation and value for histogramming */
#define S_THRESH 0.1
#define V_THRESH 0.2

/* distribution parameter */
#define LAMBDA 20

/* standard deviations for gaussian sampling in transition model */
#define TRANS_X_STD 1.0
#define TRANS_Y_STD 0.5
#define TRANS_S_STD 0.001

/* autoregressive dynamics parameters for transition model */
#define A1  2.0
#define A2 -1.0
#define B0  1.0000
/******************************** Structures *********************************/

/**
   An HSV histogram represented by NH * NS + NV bins.  Pixels with saturation
   and value greater than S_THRESH and V_THRESH fill the first NH * NS bins.
   Other, "colorless" pixels fill the last NV value-only bins.
*/
typedef struct histogram {
  float histo[NH*NS + NV];   /**< histogram array */
  int n;                     /**< length of histogram array */
} histogram;

/******************************* Structures **********************************/

/**
   A particle is an instantiation of the state variables of the system
   being monitored.  A collection of particles is essentially a
   discretization of the posterior probability of the system.
*/
typedef struct particle {
  float x;          /**< current x coordinate */
  float y;          /**< current y coordinate */
  float s;          /**< scale */
  float xp;         /**< previous x coordinate */
  float yp;         /**< previous y coordinate */
  float sp;         /**< previous scale */
  float x0;         /**< original x coordinate */
  float y0;         /**< original y coordinate */
  int width;        /**< original width of region described by particle */
  int height;       /**< original height of region described by particle */
  //histogram* histo; /**< reference histogram describing region being tracked */
  vector <int> histogram;
  float w;          /**< weight */
  float distance;
  //Rect * ptrPatches;
} particle;


class Patch
{
public:
	Rect r,boundRect;
	vector <int> histogram;
	float confidence;
	vector <particle> particles;
	Patch()
	{

	}
	Patch(Rect rr, float c): r(rr), confidence(c)
	{
	}

};



class PF_Tracker:public FrameProcessor
{
//public:
	static const int patchSize = 15;
	static const int neNum = 3;
	static const int patchNum = 6; 
	static const int num_particles = 100; //粒子数

	int frameNum, frameheight, framewidth;

	Rect boundary;
	Rect targetRegion;
	Point Focus;
	Rect searchArea;

	gsl_rng* rng;

	Mat frame,hsv_frame,hsv_ref_imgs, preFrame, showImg, roi;


	vector < Mat > splithsv;
	vector < Mat > IIV_T;  //目标积分图
	Mat mask;

	
	//histogram* ref_histo;
	//particle* particles, * new_particles;
    //Scalar color;
	
	//vector <Rect> patches;
	//int NumPatches;

public:
	PF_Tracker();
	void process(Mat & input,Mat & output);

private:
	//compute_ref_histos( Mat& frame, Rect region );
    //vector <int> targetHistogram;
	//histogram* calc_histogram( Mat & img );


	int histo_bin( float h, float s, float v );
	void normalize_histogram( histogram* histo );
	

	particle transition( particle p, int w, int h, gsl_rng* rng );
	void normalize_weights( vector <particle> & vp );

	vector <particle> resample( vector <particle> &particles );
	//int particle_cmp( void* p1, void* p2 );
	void makePatches();

	void compute_IH ();
	void compute_histogram (Rect r,vector < int >& hist);
	vector <vector < vector<int> > > hisIntegral;

	vector<int> hisTarget, hisSearchArea,hisBackground;
	vector< double> hisTargetNorm, hisSearchAreaNorm,hisBackgroundNorm;
	//Mat pToObject, pToObject1;
	Mat  pToObject1;
	vector < Patch> vpatches;
	int distance(const Rect& r1, const Rect& r2);
	int l2distance(const Rect& r1, const Rect& r2);
	float distance ( const vector <int> &v1 , const vector <int> &v2 );
	Mat clacWij(int row, vector <int> indexes);
	void init();
	void calcW();
	Mat W;
	Mat T;
	vector <vector <int>> neighbors;
	//void score(vector <vector < Patch> >  & vvpatches);
	vector< Scalar> showColors;
	float solveIP(Rect region, vector <vector < Patch> >  &vvpatches, vector <int> & locationIndexes);
	void init_distribution( Patch &p);
	float likelihood( int r, int c,int w, int h, vector <int>  &ref_histo, float &dt);
	Rect  calcBoundingRect();//找所有patch的最小外接矩形
	void calcBoundingRect(Patch & ph); //找所有patch的最小外接矩形
	Rect boundRect;
	void updatePatches(int deleteIndex);
//sfm 
private:
	Mat K;
	deque <Mat> previousImgs;
	deque <Rect> previousRegions;
	void sfm();


	// 存放结果， 中间结果
	vector <Rect> groundTruth;
};