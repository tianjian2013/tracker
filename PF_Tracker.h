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
  histogram* histo; /**< reference histogram describing region being tracked */
  float w;          /**< weight */
} particle;




class PF_Tracker:public FrameProcessor
{
public:
	int frameNum,frameheight,framewidth;
	Rect boundary;

	 gsl_rng* rng;
	 Mat frame,hsv_frame,hsv_ref_imgs;
	 histogram* ref_histo;
	 particle* particles, * new_particles;
     Scalar color;
	 Rect region;
	 int num_particles;

public:
	PF_Tracker();
	void process(Mat & input,Mat & output);

private:
	histogram* compute_ref_histos( Mat& frame, Rect region );
	histogram* calc_histogram( Mat & img );
	int histo_bin( float h, float s, float v );
	void normalize_histogram( histogram* histo );
	particle* init_distribution( Rect region, histogram* &histo, int p);
	particle transition( particle p, int w, int h, gsl_rng* rng );
	void normalize_weights( particle* particles, int n );
	float likelihood( Mat &img, int r, int c,int w, int h, histogram* ref_histo );
	float histo_dist_sq( histogram* h1, histogram* h2 );
	particle* resample( particle* particles, int n );
	//int particle_cmp( void* p1, void* p2 );
	void display_particle( Mat img, particle p, CvScalar color );
};