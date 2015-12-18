#include "PF_Tracker.h"
#include "select.h"
#include "OFFeatureMatcher.h"
#include "FindCameraMatrix.h"
#include "Triangulation.h"
#include "RichFeatureMatcher.h"
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>

//---gsl的库文件-----------
#pragma comment (lib, "libgsl.a")

extern bool gotBB;
extern Rect box;
/*
  Compare two particles based on weight.  For use in qsort.

  @param p1 pointer to a particle
  @param p2 pointer to a particle

  @return Returns -1 if the \a p1 has lower weight than \a p2, 1 if \a p1
    has higher weight than \a p2, and 0 if their weights are equal.
*/
int particle_cmp( const void* p1,const  void* p2 )
{
  particle* _p1 = (particle*)p1;
  particle* _p2 = (particle*)p2;

  if( _p1->w > _p2->w )
    return -1;
  if( _p1->w < _p2->w )
    return 1;
  return 0;
}


//static const double distcoeff[]={0.03489 ,  -0.25811,   -0.00048,   0.00330,  0.00000};

PF_Tracker::PF_Tracker():frameNum(0)
	                    ,frameheight(0)
						,framewidth(0)
						,num_particles(300)   //粒子数
						,NumPatches(3)
{
	/* parse command line and initialize random number generator */
   gsl_rng_env_setup();
   rng = gsl_rng_alloc( gsl_rng_mt19937 );
   gsl_rng_set( rng, time(NULL) );

   /*K = (Mat_ <double>(3, 3) << 1105.2, 0, 652.5, 
                              0, 1106.5, 373.6,
							  0, 0, 1);*/
   K = (Mat_ <double>(3, 3) <<643.37825 , 0, 399.26196, 0, 643.34785, 299.48355, 0, 0, 1 );

  /* K = (Mat_ <double>(3, 3) << 500.80696, 0, 307.20573, 
							   0,  500.39406, 233.52064,
                               0, 0, 1);
	*/						   
  // K = (Mat_ <double>(3, 3) <<7176.13570  ,0,662.33685,0, 7173.11117,681.71742 ,0,0,1);


}

void PF_Tracker::process(Mat & input,Mat & output)
{
	frame = input;
	output = input.clone(); 
	showImg = output;
	//cvtColor(frame,hsv_frame,CV_BGR2HSV );
	frameNum++;

	if(frameNum == 1)     //手动选择跟踪目标，在初始帧
	{
		framewidth = frame.cols;
		frameheight = frame.rows;
		boundary = Rect(0, 0, framewidth - 1, frameheight - 1);
		cvNamedWindow("SelectObject", CV_WINDOW_AUTOSIZE);
		imshow("SelectObject", frame);
	    cvSetMouseCallback("SelectObject", SelectObject, NULL); 
		while(!gotBB)
	    {
			if (cvWaitKey(10) == 27)
			    break;
		}
		cvDestroyWindow("SelectObject");
		targetRegion = box;


		searchArea.x = targetRegion.x - targetRegion.width;
        searchArea.y = targetRegion.y - targetRegion.height;
	    searchArea.width = targetRegion.width * 3;
	    searchArea.height = targetRegion.height * 3;
	    searchArea &= boundary;
		cout<< searchArea<<endl;
		roi=input(searchArea);
	    cvtColor(roi,hsv_frame, CV_BGR2HSV);
        split(hsv_frame, splithsv);
        compute_IH();
		/*int  sum=0;
		for (vector < Mat >::iterator it = IIV_T.begin() ; it != IIV_T.end() ; it++ ) 
	    {
			sum += (*it).at<int>(searchArea.height-1,searchArea.width-1);
		}
		Scalar s=mean(mask);
		cout<<s*searchArea.area()/255<<endl;*/
		//imshow("h", splithsv[1]);
		vector<int> hisTarget, hisSearchArea;
		compute_histogram(targetRegion, hisTarget);
		compute_histogram(searchArea, hisSearchArea);
		imshow("mask", mask);
		waitKey();

		makePatches();
		for (int i = 0; i < patches.size(); i++)
		{
			rectangle(output, patches[i], GREEN, 2); 
		}
		ref_histo = compute_ref_histos(hsv_frame, targetRegion);
		particles = init_distribution(targetRegion, ref_histo, num_particles);
		preFrame = frame;



		return;
	}

	//perform prediction and measurement for each particle 
	for( int j = 0; j < num_particles; j++ )
	{
	      particles[j] = transition( particles[j], framewidth, frameheight, rng);
	      float s = particles[j].s;
	      particles[j].w = likelihood( hsv_frame, cvRound(particles[j].y),
					                   cvRound( particles[j].x ),
					                   cvRound( particles[j].width * s ),
					                   cvRound( particles[j].height * s ),
									   particles[j].histo);//, particles[j].patches );
	}

	//normalize weights and resample a set of unweighted particles 
	normalize_weights( particles, num_particles );//归一化，使所有粒子权重和为1
	new_particles = resample( particles, num_particles );
	free( particles );
	particles = new_particles;

	// display all particles if requested 
    qsort( particles, num_particles, sizeof( particle ), particle_cmp );
     
	targetRegion.x = particles[0].x - particles[0].width * particles[0].s / 2;
    targetRegion.y = particles[0].y - particles[0].height * particles[0].s / 2;
	targetRegion.height = particles[0].height * particles[0].s;
	targetRegion.width = particles[0].width * particles[0].s;
	targetRegion&=boundary;
	rectangle(output, targetRegion, RED, 2); 

	Rect searchArea(targetRegion.x - targetRegion.width, targetRegion.y - targetRegion.height, targetRegion.width*3, targetRegion.height*3);
	searchArea &= boundary;
	for (int i = 0; i < patches.size(); i++)
	{
		Point loc;
		matchTlt(preFrame(patches[i]), frame(searchArea), loc);
		patches[i].x += (loc.x - patches[i].x - patches[i].width/2) + searchArea.x;
		patches[i].y += (loc.y - patches[i].y - patches[i].height/2) + searchArea.y;
		patches[i].x += (loc.x - patches[i].x) + searchArea.x;
		patches[i].y += (loc.y - patches[i].y) + searchArea.y;
	}

	for (int i = 0; i < patches.size(); i++)
	{
		rectangle(showImg, patches[i], GREEN, 2); 
	}
	

	//trackPatches();
	preFrame = frame;
}

/*
  Computes a reference histogram for each of the object regions defined by
  the user

  @param frame video frame in which to compute histograms; should have been
    converted to hsv using bgr2hsv 
  @param region region of a frame over which histogram should be computed
  @return Returns  a  normalized histogramscorresponding
    to region of a frame
*/
histogram* PF_Tracker::compute_ref_histos( Mat& frame, Rect region )
{
  histogram* histo = (histogram *) new (histogram);
  
  Mat ImageROI=frame(region);

  Mat tmp;
  tmp.create(region.height,region.width,frame.type());
  ImageROI.copyTo(tmp);
   
  histo = calc_histogram( tmp );
  normalize_histogram( histo );
 
  return histo;
}

/*
  Calculates a cumulative histogram as defined above for a given array
  of images
  @param img an array of images over which to compute a cumulative histogram;
    each must have been converted to HSV colorspace using bgr2hsv()
  @return Returns an un-normalized HSV histogram for \a imgs
*/
histogram* PF_Tracker::calc_histogram( Mat & img )
{


  histogram* histo= (histogram* )new( histogram );
  histo->n = NH*NS + NV;
  memset( histo->histo, 0, histo->n * sizeof(float) );

  int i, r, c, bin;
  /* increment appropriate histogram bin for each pixel */
  for( r = 0; r < img.rows; r++ )
  {
	  uchar *data=img.ptr<uchar>(r);
	  for( c = 0; c < img.cols; c++ )
	  {
        float h,s,v;
		h=(float)data[c++]*2;
		s=(float)data[c++]/255;
		v= (float)data[c]/255;
	    bin = histo_bin( h,s,v );
	    histo->histo[bin] += 1;
	  }
  }
  return histo;
}



/*
  Normalizes a histogram so all bins sum to 1.0
  
  @param histo a histogram
*/
void PF_Tracker::normalize_histogram( histogram* histo )
{
  float* hist;
  float sum = 0, inv_sum;
  int i, n;

  hist = histo->histo;
  n = histo->n;

  /* compute sum of all bins and multiply each bin by the sum's inverse */
  for( i = 0; i < n; i++ )
    sum += hist[i];
  inv_sum = 1.0 / sum;
  for( i = 0; i < n; i++ )
    hist[i] *= inv_sum;
}


/*
  Calculates the histogram bin into which an HSV entry falls
  
  @param h Hue
  @param s Saturation
  @param v Value
  
  @return Returns the bin index corresponding to the HSV color defined by
    \a h, \a s, and \a v.
*/
int PF_Tracker::histo_bin( float h, float s, float v )
{
  int hd, sd, vd;

  /* if S or V is less than its threshold, return a "colorless" bin */
  vd = MIN( (int)(v * NV / V_MAX), NV-1 );
  if( s < S_THRESH  ||  v < V_THRESH )
    return NH * NS + vd;
  
  /* otherwise determine "colorful" bin */
  hd = MIN( (int)(h * NH / H_MAX), NH-1 );
  sd = MIN( (int)(s * NS / S_MAX), NS-1 );
  return sd * NH + hd;
}

/*
  Creates an initial distribution of particles at specified locations
  
  
  @param regions an array of regions describing player locations around
    which particles are to be sampled
  @param histos array of histograms describing regions in \a regions
  @param n the number of regions in \a regions
  @param p the total number of particles to be assigned
  
  @return Returns an array of \a p particles sampled from around regions in
    \a regions
*/
particle* PF_Tracker::init_distribution( Rect region, histogram* &histo, int p)
{

  float x, y;
  int  j,  k = 0;
  
  particle* particles = (particle* )new particle[p];  //?
  int width = region.width;
  int height = region.height;
  x = region.x + width / 2;
  y = region.y + height / 2;
  for( j = 0; j < p; j++ )
  {
	  particles[k].x0 = particles[k].xp = particles[k].x = x;
	  particles[k].y0 = particles[k].yp = particles[k].y = y;
	  particles[k].sp = particles[k].s = 1.0;
	  particles[k].width = width;
	  particles[k].height = height;
	  particles[k].histo = histo;
	  particles[k++].w = 0;
  }
  
  /* make sure to create exactly p particles */
  while( k < p )
  {
      width = region.width;
      height = region.height;
      x = region.x + width / 2;
      y = region.y + height / 2;
      particles[k].x0 = particles[k].xp = particles[k].x = x;
      particles[k].y0 = particles[k].yp = particles[k].y = y;
      particles[k].sp = particles[k].s = 1.0;
      particles[k].width = width;
      particles[k].height = height;
      particles[k].histo = histo;
      particles[k++].w = 0;
   }

  return particles;
}

/*
  Samples a transition model for a given particle
  
  @param p a particle to be transitioned
  @param w video frame width
  @param h video frame height
  @param rng a random number generator from which to sample

  @return Returns a new particle sampled based on <EM>p</EM>'s transition
    model
*/
particle PF_Tracker::transition( particle p, int w, int h, gsl_rng* rng )
{
  float x, y, s;
  particle pn;


  //随机漂移模型
   x = p.x+10.0*gsl_ran_gaussian( rng, TRANS_X_STD ) ;
   pn.x = MAX( 0.0, MIN( (float)w - 1.0, x ) );
   y = p.y + 10.0*gsl_ran_gaussian( rng, TRANS_Y_STD ) ;
   pn.y = MAX( 0.0, MIN( (float)h - 1.0, y ) );
   s = p.s  +  10.0*gsl_ran_gaussian( rng, TRANS_S_STD );
   pn.s = MAX( 0.1, s );

  /* sample new state using second-order autoregressive dynamics *
  x = A1 * ( p.x - p.x0 ) + A2 * ( p.xp - p.x0 ) +
    B0 * gsl_ran_gaussian( rng, TRANS_X_STD ) + p.x0;
  pn.x = MAX( 0.0, MIN( (float)w - 1.0, x ) );
  y = A1 * ( p.y - p.y0 ) + A2 * ( p.yp - p.y0 ) +
    B0 * gsl_ran_gaussian( rng, TRANS_Y_STD ) + p.y0;
  pn.y = MAX( 0.0, MIN( (float)h - 1.0, y ) );
  s = A1 * ( p.s - 1.0 ) + A2 * ( p.sp - 1.0 ) +
    B0 * gsl_ran_gaussian( rng, TRANS_S_STD ) + 1.0;
  pn.s = MAX( 0.1, s );
   /* sample new state using second-order autoregressive dynamics */
  pn.xp = p.x;
  pn.yp = p.y;
  pn.sp = p.s;
  pn.x0 = p.x0;
  pn.y0 = p.y0;
  pn.width = p.width;
  pn.height = p.height;
  pn.histo = p.histo;
  pn.w = 0;

  return pn;
}



/*
  Normalizes particle weights so they sum to 1
  
  @param particles an array of particles whose weights are to be normalized
  @param n the number of particles in \a particles
*/
void PF_Tracker::normalize_weights( particle* particles, int n )
{
  float sum = 0;
  int i;

  for( i = 0; i < n; i++ )
    sum += particles[i].w;
  for( i = 0; i < n; i++ )
    particles[i].w /= sum;
}

float PF_Tracker::likelihood(Mat& img, int r, int c,int w, int h, histogram* ref_histo)//,vector <Rect> &particlePatches)
{
  //IplImage* tmp;
  histogram* histo;
  float d_sq;

  /* extract region around (r,c) and compute and normalize its histogram */
  Rect re=Rect( c - w / 2, r - h / 2, w, h );

  double sim = 0;
  //particlePatches.clear();
  for (int i = 0; i < patches.size(); i++)
  {
		//Rect searchArea(patches[i].x - patches[i].width, patches[i].y - patches[i].height, patches[i].width*3, patches[i].height*3);
		Rect searchArea = re;
		searchArea &= boundary;
		if (searchArea.area() < patches[i].area())
		{
			continue;
		}
		Point loc;
		sim +=  matchTlt(preFrame(patches[i]), frame(searchArea), loc);
		Rect r;
		//cout<<sim<<endl;
		r.x += (loc.x - patches[i].x - patches[i].width/2) + searchArea.x;
		r.y += (loc.y - patches[i].y - patches[i].height/2) + searchArea.y;
		r.x += (loc.x - patches[i].x) + searchArea.x;
		r.y += (loc.y - patches[i].y) + searchArea.y;
		//rectangle(showImg, patches[i], GREEN, 2); 
		//particlePatches.push_back(r);
  }
  return   sim / patches.size();  

  re&=boundary;
  Mat ImageROI=img(re);
  Mat tmp;
  tmp.create(re.height,re.width,CV_32FC3);
  ImageROI.copyTo(tmp);

  histo = calc_histogram( tmp );
  normalize_histogram( histo );

  /* compute likelihood as e^{\lambda D^2(h, h^*)} */
  d_sq = histo_dist_sq( histo, ref_histo );
  delete( histo );
  cout <<exp( -LAMBDA * d_sq )<<endl;
  return exp( -LAMBDA * d_sq );
}

float PF_Tracker::histo_dist_sq( histogram* h1, histogram* h2 )
{
  float* hist1, * hist2;
  float sum = 0;
  int i, n;

  n = h1->n;
  hist1 = h1->histo;
  hist2 = h2->histo;

  /*
    According the the Battacharyya similarity coefficient,
    
    D = \sqrt{ 1 - \sum_1^n{ \sqrt{ h_1(i) * h_2(i) } } }
  */
  for( i = 0; i < n; i++ )
    sum += sqrt( hist1[i]*hist2[i] );
  return 1.0 - sum;
}


/*
  Re-samples a set of particles according to their weights to produce a
  new set of unweighted particles
  
  @param particles an old set of weighted particles whose weights have been
    normalized with normalize_weights()
  @param n the number of particles in \a particles
  
  @return Returns a new set of unweighted particles sampled from \a particles
*/
particle* PF_Tracker::resample( particle* particles, int n )
{
  particle* new_particles;
  int i, j, np, k = 0;

  qsort( particles, n, sizeof( particle ), &particle_cmp );
  new_particles =  (particle*)malloc( n * sizeof( particle ) );
  for( i = 0; i < n; i++ )
    {
      np = cvRound( particles[i].w * n );
      for( j = 0; j < np; j++ )
	{
	  new_particles[k++] = particles[i];
	  if( k == n )
	    goto exit;
	}
    }
  while( k < n )
    new_particles[k++] = particles[0];

 exit:
  return new_particles;
}

void PF_Tracker::makePatches()
{
	while (patches.size() < NumPatches)
	{
		Rect r;
		int length = min(targetRegion.width, targetRegion.height);
		length = (rand() % 10) * length / 10;
		r.x = targetRegion.x + rand() % targetRegion.width;
		r.y = targetRegion.y + rand() % targetRegion.height;
		r.width = length;
		r.x -= length/2;
		r.height = length;
		r.y -= length/2;
		if (r.area() * 10 >= targetRegion.area() && r.area() <= targetRegion.area() / 3)
		{
			r &= targetRegion;
			patches.push_back(r);
		}
	}
}

void PF_Tracker::trackPatches()
{
	for (int i = 0; i < patches.size(); i++)
	{
		Rect searchArea(patches[i].x - patches[i].width, patches[i].y - patches[i].height, patches[i].width*3, patches[i].height*3);
		
		searchArea &= boundary;
		Point loc;
		float sim  = matchTlt(preFrame(patches[i]), frame(searchArea), loc);
		cout<<sim<<endl;
		//patches[i].x += (loc.x - patches[i].x - patches[i].width/2) + searchArea.x;
		//patches[i].y += (loc.y - patches[i].y - patches[i].height/2) + searchArea.y;
		patches[i].x += (loc.x - patches[i].x) + searchArea.x;
		patches[i].y += (loc.y - patches[i].y) + searchArea.y;
		//circle(showImg, loc, 10, RED);
		rectangle(showImg, patches[i], GREEN, 2); 
		
	}
}

double PF_Tracker::matchTlt(Mat src,Mat dst,Point& maxLoc)
{
	double minVal,maxVal;
	Point minLoc;

	Mat srcGray(src.rows,src.cols,CV_8UC1);
	Mat dstGray(dst.rows,dst.cols,CV_8UC1);

	Mat result(dst.rows-src.rows+1,dst.cols-src.cols+1,CV_32FC1);

	cvtColor(src,srcGray,CV_BGR2GRAY);
	cvtColor(dst,dstGray,CV_BGR2GRAY);

	matchTemplate(srcGray,dstGray,result,CV_TM_CCOEFF_NORMED);
	minMaxLoc(result,&minVal,&maxVal,&minLoc,&maxLoc);
	return maxVal;
}

void PF_Tracker::sfm()
{
	Mat prev = previousImgs.back();
	Rect prevRegion = previousRegions.back();
	Mat src1 = prev;//(prevRegion);

	Rect regionNew = targetRegion;
	regionNew.height = prevRegion.height;
	regionNew.width = prevRegion.width;
	Mat src2 = frame;//(regionNew);

	SiftFeatureDetector  siftdtc;
    vector<KeyPoint>kp1,kp2;
	siftdtc.detect(src1,kp1);
    //Mat outimg1;
    //drawKeypoints(src1,kp1,outimg1);
    //imshow("image1 keypoints",outimg1);
    KeyPoint kp;
	siftdtc.detect(src2,kp2);
    //Mat outimg2;
    //drawKeypoints(src2,kp2,outimg2);
   // imshow("image2 keypoints",outimg2);

	SiftDescriptorExtractor extractor;
    Mat descriptor1,descriptor2;
    BruteForceMatcher<L2<float>> matcher;
    vector<DMatch> matches;
    Mat img_matches;
    extractor.compute(src1,kp1,descriptor1);
    extractor.compute(src2,kp2,descriptor2);
 
 
    //imshow("desc",descriptor1);
    //cout<<endl<<descriptor1<<endl;
    matcher.match(descriptor1,descriptor2,matches);
 
    //drawMatches(src1,kp1,src2,kp2,matches,img_matches);
    //imshow("matches",img_matches);
	//waitKey();
	/*
	//imshow("src1", src1);
	//imshow("src2", src2);
	//waitKey();

	vector<cv::Mat> imgs; 
	vector<vector<cv::KeyPoint> > imgpts;
	vector<DMatch> matches;

	imgs.push_back(src1);
	imgs.push_back(src2);

	RichFeatureMatcher richMatcher(imgs, imgpts);
	richMatcher.MatchFeatures(0, 1, &matches);

	//OFFeatureMatcher offMatcher(imgs, imgpts);
	//offMatcher.MatchFeatures(0, 1, &matches);
	*/

	//

	
	if (matches.size() < 10)
	{
		cout << "not enough matches " << "\n";
		return;
	}

	vector<KeyPoint> imgpts1_good, imgpts2_good;
	Mat F = GetFundamentalMat(kp1, kp2, imgpts1_good, imgpts2_good, matches);
	//ShowMatchResult(src1, src2, kp1, kp2, matches);
	Mat_<double> E = K.t() * F * K; //according to HZ (9.12)
    //according to http://en.wikipedia.org/wiki/Essential_matrix#Properties_of_the_essential_matrix
    if(fabsf(determinant(E)) > 1e-07) 
	{
		cout << "det(E) != 0 : " << determinant(E) << "\n";
	}
	//else
	{
		Mat_<double> R1(3,3);
		Mat_<double> R2(3,3);
		Mat_<double> t1(1,3);
		Mat_<double> t2(1,3);	
		

		DecomposeEtoRandT(E,R1,R2,t1,t2);
		if(determinant(R1)+1.0 < 1e-09) 
		{
			//according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
			cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign" << endl;
			E = -E;
			DecomposeEtoRandT(E,R1,R2,t1,t2);
	    }
		cout<<t1<<endl;
		//cout<<R1<<endl;
		cout<<t2<<endl;
		//cout<<R2<<endl;

		//Mat test=Mat::eye(3,3,CV_64F);

		
		Mat xx;//=(R2-R2.t())/2;
		Rodrigues(R1,xx);
		 cout<<xx<<endl;
		 Mat vec=xx.t()*xx;
	
		 //cout<<vec.type()<<endl;
		 cout<<sqrt( vec.at<double>(0,0) )<<endl;

		 cout<<(R1-R1.t())/2<<endl;

		//t1*=50;
		//t2*=50;
	    Matx34d P(1,0,0,0,  //	R1(0,1),	R1(0,2),	t1(0),
				   0,1,0,0,//		 R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
				   0,0,1,0);		 //R1(2,0),	R1(2,1),	R1(2,2),	t1(2));;
	    Matx34d P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
							 R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
						     R1(2,0),	R1(2,1),	R1(2,2),	t1(2));
	    //cout << "Testing P1 " << endl << Mat(P1) << endl;

	    vector<CloudPoint> pcloud,pcloud1; 
	    vector<KeyPoint> corresp;
	    Mat distcoeff;
	    double reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, K.inv(), distcoeff, P, P1, pcloud, corresp);
	    double reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, K.inv(), distcoeff, P1, P, pcloud1, corresp);
	    vector<uchar> tmp_status;
	    if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) 
	    {
		   P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t2(0),
							 R1(1,0),	R1(1,1),	R1(1,2),	t2(1),
							 R1(2,0),	R1(2,1),	R1(2,2),	t2(2));
		   //cout << "Testing P1 "<< endl << Mat(P1) << endl;
		   pcloud.clear(); pcloud1.clear(); corresp.clear();
		   reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, K.inv(), distcoeff, P, P1, pcloud, corresp);
		   reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, K.inv(), distcoeff, P1, P, pcloud1, corresp);
		   if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) 
	       {
			   P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t1(0),
								 R2(1,0),	R2(1,1),	R2(1,2),	t1(1),
								 R2(2,0),	R2(2,1),	R2(2,2),	t1(2));
			   //cout << "Testing P1 "<< endl << Mat(P1) << endl;
			   pcloud.clear(); pcloud1.clear(); corresp.clear();
			   reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, K.inv(), distcoeff, P, P1, pcloud, corresp);
			   reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, K.inv(), distcoeff, P1, P, pcloud1, corresp);
			   if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) 
			   {
						P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
									 R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
									 R2(2,0),	R2(2,1),	R2(2,2),	t2(2));
						//cout << "Testing P1 "<< endl << Mat(P1) << endl;

						pcloud.clear(); pcloud1.clear(); corresp.clear();
						reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, K.inv(), distcoeff, P, P1, pcloud, corresp);
						reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, K.inv(), distcoeff, P1, P, pcloud1, corresp);
						
						if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) 
						{
							cout << "Shit." << endl; 
				        }
			   }			
		   }
	   }
	   //cout<<t1<<endl;
  	}

}


void PF_Tracker::compute_IH ()
{
	int sum = 0;
	cout << splithsv[0].type()<<endl;
	cout << splithsv[1].type()<<endl;


	/*for (int i = 1; i < searchArea.height; i++)
	{
		for (int j = 1; j < searchArea.width; j++)
		{
			int hIndex = splithsv[0].at<int>(i, j) * NH/ 181;
			cout <<hIndex<<",";
		}
		cout <<endl;
	}*/


	vector <vector < vector<int> > > hisIntegral(NH * NS, vector < vector<int> > (searchArea.height, vector<int> (searchArea.width,0)));
	/*for (int i = 0; i < NH * NS;i ++)
	{
		vector < vector<int> > m(searchArea.height, vector<int> (searchArea.width,0));
		hisIntegral.push_back(m);
	}*/
	//int hIndex = splithsv[0].at<uchar>(0, 0) * NH/ 181;
	//int sIndex = splithsv[1].at<uchar>(0, 0) * NS/ 256;
	//int k = 10 * hIndex + sIndex; // 10 == NH

	int loc00 = 10 * splithsv[0].at<uchar>(0, 0) * NH/ 181 + splithsv[1].at<uchar>(0, 0) * NS/ 256;
	hisIntegral[loc00][0][0] = 1;
	for (int i = 1; i < searchArea.height; i++)
	{
		int hIndex = splithsv[0].at<uchar>(i, 0) * NH/ 181;
		int sIndex = splithsv[1].at<uchar>(i, 0) * NS/ 256;
		int k = 10 * hIndex + sIndex; // 10 == NH
		for (int l = 0; l < NH * NS; l++)
		{
				hisIntegral[l][i][0] =    hisIntegral[l][i-1][0];
										
		}
		hisIntegral[k][i][0] += 1;		
	}

	for (int i = 1; i < searchArea.width; i++)
	{
		int hIndex = splithsv[0].at<uchar>(0, i) * NH/ 181;
		int sIndex = splithsv[1].at<uchar>(0, i) * NS/ 256;
		int k = 10 * hIndex + sIndex; // 10 == NH
		for (int l = 0; l < NH * NS; l++)
		{
				hisIntegral[l][0][i] =    hisIntegral[l][0][i-1];
										
		}
		hisIntegral[k][0][i] += 1;		
	}

	for (int i = 1; i < searchArea.height; i++)
	{
		for (int j = 1; j < searchArea.width; j++)
		{
			int hIndex = splithsv[0].at<uchar>(i, j) * NH/ 181;
			int sIndex = splithsv[1].at<uchar>(i, j) * NS/ 256;
			int k = 10 * hIndex + sIndex; // 10 == NH
			
			for (int l = 0; l < NH * NS; l++)
			{
				hisIntegral[l][i][j] =    hisIntegral[l][i][j-1]
									     + hisIntegral[l][i-1][j]
										 - hisIntegral[l][i-1][j-1]; 
			}
			//if (k > 99) k = 99;
			hisIntegral[k][i][j] += 1;		
			//cout <<i<<","<<j<<endl;
		}
	}
	for (int i = 0; i < NH * NS;i ++)
	{
		sum += hisIntegral[i][searchArea.height - 1][searchArea.width - 1];
	}
	cout<<sum<<endl;
	cout<<searchArea.area()<<endl;
	/*IIV_T.clear();
	inRange(hsv_frame, Scalar(0, S_THRESH*255, V_THRESH*255), Scalar(181, 256, 256), mask);
	cout << countNonZero(mask) <<endl;
	cout << hsv_frame.cols * hsv_frame.rows<<endl;
	int sum = 0;
	for (int i=0; i<NH; i++)
	{
		Mat binh;
	    inRange(splithsv[0],180*i/NH,180*(i+1)/NH,binh);
		//binh.convertTo(binh,-1,1.0,-254);
		for (int j=0; j<NS; j++) 
		{
	          Mat bins;
	          inRange(splithsv[1],255*i/NS,255*(i+1)/NS,bins);
	         // bins.convertTo(bins,-1,1.0,-254);
			 // Mat binbin=10*binh+bins;
			  Mat binbin=binh&bins;
			  //binbin&=mask;
			  binbin /= 255;
			  Mat output;
			  integral(binbin,output);
			  sum += output.at<int>(searchArea.height,searchArea.width);
			  IIV_T.push_back(output);
		 }
	 }
	cout<<sum<<endl;*/
	
}


void PF_Tracker::compute_histogram (Rect r,vector < int >& hist)
{
	r.x-=searchArea.x;
	r.y-=searchArea.y;
	Rect boud=Rect(0,0,searchArea.width,searchArea.height);
	r&=boud;

	int left , up , diag;
	double sum = 0;
	double z;
	for (vector < Mat >::iterator it = IIV_T.begin() ; it != IIV_T.end() ; it++ ) 
	{
		if ( r.x == 0 ) {
			left = 0;
			diag = 0;
		} else {
			left = (*it).at<int>(r.y+r.height , r.x );
		}

		if ( r.y == 0 ) {
			up = 0;
			diag = 0;
		} else {
			up = (*it).at<int>(r.y , r.x +r.width);
		
		}
		if ( r.x > 0 && r.y > 0 ) {
			diag = (*it).at<int>(r.y , r.x );
		}
		z =(*it).at<int>(r.y+r.height , r.x +r.width) - left - up + diag;
		hist.push_back(z);
		sum += z;
	}
}