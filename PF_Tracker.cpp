#include "PF_Tracker.h"
#include "select.h"
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

PF_Tracker::PF_Tracker():frameNum(0)
	                    ,frameheight(0)
						,framewidth(0)
						,num_particles(300)   //粒子数
{
	/* parse command line and initialize random number generator */
   gsl_rng_env_setup();
   rng = gsl_rng_alloc( gsl_rng_mt19937 );
   gsl_rng_set( rng, time(NULL) );
}

void PF_Tracker::process(Mat & input,Mat & output)
{
	frame=input.clone();
	cvtColor(frame,hsv_frame,CV_BGR2HSV );

	/* allow user to select object to be tracked in the first frame */
	frameNum++;
	if(frameNum==1)     //手动选择跟踪目标，在初始帧
	{
		framewidth=frame.cols;
		frameheight=frame.rows;
		boundary=Rect(0,0,framewidth-1,frameheight-1);

		cvNamedWindow("SelectObject", CV_WINDOW_AUTOSIZE);
		imshow("SelectObject",frame);
	    cvSetMouseCallback("SelectObject", SelectObject, NULL ); 
		while(!gotBB)
	    {
			if (cvWaitKey(10) == 27)
			    break;
		}
		region=box;
		ref_histo = compute_ref_histos( hsv_frame, region );
		particles = init_distribution( region, ref_histo, num_particles );
		output=frame.clone();
		return;
	}
	 /* perform prediction and measurement for each particle */
	for( int j = 0; j < num_particles; j++ )
	{
	      particles[j] = transition( particles[j], framewidth, frameheight, rng );
	      float s = particles[j].s;
	      particles[j].w = likelihood( hsv_frame, cvRound(particles[j].y),
					   cvRound( particles[j].x ),
					   cvRound( particles[j].width * s ),
					   cvRound( particles[j].height * s ),
					   particles[j].histo );
	}
	  
	  /* normalize weights and resample a set of unweighted particles */
	  normalize_weights( particles, num_particles );//归一化，使所有粒子权重和为1
	  new_particles = resample( particles, num_particles );
	  free( particles );
	  particles = new_particles;


	   /* display all particles if requested */
      qsort( particles, num_particles, sizeof( particle ), particle_cmp );
      if( 0)
	  for(int  j = num_particles - 1; j > 0; j-- )
	  {
	    color = CV_RGB(255,255,0);
	    //display_particle( frames[i], particles[j], color );
	  }
      
      /* display most likely particle */
      color = CV_RGB(0,255,255);
      display_particle( frame, particles[0], color );
	  output=frame.clone();
     
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

float PF_Tracker::likelihood(Mat& img, int r, int c,int w, int h, histogram* ref_histo )
{
  //IplImage* tmp;
  histogram* histo;
  float d_sq;

  /* extract region around (r,c) and compute and normalize its histogram */
  Rect re=Rect( c - w / 2, r - h / 2, w, h );
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
  //cout <<exp( -LAMBDA * d_sq )<<endl;
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

void PF_Tracker::display_particle( Mat img, particle p, CvScalar color )
{
  int x0, y0, x1, y1;

  x0 = cvRound( p.x - 0.5 * p.s * p.width );
  y0 = cvRound( p.y - 0.5 * p.s * p.height );
  x1 = x0 + cvRound( p.s * p.width );
  y1 = y0 + cvRound( p.s * p.height );
  
 rectangle( img, cvPoint( x0, y0 ), cvPoint( x1, y1 ), color, 2, 8, 0 );
}