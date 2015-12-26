#include "PF_Tracker.h"
#include "select.h"
#include "OFFeatureMatcher.h"
#include "FindCameraMatrix.h"
#include "Triangulation.h"
#include "RichFeatureMatcher.h"
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>
#include <math.h>
#include <algorithm>
#include <fstream>

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

bool particle_cmp( const  particle &p1, const  particle & p2 )

{
	return p1.w > p2.w;
}


bool patchCmp(const Patch & p1, const Patch & p2)
{
	return p1.confidence > p2.confidence;
}
bool patchCmp1(const Patch & p1, const Patch & p2)
{
	return p1.confidence < p2.confidence;
}

float absSum(Mat &m)
{
	float sum;
	for (int i = 0; i< m.rows;i++)
	{
		for(int j = 0;j <m.cols;j++)
		{
			sum+= abs( m.at<float>(i,j) );
		}
	}
	return sum;
}

//static const double distcoeff[]={0.03489 ,  -0.25811,   -0.00048,   0.00330,  0.00000};

PF_Tracker::PF_Tracker():frameNum(0)
	                    ,frameheight(0)
						,framewidth(0)
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
   showColors.push_back(YELLOW);
   showColors.push_back(BLUE);
   showColors.push_back(GREEN);
   showColors.push_back(RED);
   showColors.push_back(PURPLE);

   W = Mat::zeros(patchNum,patchNum,CV_32F);
   T.create(patchNum, 2,CV_32F);


   ifstream gtFile(libraryPath+vedioName+"//groundtruth.txt",ios::in);
   string str;
   while(gtFile >> str)
   {
	   //cout<<str<<endl;
	   vector <int> xywh;
	   int i = 0,pre=0;
	   while (i < str.size())
	   {
		   if (str[i] == ',')
		   {
			   xywh.push_back(atoi(str.substr(pre,i - pre).c_str()));
			   pre = i+1;
		   }
		   i++;
	   }
	   xywh.push_back(atoi(str.substr(pre).c_str()));
	   Rect r(xywh[0], xywh[1], xywh[2], xywh[3]);
	   groundTruth.push_back(r);
   }


}

void generateIndexes(vector <vector<int>> &OneSituation, vector <int> &sizes, int index, vector<int> &base)
{
	if (index == sizes.size())
	{
		OneSituation.push_back(base);
		return ;
	}
	for (int i = 0; i < sizes[index]; i++)
	{
		base.push_back(i);
		generateIndexes(OneSituation, sizes, index+1, base);
		base.pop_back();
	}

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
		init();
		
		stringstream ss;
		string str;
		ss<<frameNum;
		ss>>str;
		imwrite(outputImgPath+str+".jpg",showImg);
		return;
	}
	/*if(frameNum <20  && frameNum >2 )
	{
		return ;
	}*/
	//waitKey();
	//if ( frameNum == 5)
	{
		searchArea.x = targetRegion.x - targetRegion.width;
        searchArea.y = targetRegion.y - targetRegion.height;
	    searchArea.width = targetRegion.width * 3;
	    searchArea.height = targetRegion.height * 3;
	    searchArea &= boundary;
		roi=frame(searchArea);
	    cvtColor(roi,hsv_frame, CV_BGR2HSV);
        split(hsv_frame, splithsv);

		//cout<<"begin() compute_IH"<<endl;
        compute_IH();
		//imshow("roi",roi);
		//cout<<"end() compute_IH"<<endl;
		//rectangle(showImg, searchArea,RED);
		//cout<<"begin() "<<endl;
		vector <vector <int>> mcmc(patchNum, vector <int>()); 
		for (int i = 0; i < patchNum; i++)
		{
			for( int j = 0; j < num_particles; j++ )
			{
				vpatches[i].particles[j] = transition( vpatches[i].particles[j], framewidth, frameheight, rng );
				//float s = particles[j].s;
				vpatches[i].particles[j].w = likelihood(  
															cvRound(vpatches[i].particles[j].y),
															cvRound( vpatches[i].particles[j].x ),
															cvRound( vpatches[i].particles[j].width  ),
															cvRound( vpatches[i].particles[j].height ),
															vpatches[i].particles[j].histogram ,
															vpatches[i].particles[j].distance
														);
				//rectangle(showImg, Rect(vpatches[i].particles[j].x, vpatches[i].particles[j].y, vpatches[i].particles[j].width,vpatches[i].particles[j].height), showColors[i]);	
			}
			normalize_weights(vpatches[i].particles);

			for( int j = 0; j < num_particles; j++ )
			{
				int num = vpatches[i].particles[j].w *100;
				for(int k = 0; k < num; k++)
				{
					mcmc[i].push_back(j);
				}
				if (num == 0 && vpatches[i].particles[j].w > 0.005)
				{
					mcmc[i].push_back(j);
				}
				//cout << vpatches[i].particles[j].w << ",";
				//std::random_shuffle( mcmc[i].begin(), mcmc[i].end() );
			}
			if (mcmc[i].empty())
			{
				cout << vpatches[i].r<<endl;
				//showVector(
			}
			//cout << endl;
		}
		//cout<<"begin() "<<endl;


		float MAX = FLT_MAX;
		float maxSim, maxW;
		vector <int> resultIndexes;

		for (int i = 0;i < 10000; i++)
		{
			//resultIndexes.clear();
			vector <int> caseIndexes;
			//Mat showS = frame.clone();
			float sim = 0;

		 	for (int j = 0; j < patchNum; j++)
		    {
				int randN = rand() % mcmc[j].size();
				caseIndexes.push_back(mcmc[j][randN]);
				T.at<float>(j,0) = vpatches[j].particles[mcmc[j][randN]].x - vpatches[j].particles[mcmc[j][randN]].width/2 ;//vvpatches[j][OneSituation[i][j]].r.x;
				T.at<float>(j,1) = vpatches[j].particles[mcmc[j][randN]].y - vpatches[j].particles[mcmc[j][randN]].height/2 ;
				sim += vpatches[j].particles[mcmc[j][randN]].distance;
				//swap(mcmc[j][0],mcmc[j][randN]);
				//random_shuffle( mcmc[j].begin(), mcmc[j].end() );
				//circle(showS, Point(       vpatches[j].particles[mcmc[j][0]].x,vpatches[j].particles[mcmc[j][0]].y),4,RED);          
			}
			Mat delta = T - W*T;
			double ret = absSum(delta);

			if ( ret+sim < MAX)
			{
				MAX = ret+sim;
				maxSim = sim;
				maxW = ret;
				resultIndexes = caseIndexes;
			}
		}

		//cout<<"begin() "<<endl;
		//cout<<"Max : "<<MAX<<" maxSim : "<<maxSim<<" maxW : "<<maxW<<endl;
		//int dx=0,dy=0;
		for(int j=0; j< patchNum;j++)
		{ 
			
			vpatches[j].particles = resample( vpatches[j].particles );
			//dx += vpatches[j].particles[0].x - vpatches[j].particles[0].width/2 - vpatches[j].r.x;
			//dy += vpatches[j].particles[0].y - vpatches[j].particles[0].height/2- vpatches[j].r.y;
			vpatches[j].r.x = vpatches[j].particles[0].x - vpatches[j].particles[0].width/2;
			vpatches[j].r.y = vpatches[j].particles[0].y - vpatches[j].particles[0].height/2;

			//rectangle(showImg, vpatches[j].r, showColors[j]);
			//circle(showImg, Point(  vpatches[j].particles[resultIndexes[j]].x,vpatches[j].particles[resultIndexes[j]].y),4,RED);
		}
		//targetRegion.x -= dx / patchNum;
		//targetRegion.y -= dy / patchNum;
		Rect boundRectNew = calcBoundingRect();
	   //cout<<"end1() "<<endl;
		
	    pToObject1.create(searchArea.height, searchArea.width, CV_64FC1);
	    for (int i = 0; i < searchArea.height; i++)
	    {
		  for (int j = 0; j < searchArea.width; j++)
		  {
				int hIndex = splithsv[0].at<uchar>(i, j) * NH/ 181;
			    int sIndex = splithsv[1].at<uchar>(i, j) * NS/ 256;
				int k = 10 * hIndex + sIndex;
				if (mask.at<uchar>(i,j) == 0)
				{
				    pToObject1.at< double>(i,j)= 0;
				}
				else
				{
					 pToObject1.at<double>(i,j)  = log(hisTargetNorm[k]) - log(hisBackgroundNorm[k]) ;//hisTarget[k]*255/(hisSearchArea[k]);//*3/2 ;
				}
		  }
	   }
	   imshow("log", pToObject1 );
	   //hisTarget=hisTargetNew;
	   	cout<<"end2() "<<endl;
	   //vector < Rect> vRegionSearchRect;
		Rect regionSearchRect = targetRegion;
		regionSearchRect.x = targetRegion.x -patchSize;
		regionSearchRect.y = targetRegion.y -patchSize;
		float gSim = FLT_MIN;
		Rect result = targetRegion;
		for (; regionSearchRect.y < targetRegion.y+patchSize ; regionSearchRect.y += 4)  
		{
			for (; regionSearchRect.x < targetRegion.x+patchSize; regionSearchRect.x += 4)
			{
				  //circle(showImg, Point( regionSearchRect.x, regionSearchRect.y) ,2 ,RED);
				  //rectangle(showImg, regionSearchRect,showColors[rand()%4]);
				 // imshow("case", showImg(regionSearchRect));
				  //waitKey();
				  if ( ( boundRectNew&regionSearchRect) == boundRectNew )
				  {
					 //vRegionSearchRect.push_back(regionSearchRect);
					 //vector<int> his;
					 // compute_histogram (regionSearchRect,his);
					 //float dis = distance(hisTarget, his);
					  Rect s1 = regionSearchRect;
					  s1.x -= searchArea.x;
					  s1.y -= searchArea.y;
					  s1 &= Rect(0, 0, searchArea.width, searchArea.height);
					  cout<<"mean before "<<endl;
					  Scalar s = mean(pToObject1(s1));
					  cout<<"mean() end; "<<endl;
					 double score = s[0];
					 if (score > gSim)
					 {
						 gSim = score;
						 result = regionSearchRect;
					 }
				  }
			}
			regionSearchRect.x = targetRegion.x -patchSize;
		}
		cout<<"end3() "<<endl;
		//targetRegion = result;
		targetRegion = groundTruth[frameNum-1];

		compute_histogram(targetRegion, hisTarget);
		vector<int> hisTargetNew, hisSearchAreaNew,hisBackgroundNew;
	    compute_histogram(targetRegion, hisTargetNew);
	    compute_histogram(searchArea, hisSearchAreaNew);
	    for (int i = 0; i< hisTarget.size(); i++)
	    {
		    hisBackgroundNew.push_back(hisSearchAreaNew[i] - hisTargetNew[i]);
	    }
	    vector < double>  hisTargetNormNew = NormHis(hisTargetNew);
	    vector < double>  hisBackgroundNormNew = NormHis(hisBackgroundNew);
	    weightAdd( hisTargetNorm,  hisTargetNormNew);
        weightAdd( hisBackgroundNorm,  hisBackgroundNormNew);

			cout<<"end4() "<<endl;

	/*
	   float minConfidence = FLT_MAX;
	   int deleteIndex = -1;

	   for (int i = 0;i < patchNum; i++)
	   {
		   double confidence = 0.0;
		   for(int j=0; j<num_particles;j++)
		   {
			   circle(showImg,Point(vpatches[i].particles[j].x, vpatches[i].particles[j].y), 2,showColors[i]);
			   Point pt;
			   pt.x = vpatches[i].particles[j].x -searchArea.x;
			   pt.y = vpatches[i].particles[j].y -searchArea.y;
			   if ( pt.x > 0 && pt.y > 0 &&  
				               ( pt.x < searchArea.x+searchArea.width) &&
							  (pt.y <searchArea.y+searchArea.height) )
			   {
				   confidence +=  pToObject1.at<double>(pt.x,pt.y) * vpatches[i].particles[j].w;
			   }
		   }
		   vpatches[i].confidence = confidence;
		   calcBoundingRect(vpatches[i]);
		   rectangle(showImg, vpatches[i].boundRect, showColors[i], 1); 
		   //cout<<"vpatches[i].confidence : "<<vpatches[i].confidence<<endl;

		   if (minConfidence > vpatches[i].confidence)
		   {
			   minConfidence = vpatches[i].confidence;
			   deleteIndex = i;
		   }
	   }

	   updatePatches(deleteIndex);*/
		//targetRegion.x -=boundRectNew.x - boundRect.x;
        //targetRegion.y -=boundRectNew.y - boundRect.y;
		//boundRect = boundRectNew;
	   rectangle(showImg, targetRegion, GREEN, 1); 
	   calcW();
			cout<<"end5() "<<endl;
		return;
	}
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
void PF_Tracker::init_distribution( Patch &p)
{
  Rect region = p.r;
  float x, y;
  int  j,  k = 0;
  
  int width = region.width;
  int height = region.height;
  x = region.x + width / 2;
  y = region.y + height / 2;

  for( j = 0; j < num_particles; j++ )
  {
	  particle pTemp;
	  pTemp.x0 = pTemp.xp = pTemp.x = x;
	  pTemp.y0 = pTemp.yp = pTemp.y = y;
	  pTemp.sp = pTemp.s = 1.0;
	  pTemp.width = width;
	  pTemp.height = height;
	  //pTemp.histo = histo;
	  pTemp.histogram = p.histogram;
	  p.particles.push_back(pTemp);
  }
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
  x = p.x+5.0*gsl_ran_gaussian( rng, TRANS_X_STD ) ;
  pn.x = MAX( 0.0, MIN( (float)w - 1.0, x ) );
  y = p.y + 5.0*gsl_ran_gaussian( rng, TRANS_Y_STD ) ;
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
  pn.histogram = p.histogram;
  //pn.histo = p.histo;
  pn.w = 0;
  return pn;
}



/*
  Normalizes particle weights so they sum to 1
  
  @param particles an array of particles whose weights are to be normalized
  @param n the number of particles in \a particles
*/
void PF_Tracker::normalize_weights( vector <particle> & vp )
{
  float sum = 0;
  int i;
  for( i = 0; i < num_particles; i++ )
    sum += vp[i].w;
  for( i = 0; i < num_particles; i++ )
    vp[i].w /= sum;
}

float PF_Tracker::likelihood( int r, int c,int w, int h, vector <int>  &ref_histo,float &dt)
{
  /* extract region around (r,c) and compute and normalize its histogram */
  Rect re=Rect( c - w / 2, r - h / 2, w, h );
  re&=boundary;

  vector <int> histogram;
  compute_histogram(re, histogram);
  dt = distance ( ref_histo , histogram );

  //cout <<exp( -LAMBDA * d_sq )<<endl;
  return exp( -LAMBDA * dt );
}


float PF_Tracker::distance ( const vector <int> &v1 , const vector <int> &v2 )
{
	vector <float> vv1,vv2;
	float sum1=0, sum2=0;
	for (int i = 0; i< v1.size();i++)
	{
		sum1 += v1[i];
		sum2 += v2[i];
	}
	for (int i = 0; i< v1.size();i++)
	{
		vv1.push_back(v1[i]/sum1);
		vv2.push_back(v2[i]/sum2);
	}
	float sum = 0;
	for( int i = 0; i < v1.size(); i++ )
		sum += sqrt( vv1[i]*vv2[i] );
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

vector <particle> PF_Tracker::resample(  vector <particle> &particles )
{
	vector <particle> new_particles;
	sort(particles.begin(), particles.end(), particle_cmp);
    for( int i = 0; i < num_particles; i++ )
    {
      int np = cvRound( particles[i].w * num_particles );
      for( int j = 0; j < np; j++ )
	  {
		  new_particles.push_back(particles[i]);
		  if( new_particles.size() ==  num_particles)
		  {
			  return new_particles;
		  }
	   }
    }
    while( new_particles.size() < num_particles )
	{
		 new_particles.push_back(particles[0]);
	}
    return new_particles;
}


void PF_Tracker:: makePatches()
{
	vector <Patch> initPatches;
	Rect region=targetRegion;
	//region.x-=searchArea.x;
    //region.y-=searchArea.y;
	Rect r(region.x+1, region.y+1, patchSize, patchSize);

	
	while (r.y < region.y+region.height - patchSize)
	{

		while (r.x < region.x+region.width - patchSize)
		{
			Patch tmp;
			tmp.r = r;
			Rect rt=r;
			rt.x -= searchArea.x;
			rt.y -= searchArea.y;
			Scalar s = mean(pToObject1(rt));
			tmp.confidence = s[0];
			initPatches.push_back(tmp);
			r.x += patchSize;
		}
		r.y += patchSize;
		r.x = region.x+1;
	}
	sort(initPatches.begin(), initPatches.end(), patchCmp);
    int k = 0;


	//vector <Patch > initPatches1;
	while (vpatches.size() < patchNum && k < initPatches.size())
	{
		bool canAdd = true;
		for (int i = 0; i < vpatches.size(); i++)
		{
			if (distance(vpatches[i].r,initPatches[k].r) < patchSize*2)
			{
				canAdd = false;
				break;
				//initPatches1.push_back(initPatches[k]);
			}
		}
		if (canAdd)
		{
			vpatches.push_back(initPatches[k]);
		}
		k++;
	}
	
	Focus.x = 0;
	Focus.y = 0;

	for (int i = 0; i < vpatches.size(); i++)
	{
			//vpatches[i].r.x+= searchArea.x;
			//vpatches[i].r.y+= searchArea.y;
			compute_histogram (vpatches[i].r,vpatches[i].histogram);	
			rectangle(showImg, vpatches[i].r, GREEN, 2); 
			Focus.x += vpatches[i].r.x;
			Focus.y += vpatches[i].r.y;
	}
	Focus.x /= vpatches.size();
	Focus.y /= vpatches.size();

	for (int i = 0; i < patchNum; i++)
	{
		init_distribution(vpatches[i]);
	}
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
	//cout<< "compute_IH  begin()"<<endl;
	inRange(hsv_frame, Scalar(0, S_THRESH*255, V_THRESH*255), Scalar(181, 256, 256), mask);
	int sum = 0;
	int loc00 = 10 * splithsv[0].at<uchar>(0, 0) * NH/ 181 + splithsv[1].at<uchar>(0, 0) * NS/ 256;
	//if (mask.at<uchar>(0,0) )
	//	hisIntegral[loc00][0][0] = 1;
	for (int i = 1; i < searchArea.height; i++)
	{
		int hIndex = splithsv[0].at<uchar>(i, 0) * NH/ 181;
		int sIndex = splithsv[1].at<uchar>(i, 0) * NS/ 256;
		int k = 10 * hIndex + sIndex; // 10 == NH
		for (int l = 0; l < NH * NS; l++)
		{
				hisIntegral[l][i][0] =    hisIntegral[l][i-1][0];
										
		}
		if (mask.at<uchar>(i,0) )
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
		if (mask.at<uchar>(0,i) )
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
			if (mask.at<uchar>(i,j) )
				hisIntegral[k][i][j] += 1;		
			
		}
	}
	for (int i = 0; i < NH * NS;i ++)
	{
		sum += hisIntegral[i][searchArea.height - 1][searchArea.width - 1];
	}
	//cout<< "compute_IH end"<<endl;
	
}

int PF_Tracker::distance(const Rect& r1, const Rect& r2)
{
	return std::max( abs(r1.x - r2.x), abs(r1.y - r2.y));
}

int PF_Tracker::l2distance(const Rect& r1, const Rect& r2)
{
	return (r1.x - r2.x)*(r1.x - r2.x)+(r1.y - r2.y)*(r1.y - r2.y);
}

void PF_Tracker::compute_histogram (Rect r,vector < int >& hist)
{
	hist.clear();
	r.x-=searchArea.x;
	r.y-=searchArea.y;
	Rect boud=Rect(0,0,searchArea.width-1,searchArea.height-1);
	r&=boud;

	if (r.area() < 10.0)
	{
		for (int l = 0; l < NH * NS; l++)
	    {
			hist.push_back(0);
		}
		return ;
	}

	int left , up , diag;
	//double sum = 0;
	int z;
	//for (vector < Mat >::iterator it = IIV_T.begin() ; it != IIV_T.end() ; it++ ) 
	for (int l = 0; l < NH * NS; l++)
	{
		if ( r.x == 0 ) {
			left = 0;
			diag = 0;
		} else {
			left = hisIntegral[l][r.y+r.height-1][r.x-1];
		}

		if ( r.y == 0 ) {
			up = 0;
			diag = 0;
		} else {
			up = hisIntegral[l][r.y-1][r.x +r.width-1];
		
		}
		if ( r.x > 0 && r.y > 0 ) {
			diag = hisIntegral[l][r.y-1][r.x-1];
		}
		z =hisIntegral[l][r.y+r.height][r.x +r.width] - left - up + diag;
		hist.push_back(z);
		//sum += z;
	}
}


void PF_Tracker::init()
{
	    searchArea.x = targetRegion.x - targetRegion.width;
        searchArea.y = targetRegion.y - targetRegion.height;
	    searchArea.width = targetRegion.width * 3;
	    searchArea.height = targetRegion.height * 3;
	    searchArea &= boundary;
		
	    for (int i = 0; i < NH * NS;i ++)
	    {
			vector < vector<int> > m(frameheight, vector<int> (framewidth,0));
			//vector < vector<int> > m(searchArea.height, vector<int> (searchArea.width,0));
			hisIntegral.push_back(m);
		}
		roi=frame(searchArea);
	    cvtColor(roi,hsv_frame, CV_BGR2HSV);
        split(hsv_frame, splithsv);
        compute_IH();
		
		//cout << "1" <<endl;
		compute_histogram(targetRegion, hisTarget);
		compute_histogram(searchArea, hisSearchArea);
		for (int i = 0; i< hisTarget.size(); i++)
		{
			hisBackground.push_back(hisSearchArea[i] - hisTarget[i]);
		}
		//cout << "2" <<endl;
		hisTargetNorm = NormHis(hisTarget);
		hisBackgroundNorm = NormHis(hisBackground);
		pToObject1.create(searchArea.height, searchArea.width, CV_64FC1);
		for (int i = 0; i < searchArea.height; i++)
		{
			for (int j = 0; j < searchArea.width; j++)
			{
				int hIndex = splithsv[0].at<uchar>(i, j) * NH/ 181;
			    int sIndex = splithsv[1].at<uchar>(i, j) * NS/ 256;
				int k = 10 * hIndex + sIndex;
				if (mask.at<uchar>(i,j) == 0)
				{
				    pToObject1.at< double>(i,j)= 0;
				}
				else
				{
					 pToObject1.at<double>(i,j)  = log(hisTargetNorm[k]) - log(hisBackgroundNorm[k]) ;//hisTarget[k]*255/(hisSearchArea[k]);//*3/2 ;
				}
			}
		}
		imshow("log", pToObject1 );
		//cout << "3" <<endl;
		makePatches();
		//cout << "4" <<endl;
		boundRect = calcBoundingRect();
		rectangle(showImg, targetRegion, GREEN, 1); 
		rectangle(showImg, boundRect, GREEN, 1); 
		calcW();
		
		preFrame = frame;
		/*
		Mat hsvTemp;
		vector<Mat> splithsvTemp;
		cvtColor(frame,hsvTemp, CV_BGR2HSV);
	    split(hsvTemp, splithsvTemp);
		Mat pToObjectShow(boundary.height+1, boundary.width+1, CV_8UC3);
		for (int i = 0; i <= boundary.height; i++)
		{
			for (int j = 0; j <= boundary.width; j++)
			{
				int hIndex = splithsvTemp[0].at<uchar>(i, j) * NH/ 181;
			    int sIndex = splithsvTemp[1].at<uchar>(i, j) * NS/ 256;
				int k = 10 * hIndex + sIndex;
				pToObjectShow.at<cv::Vec3b>(i,j)[1] = 0;
				pToObjectShow.at<cv::Vec3b>(i,j)[0] = 0;
				if (hisSearchArea[k] == 0)
				{
					//pToObject.at<cv::Vec3b>(i,j)[2]= 0;
				    pToObjectShow.at<cv::Vec3b>(i,j)[2]= 0;
				}
				else
				{
					pToObjectShow.at<cv::Vec3b>(i,j)[2]= hisTarget[k]*255/(hisSearchArea[k]);//*3/2 ;
					//pToObject.at<cv::Vec3b>(i,j)[2] = hisTarget[k]*255/(hisTarget[k]+hisBackground[k]) *3 /2;
				}

			}
		}
		//imwrite("E:\\论文结果图片\\初始化\\" + vedioName + "0.jpg",showImg);
		//imwrite("E:\\论文结果图片\\初始化\\" + vedioName + "1.jpg",pToObjectShow);
		*/
		//imshow("mask", mask);
		//imshow("pToObject", pToObject);
		//imwrite("E:\\论文结果图片\\初始化\\" + vedioName + "2.jpg",showImg);
		//ref_histo = compute_ref_histos(hsv_frame, targetRegion);
		//particles = init_distribution(targetRegion, ref_histo, num_particles);
}


Mat PF_Tracker::clacWij(int row, vector <int> indexes)
{
	Mat Q1(3,3,CV_32F);
	Q1.at<float>(2,0) = 1.0;
	Q1.at<float>(2,1) = 1.0;
	Q1.at<float>(2,2) = 1.0;

	if (indexes.size() != 3 )
	{
		cout<<"error : indexes.size() != 3"<<endl;
		return Mat::zeros(3,1,CV_32F);
	}

	Q1.at<float>(0,0) = vpatches[indexes[0]].r.x;
	Q1.at<float>(1,0) = vpatches[indexes[0]].r.y;

	Q1.at<float>(0,1) = vpatches[indexes[1]].r.x;
	Q1.at<float>(1,1) = vpatches[indexes[1]].r.y;

	Q1.at<float>(0,2) = vpatches[indexes[2]].r.x;
	Q1.at<float>(1,2) = vpatches[indexes[2]].r.y;

	Mat P1(3,1,CV_32F);
	P1.at<float>(0,0) = vpatches[row].r.x;
	P1.at<float>(1,0) = vpatches[row].r.y;
	P1.at<float>(2,0) = 1.0;

	return Q1.inv()*P1;
	
	//vector <float> ret;
	//ret.push_back(W1.at<float>(0,0));
	//ret.push_back(W1.at<float>(0,0));
	//ret.push_back(W1.at<float>(0,0));
}

void PF_Tracker::calcW()
{
	for (int i = 0 ; i < patchNum; i++)
	{
		vector <int> distances;
		for (int j = 0; j < patchNum; j++)
		{
			if (j == i) continue;
			int dis = l2distance(vpatches[i].r, vpatches[j].r);
			distances.push_back(dis);
		}
		sort(distances.begin(), distances.end());
		int t = distances[2];
		vector <int> indexes;
		for (int j = 0; j < patchNum; j++)
		{
			if (j == i) continue;
			int dis = l2distance(vpatches[i].r, vpatches[j].r);
			if(dis <=t ) 
			{
				indexes.push_back(j);
			}
		}
		indexes.resize(3);
		neighbors.push_back(indexes);
	}
	for (int i = 0 ; i < patchNum; i++)
	{
		//showVector(neighbors[i]);
		Mat wi = clacWij(i, neighbors[i]);
		for (int j = 0; j < neNum; j++)
		{
			W.at<float>(i,neighbors[i][j]) = wi.at<float>(j,0);
		}
	}
	//cout<<W<<endl;
}

void score(vector <vector < Patch> >  & vvpatches, vector <int> &sizes, int index)
{

}


float PF_Tracker::solveIP(Rect region, vector <vector < Patch> >  &vvpatches,vector <int> & locationIndexes)
{
	vector<int> sizes;
	for (int i = 0; i < vvpatches.size(); i++)
	{
		sizes.push_back(vvpatches[i].size());
	}

	vector <vector<int>> OneSituation;
    int index = 0;
	vector<int> base;
    generateIndexes(OneSituation, sizes, index, base);
	//cout<<"generateIndexes done"<<endl;

	float MAX = 100000.0;
	int answerIndex = 0;

	//Mat showACase=frame.clone();

	for (int i = 0;i<OneSituation.size();i++)
	{

		    float sim = 0;

			for(int j=0; j< patchNum;j++)
			{
				//rectangle(showImg,vvpatches[j][OneSituation[i][j]].r,showColors[j]);
				T.at<float>(j,0) = vvpatches[j][OneSituation[i][j]].r.x;
				T.at<float>(j,1) = vvpatches[j][OneSituation[i][j]].r.y;
				sim += vvpatches[j][OneSituation[i][j]].confidence;
			}
			//cout<<T<<endl;
			Mat delta = T - W*T;
			double ret = absSum(delta);
			if ( ret+sim < MAX)
			{
				MAX = ret+sim;
				answerIndex =i;

			}
			//cout<<absSum(delta)<<endl;
			//showVector(OneSituation[i]);//.size()<<endl;
	}
	locationIndexes = OneSituation[answerIndex];
	//cout<<"MAX"<<MAX<<endl;
	/*for(int j=0; j< patchNum;j++)
	{
			rectangle(showImg,vvpatches[j][OneSituation[answerIndex][j]].r,RED);
			circle(showACase,
				   Point( 
				          vvpatches[j][OneSituation[answerIndex][j]].r.x, vvpatches[j][OneSituation[answerIndex][j]].r.y
						 ) ,
						 4,
						 showColors[j]
			);
	}*/
	//cout<<"draw done"<<endl;
	//rectangle(showACase, region,RED);
	//imshow("showACase",showACase);
	//waitKey(500);
	return MAX;
}
	/*
		vector <vector<int>> OneSituation;
		int index = 0;
		vector<int> base;
		generateIndexes(OneSituation, sizes, index, base);

		float MAX = 100000.0;
		int answerIndex = 0;
		
		for (int i = 0;i<OneSituation.size();i++)
		{
			for(int j=0; j< patchNum;j++)
			{
				//rectangle(showImg,vvpatches[j][OneSituation[i][j]].r,showColors[j]);
				T.at<float>(j,0) = vvpatches[j][OneSituation[i][j]].r.x;
				T.at<float>(j,1) = vvpatches[j][OneSituation[i][j]].r.y;
			}
			//cout<<T<<endl;
			Mat delta = T - W*T;
			double ret = absSum(delta);
			if (ret < MAX)
			{
				MAX=ret;
				answerIndex =i;

			}
			//cout<<absSum(delta)<<endl;
			//showVector(OneSituation[i]);//.size()<<endl;
		}
		for(int j=0; j< patchNum;j++)
		{
			//rectangle(showImg,vvpatches[j][OneSituation[answerIndex][j]].r,RED);
			circle(showImg,
				   Point( 
				          vvpatches[j][OneSituation[answerIndex][j]].r.x, vvpatches[j][OneSituation[answerIndex][j]].r.y
						 ) ,
						 4,
						 showColors[j]
			);
				    
		}*/

Rect  PF_Tracker::calcBoundingRect() //找所有patch的最小外接矩形
{
	Rect result(10000,10000,0,0);
	for(size_t  i=0;i<vpatches.size();i++)
	{
		if(vpatches[i].r.x<result.x)
			result.x=vpatches[i].r.x;
		if(vpatches[i].r.y<result.y)
			result.y=vpatches[i].r.y;
		if(vpatches[i].r.x+vpatches[i].r.width>result.width)
			result.width=vpatches[i].r.x+vpatches[i].r.width;
		if(vpatches[i].r.y+vpatches[i].r.height>result.height)
			result.height=vpatches[i].r.y+vpatches[i].r.height;
	}
	result.width=result.width-result.x;
	result.height=result.height-result.y;
	return result;
}

void  PF_Tracker::calcBoundingRect(Patch & ph) //找所有patch的最小外接矩形
{
	Rect result(10000,10000,0,0);

	for(size_t  i=0;i<ph.particles.size();i++)
	{
		if(ph.particles[i].x<result.x)
			result.x=ph.particles[i].x;
		if(ph.particles[i].y<result.y)
			result.y=ph.particles[i].y;
		if(ph.particles[i].x > result.width)
			result.width=ph.particles[i].x;
		if(ph.particles[i].y > result.height)
			result.height=ph.particles[i].y;
	}
	result.width=result.width-result.x;
	result.height=result.height-result.y;
	ph.boundRect = result;
	//return result;
}

void PF_Tracker::updatePatches(int deleteIndex)
{
	/*vector <Patch> initPatches;
	Rect region=targetRegion;
	
	Rect r(region.x+1, region.y+1, patchSize, patchSize);

	
	while (r.y < region.y+region.height - patchSize)
	{

		while (r.x < region.x+region.width - patchSize)
		{
			Patch tmp;
			tmp.r = r;
			Rect rt=r;
			rt.x -= searchArea.x;
			rt.y -= searchArea.y;
			Scalar s = mean(pToObject1(rt));
			tmp.confidence = s[0];
			initPatches.push_back(tmp);
			r.x += patchSize;
		}
		r.y += patchSize;
		r.x = region.x+1;
	}
	sort(initPatches.begin(), initPatches.end(), patchCmp);*/
}



