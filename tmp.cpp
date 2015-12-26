vector < Rect> vRegionSearchRect;

		Rect regionSearchRect = targetRegion;
		regionSearchRect.x = targetRegion.x -patchSize;
		regionSearchRect.y = targetRegion.y -patchSize;
		for (; regionSearchRect.y < targetRegion.y+patchSize
			 ; regionSearchRect.y += 4)  //.y < vpatches[i].r.y + vpatches[i].r.height; patchSearchRect.y += patchSize/2)
		{
			for (; regionSearchRect.x < targetRegion.x+patchSize
				    ; regionSearchRect.x += 4)
			  {
				  //circle(showImg, Point( regionSearchRect.x, regionSearchRect.y) ,2 ,RED);
				  //rectangle(showImg, regionSearchRect,showColors[rand()%4]);
				 // imshow("case", showImg(regionSearchRect));
				  //waitKey();
				  vRegionSearchRect.push_back(regionSearchRect);
			  }
			  regionSearchRect.x = targetRegion.x -patchSize;
		}
		/*
		Rect regionSearchRect = targetRegion;
		regionSearchRect.x = searchArea.x + 1 + patchSize;
		regionSearchRect.y = searchArea.y + 1 + patchSize;
		for (; (regionSearchRect&searchArea) ==  regionSearchRect; regionSearchRect.y += patchSize/2)  //.y < vpatches[i].r.y + vpatches[i].r.height; patchSearchRect.y += patchSize/2)
		{
		      for (; (regionSearchRect&searchArea) ==  regionSearchRect; regionSearchRect.x += patchSize/2)
			  {
				  //circle(showImg, Point( regionSearchRect.x, regionSearchRect.y) ,2 ,RED);
				  //rectangle(showImg, regionSearchRect,showColors[rand()%4]);
				 // imshow("case", showImg(regionSearchRect));
				  //waitKey();
				  vRegionSearchRect.push_back(regionSearchRect);
			  }
			  regionSearchRect.x = searchArea.x + 1 + patchSize;
		}*/
		cout<< vRegionSearchRect.size() <<endl;

	
	

		vector <vector < Patch> >  vvpatches(patchNum, vector < Patch>() );
		vector<int> sizes;
		for (int i = 0; i < patchNum; i++)
		{
			Rect patchSearchRect = vpatches[i].r;
			patchSearchRect.x -=  patchSearchRect.width ;
		    patchSearchRect.y -=  patchSearchRect.height;
			for (; patchSearchRect.y < vpatches[i].r.y + vpatches[i].r.height; patchSearchRect.y += patchSize/3)
		    {
				for (; patchSearchRect.x < vpatches[i].r.x + vpatches[i].r.width; patchSearchRect.x += patchSize/3)
			    {
					vector <int> tempHis;
				    compute_histogram (patchSearchRect ,tempHis);
				    float dis = distance(tempHis,vpatches[i].histogram);
				    if (dis < 0.1)
					{
						vvpatches[i].push_back(Patch(patchSearchRect,dis));
						//circle(showImg,Point(patchSearchRect.x, patchSearchRect.y),2,RED);
					}
				}
				patchSearchRect.x = vpatches[i].r.x-patchSearchRect.width;
			}
			//sort(vvpatches[i].begin(), vvpatches[i].end(), patchCmp1);
			/*if (vvpatches[i].size()>10)
			{
				vvpatches[i].resize(10);
			}*/
			if (vvpatches[i].size() ==0 )
			{
				vvpatches[i].push_back(vpatches[i]);
				//cout<<"vvpatches[].size()"<<i<<","<<vvpatches[i].size()<<endl;
			}
			cout<<"vvpatches[].size()"<<i<<","<<vvpatches[i].size()<<endl;
			
			sizes.push_back(vvpatches[i].size());
		}
		float MAX = 10000;
		int resultLocationIndex = -1;
		vector<Patch> vpatchResult;

		for (int i = 0; i < vRegionSearchRect.size(); i++)
		{
			vector <vector < Patch> >  vvpatchesINaPaticle(patchNum, vector < Patch>() );
			int j = 0;
			for( ; j < patchNum; j++)
			{
				int k=0;
				while ( k <vvpatches[j].size())
				{
					if ( (vvpatches[j][k].r&vRegionSearchRect[i]) == vvpatches[j][k].r)
				    {
					    vvpatchesINaPaticle[j].push_back(vvpatches[j][k]);
				    }
					k++;
				}
				if(vvpatchesINaPaticle[j].size() ==0)
				{
					break;
				}
				else
				{
					sort(vvpatchesINaPaticle[j].begin(),vvpatchesINaPaticle[j].end(),patchCmp1);
					if (vvpatchesINaPaticle[j].size()>8) 
					{
							vvpatchesINaPaticle[j].resize(8);
					}
				}
			}
			if ( j == patchNum)
			{
				//circle(showImg, Point( vRegionSearchRect[i].x, vRegionSearchRect[i].y) ,2 ,RED);
				//rectangle(showImg, vRegionSearchRect[i],showColors[0]);
				vector <int> locationindexes;
			    float ret = solveIP(vRegionSearchRect[i], vvpatchesINaPaticle,locationindexes);
				if (ret< MAX)
				{
					MAX = ret;
					resultLocationIndex = i;
					vpatchResult.clear();
					for (int k =0; k<patchNum; k++)
					{
						vpatchResult.push_back(vvpatchesINaPaticle[k][locationindexes[k]]);
					}
				}
			}
		}
		if (resultLocationIndex >= 0)
		{
			cout<<"max : "<<MAX<<endl;
			//targetRegion = vRegionSearchRect[resultLocationIndex];
		}
		

		Point newFocus;
		newFocus.x = 0;
		newFocus.y = 0;

		for (int k =0; k<patchNum; k++)
		{
			vpatches[k].r = vpatchResult[k].r;
			//vpatches[k].histogram = vpatchResult[k].histogram;
			newFocus.x += vpatches[k].r.x;
			newFocus.y += vpatches[k].r.y;
			rectangle(showImg, vpatchResult[k].r, GREEN, 1);
		}
		newFocus.x /= patchNum;
		newFocus.y /= patchNum;
		targetRegion.x = targetRegion.x+newFocus.x- Focus.x;
		targetRegion.y = targetRegion.y+newFocus.y- Focus.y;
		Focus=newFocus;
		rectangle(showImg, targetRegion, GREEN, 1);
	}
	
	
	
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





	/*
	pToObject.create(searchArea.height, searchArea.width, CV_8UC1);

	for (int i = 0; i < searchArea.height; i++)
	{
			for (int j = 0; j < searchArea.width; j++)
			{
				int hIndex = splithsv[0].at<uchar>(i, j) * NH/ 181;
			    int sIndex = splithsv[1].at<uchar>(i, j) * NS/ 256;
				int k = 10 * hIndex + sIndex;
				if (hisSearchArea[k] == 0)
				{
				    pToObject.at<uchar>(i,j)= 0;
				}
				else
				{
					 pToObject.at<uchar>(i,j)  = hisTarget[k]*255/(hisSearchArea[k]);//*3/2 ;
				}
			}
	}
	//imshow("mask", mask);
	imshow("pToObject", pToObject);*/

	//vpatches.clear();
	//makePatches();
	calcW();
	/*
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
	//trackPatches();*/
	preFrame = frame;