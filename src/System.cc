/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

//主进程的实现文件

//包含了一些自建库
#include "System.h"
#include "Converter.h"		// TODO 目前还不是很明白这个是做什么的
//包含共有库
#include <thread>					//多线程
#include <pangolin/pangolin.h>		//可视化界面
#include <iomanip>					//主要是对cin,cout之类的一些操纵运算子
#include <unistd.h>
namespace ORB_SLAM2{
//系统的构造函数，将会启动其他的线程
System::System(const string &strVocFile,					//词典文件路径
			   const string &strSettingsFile,				//配置文件路径
			   const eSensor sensor,						//传感器类型
               const bool bUseViewer):						//是否使用可视化界面
					 mSensor(sensor), 							//初始化传感器类型
					 mpViewer(static_cast<Viewer*>(NULL)),		//空。。。对象指针？  TODO 
					 mbReset(false),							//无复位标志
					 mbActivateLocalizationMode(false),			//没有这个模式转换标志
        			 mbDeactivateLocalizationMode(false)		//没有这个模式转换标志
{

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(),cv::FileStorage::READ);		//只读
    //如果打开失败，就输出调试信息
    if(!fsSettings.isOpened()){
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }

    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    //建立一个新的ORB字典
    mpVocabulary = new ORBVocabulary();
    //获取字典加载状态
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    //如果加载失败，就输出调试信息
    if(!bVocLoad){
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    else{
        cout << "Vocabulary loaded!" << endl << endl;
    }

    //Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary,strSettingsFile);
    int nMaxObjectID=fsSettings["Track.MaxObjectID"];
    float fCullKFRedundantMPRate = fsSettings["LocalMap.CullKFRedundantMPRate"];
    //Create the Map
    mpMap = new Map(nMaxObjectID);

    //Create Drawers. These are used by the Viewer
    //这里的帧绘制器和地图绘制器将会被可视化的Viewer所使用
    mpFrameDrawer = new FrameDrawer(mpMap, strSettingsFile);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    //在本主进程中初始化追踪线程
    //GetInitializationMatrixRTAndMPs the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this,						//现在还不是很明白为什么这里还需要一个this指针  TODO  
    						 mpVocabulary,				//字典
    						 mpFrameDrawer, 			//帧绘制器
    						 mpMapDrawer,				//地图绘制器
                             mpMap, 					//地图
                             mpKeyFrameDatabase, 		//关键帧地图
                             strSettingsFile, 			//设置文件路径
                             mSensor);					//传感器类型iomanip

    //初始化局部建图线程并运行
    //GetInitializationMatrixRTAndMPs the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, 				//指定使iomanip
    								 mSensor==MONOCULAR,
                                     fCullKFRedundantMPRate);
    //运行这个局部建图线程
    mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,	//这个线程会调用的函数
    							 mpLocalMapper);				//这个调用函数的参数

    //GetInitializationMatrixRTAndMPs the Loop Closing thread and launchiomanip
    mpLoopCloser = new LoopClosing(mpMap, 						//地图
    							   mpKeyFrameDatabase, 			//关键帧数据库
    							   mpVocabulary, 				//ORB字典
                                   nMaxObjectID,
                                   strSettingsFile,
    							   mSensor!=MONOCULAR);			//当前的传感器是否是单目
    //创建回环检测线程
    mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run,	//线程的主函数
    							mpLoopCloser);					//该函数的参数

    //GetInitializationMatrixRTAndMPs the Viewer thread and launch
    if(bUseViewer){
    	//如果指定了，程序的运行过程中需要运行可视化部分
    	//新建viewer
        mpViewer = new Viewer(this, 			//又是这个
        					  mpFrameDrawer,	//帧绘制器
        					  mpMapDrawer,		//地图绘制器
        					  mpTracker,		//追踪器
        					  strSettingsFile);	//配置文件的访问路径
        //新建viewer线程
        mptViewer = new thread(&Viewer::Run, mpViewer);
        //给运动追踪器设置其查看器
        mpTracker->SetViewer(mpViewer);
    }

    //Set pointers between threads
    //设置进程间的指针
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

//同理，输入为单目图像时的追踪器接口
cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp, long unsigned int FrameID, vector<cv::KeyPoint> vTarsInFrame, cv::Mat TgpsFrame)
{    if(mSensor!=MONOCULAR){
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
        exit(-1);
    }

    // Check mode change
    {
        // 独占锁，主要是为了mbActivateLocalizationMode和mbDeactivateLocalizationMode不会发生混乱
        unique_lock<mutex> lock(mMutexMode);
        // mbActivateLocalizationMode为true会关闭局部地图线程
        if(mbActivateLocalizationMode){
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped()){
                usleep(1000);
            }
            // 局部地图关闭以后，只进行追踪的线程，只计算相机的位姿，没有对局部地图进行更新
            // 设置mbOnlyTracking为真
            mpTracker->InformOnlyTracking(true);
            // 关闭线程可以使得别的线程得到更多的资源
            mbActivateLocalizationMode = false;
        }
        // 如果mbDeactivateLocalizationMode是true，局部地图线程就被释放, 关键帧从局部地图中删除.
        if(mbDeactivateLocalizationMode){
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->ReleaseNewKFinList();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset){
            mpTracker->Reset();
            mbReset = false;
        }
    }
    cv::Mat Tcw;
    //获取相机位姿的估计结果
    auto TimeSystemInit = chrono::system_clock::now();
    if(mpTracker->CreatORBFrameOrOpticalTrack(im, timestamp, FrameID, vTarsInFrame, TgpsFrame, Tcw)){//false LK optical flow; true ORB_frame
//        Step 3 ：跟踪
        Tcw = mpTracker->InitialOrDoORBTrack();
        //返回当前帧的位姿
        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    }
    return Tcw;
}

//激活定位模式
void System::ActivateLocalizationMode()
{
	//上锁
    unique_lock<mutex> lock(mMutexMode);
    //设置标志
    mbActivateLocalizationMode = true;
}

//取消定位模式
void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}



//准备执行复位
void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

//退出
void System::Shutdown()
{
	//对局部建图线程和回环检测线程发送终止请求
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    //如果使用了可视化窗口查看器
    if(mpViewer)
    {
    	//向查看器发送终止请求
        mpViewer->RequestFinish();
        //等到，知道真正地停止
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || 
    	  !mpLoopCloser->isFinished()  || 
    	   mpLoopCloser->isRunningGBA())			
    {
        usleep(5000);
    }

    if(mpViewer)
    	//如果使用了可视化的窗口查看器执行这个
    	// TODO 但是不明白这个是做什么的。如果我注释掉了呢？
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

//按照TUM格式保存相机运行轨迹并保存到指定的文件中
void System::SaveKeyFrameAndMapPointInGps(const string &filename, bool bSaveKeyFramesGps, bool bSaveObjectsGps)
{
    cout << endl << "Saving KeyFrame And MapPoint In Gps to " << filename << " ..." << endl;
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    if(bSaveKeyFramesGps){
        vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
        for(vector<KeyFrame *>::iterator vKFit=vpKFs.begin(), lend=vpKFs.end(); vKFit != lend; vKFit++){
            if(!(*vKFit)){
                continue;
            }
            if((*vKFit)->isBad()){
                continue;
            }

            cv::Mat twc = (*vKFit)->mTgpsFrame;
            f <<setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << endl;
        }
        f<<"-1 -1 -1"<<endl;
    }
    if(bSaveObjectsGps){
        vector<MapPoint*> vObjectsInGlobalMap;
        mpMap->GetAllObjectsInMap(vObjectsInGlobalMap);
        sort(vObjectsInGlobalMap.begin(), vObjectsInGlobalMap.end(), MapPoint::lId);
        for(vector<MapPoint*>::iterator vMPit=vObjectsInGlobalMap.begin(), vend=vObjectsInGlobalMap.end(); vMPit != vend; vMPit++) {
            MapPoint *pMPObj = *vMPit;
            if (!pMPObj) {
                continue;
            }
            if (pMPObj->GetbBad()) {
                continue;
            }
            cv::Mat twc = pMPObj->GetGpsPos();
            f <<setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << endl;
        }
        f<<"-1 -1 -1"<<endl;
    }
    f.close();
    cout << "File saved!" << endl;
}



//获取追踪器状态
int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

//获取追踪到的地图点（其实实际上得到的是一个指针）
vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

//获取追踪到的关键帧的点
vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

} //namespace ORB_SLAM
