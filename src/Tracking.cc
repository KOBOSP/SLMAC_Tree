/**
 * @file Tracking.cc
 * @author guoqing (1337841346@qq.com)
 * @brief 追踪线程
 * @version 0.1
 * @date 2019-02-21
 * 
 * @copyright Copyright (c) 2019
 * 
 */

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


#include "Tracking.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"

#include "Optimizer.h"
#include "PnPsolver.h"

#include <iostream>
#include <cmath>
#include <mutex>
#include <time.h>


using namespace std;

// 程序中变量名的第一个字母如果为"m"则表示为类中的成员变量，member
// 第一个、第二个字母:
// "p"表示指针数据类型
// "n"表示int类型
// "b"表示bool类型
// "s"表示set类型
// "v"表示vector数据类型
// 'l'表示list数据类型
// "KF"表示KeyFrame数据类型   

namespace ORB_SLAM2
{

///构造函数
Tracking::Tracking(
    System *pSys,                       //系统实例
    ORBVocabulary* pVoc,                //BOW字典
    FrameDrawer *pFrameDrawer,          //帧绘制器
    MapDrawer *pMapDrawer,              //地图点绘制器
    Map *pMap,                          //地图句柄
    KeyFrameDatabase* pKFDB,            //关键帧产生的词袋数据库
    const string &strSettingPath,       //配置文件路径
    const int sensor):                  //传感器类型
        mState(NO_IMAGES_YET),                              //当前系统还没有准备好
        mSensor(sensor),
        mbBadVO(false),                                        //当处于纯跟踪模式的时候，这个变量表示了当前跟踪状态的好坏
        mpORBVocabulary(pVoc),
        mpKeyFrameDB(pKFDB),
        mpInitializer(static_cast<Initializer*>(NULL)),     //暂时给地图初始化器设置为空指针
        mpSystem(pSys),
        mpViewer(static_cast<Viewer*>(NULL)),         //注意可视化的查看器是可选的，因为ORB-SLAM2最后是被编译成为一个库，所以对方人拿过来用的时候也应该有权力说我不要可视化界面（何况可视化界面也要占用不少的CPU资源）
        mpFrameDrawer(pFrameDrawer),
        mpMapDrawer(pMapDrawer),
        mpMap(pMap),
        mnLastRelocFrameId(0)                               //恢复为0,没有进行这个过程的时候的默认值
{
    // Load camera parameters from settings file
    // Step 1 从配置文件中加载相机参数
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    int frame_width=fSettings["Camera.width"];
    int frame_height=fSettings["Camera.height"];
    int image_width=fSettings["Image.width"];
    int image_height=fSettings["Image.height"];
    //     |fx  0   cx|
    // K = |0   fy  cy|
    //     |0   0   1 |
    //构造相机内参矩阵
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    // 图像矫正系数
    // [k1 k2 p1 p2 k3]
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    //有些相机的畸变系数中会没有k3项
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    // Max/Min Frames to insert keyframes and to check relocalisation
    mnfpsByCfgFile = fSettings["Camera.fps"];
    mSim3VoGps=cv::Mat();
    mnSim3Inliers=0;

    //输出
    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- mnfpsByCfgFile: " << mnfpsByCfgFile << endl;
    cout << "- frame_width: " << frame_width << endl;
    cout << "- frame_height: " << frame_height << endl;
    cout << "- image_width: " << image_width << endl;
    cout << "- image_height: " << image_height << endl;
    // 1:RGB 0:BGR
    int ntmp = fSettings["Camera.RGB"];
    mbRGB = ntmp;
    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    // Step 2 加载ORB特征点有关的参数,并新建特征点提取器

    // 每一帧提取的特征点数 1000
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    // 图像建立金字塔时的变化尺度 1.2
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    // 尺度金字塔的层数 8
    int nLevels = fSettings["ORBextractor.nLevels"];
    // 提取fast特征点的默认阈值 20
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];
    mnGoodMMOrRKFVOThreshold = fSettings["Track.GoodMMOrRKFVOThreshold"];
    mnLossMMOrRKFVOThreshold = fSettings["Track.LossMMOrRKFVOThreshold"];
    mnGoodLMVOThreshold = fSettings["Track.GoodLMVOThreshold"];
    mnLossLMVOThreshold = fSettings["Track.LossLMVOThreshold"];
    mnMotionProjectRadius = fSettings["Track.MotionProjectRadius"];
    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;
    cout << "- Track.GoodMMOrRKFVOThreshol: " << mnGoodMMOrRKFVOThreshold << endl;
    cout << "- Track.LossMMOrRKFVOThreshold: " << mnLossMMOrRKFVOThreshold << endl;
    cout << "- Track.GoodLMVOThreshold: " << mnGoodLMVOThreshold << endl;
    cout << "- Track.LossLMVOThreshold: " << mnLossLMVOThreshold << endl;


    // tracking过程都会用到mpORBextractorLeft作为特征点提取器
    mpORBextractorLeft = new ORBextractor(
            nFeatures,      //参数的含义还是看上面的注释吧
            fScaleFactor,
            nLevels,
            fIniThFAST,
            fMinThFAST);

    // 在单目初始化的时候，会用mpIniORBextractor来作为特征点提取器
    mpORBextractorInit = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
}



//设置局部建图器
void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

//设置回环检测器
void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

//设置可视化查看器
void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

/**
 * @brief 
 * 输入左目RGB或RGBA图像，输出世界坐标系到该帧相机坐标系的变换矩阵
 * 
 * @param[in] Img 单目图像
 * @param[in] timestamp 时间戳
 * @return bool false: optical, true: ORB Frame
 * 
 * Step 1 ：将彩色图像转为灰度图像
 * Step 2 ：构造Frame
 * Step 3 ：跟踪
 */
bool Tracking::CreatORBFrameOrOpticalTrack(const cv::Mat &Img, const double &timestamp, long unsigned int FrameID, vector<cv::KeyPoint> &vTarsInFrame, cv::Mat &TgpsFrame, cv::Mat &FAITcw){
    mImgGray = Img;
    // Step 1 ：将彩色图像转为灰度图像
    //若图片是3、4通道的，还需要转化成灰度图
    if(mImgGray.channels() == 3){
        if(mbRGB)
            cvtColor(mImgGray, mImgGray, CV_RGB2GRAY);
        else
            cvtColor(mImgGray, mImgGray, CV_BGR2GRAY);
    }
    else if(mImgGray.channels() == 4){
        if(mbRGB)
            cvtColor(mImgGray, mImgGray, CV_RGBA2GRAY);
        else
            cvtColor(mImgGray, mImgGray, CV_BGRA2GRAY);
    }
    // Step 2 ：构造Frame
    //判断该帧是不是初始化
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET) { //没有成功初始化的前一个状态就是NO_IMAGES_YET
        mCurrentFrame = Frame(
                FrameID,
                mImgGray,
                timestamp,
                mpORBextractorInit,      //初始化ORB特征点提取器会提取2倍的指定特征点数目
                mpORBVocabulary,
                mK,
                mDistCoef,
                mThDepth,
                vTarsInFrame,
                TgpsFrame);

    }
    else{// not the initial stage
        mCurrentFrame = Frame(
                FrameID,
                mImgGray,
                timestamp,
                mpORBextractorLeft,     //正常运行的时的ORB特征点提取器，提取指定数目特征点
                mpORBVocabulary,
                mK,
                mDistCoef,
                mThDepth,
                vTarsInFrame,
                TgpsFrame);
    }
    return true;
}

/*
 * @brief Main tracking function. It is independent of the input sensor.
 *
 * track包含两部分：估计运动、跟踪局部地图
 * 
 * Step 1：初始化
 * Step 2：跟踪
 * Step 3：记录位姿信息，用于轨迹复现
 */
cv::Mat Tracking::InitialOrDoORBTrack(){
    // track包含两部分：估计运动、跟踪局部地图
    // mState为tracking的状态，包括 SYSTME_NOT_READY, NO_IMAGE_YET, NOT_INITIALIZED, OK, LOST
    // 如果图像复位过、或者第一次运行，则为NO_IMAGE_YET状态
    if(mState==NO_IMAGES_YET){
        mState = NOT_INITIALIZED;
    }
    // mLastProcessedState 存储了Tracking最新的状态，用于FrameDrawer中的绘制
    mLastProcessedState=mState;
    // Get Map Mutex -> Map cannot be changed
    // 地图更新时加锁。保证地图不会发生变化
    // 疑问:这样子会不会影响地图的实时更新?
    // 回答：主要耗时在构造帧中特征点的提取和匹配部分,在那个时候地图是没有被上锁的,有足够的时间更新地图
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    // Step 1：地图初始化
    if(mState==NOT_INITIALIZED){
        if(mSensor==System::MONOCULAR){
            //单目初始化
            MonocularInitialization();
        }

        //更新帧绘制器中存储的最新状态
        mpFrameDrawer->UpdateImgKPMPState(this);

        //这个状态量在上面的初始化函数中被更新
        if(mState!=OK){
            return cv::Mat();
        }
    }
    else {//trace mode and Relocalization mode
        // System is initialized. InitialOrDoORBTrack Frame.
        // bOK为临时变量，用于表示每个函数是否执行成功
        bool bOK;
        // 将最新的关键帧作为当前帧的参考关键帧
        vector<KeyFrame*> vKFNearest;
        vKFNearest.reserve(10);
        mpMap->GetNearestKeyFramesByGps(10, vKFNearest, mCurrentFrame.mtFrameGps);
        mCurrentFrame.mpReferenceKF = vKFNearest[0];

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        // mbOnlyTracking等于false表示正常SLAM模式（定位+地图更新），mbOnlyTracking等于true表示仅定位模式
        // tracking 类构造时默认为false。在viewer中有个开关ActivateLocalizationMode，可以控制是否开启mbOnlyTracking
        // Local Mapping is activated. This is the normal behaviour, unless
        // you explicitly activate the "only tracking" mode.

        // Step 2：跟踪进入正常SLAM模式，有地图更新
        // 是否正常跟踪
        if (mState == OK) {
            // Local Mapping might have changed some MapPoints tracked in last frame
            // Step 2.1 检查并更新上一帧被替换的MapPoints
            // 局部建图线程则可能会对原有的地图点进行替换.在这里进行检查
            CheckReplacedInLastFrame();

            // Step 2.2 运动模型是空的或刚完成重定位，跟踪参考关键帧；否则恒速模型跟踪
            // 第一个条件,如果运动模型为空,说明是刚初始化开始，或者已经跟丢了
            // 第二个条件,如果当前帧紧紧地跟着在重定位的帧的后面，我们将重定位帧来恢复位姿
            // mnLastRelocFrameId 上一次重定位的那一帧
            if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + mnfpsByCfgFile) {
                // 用最近的关键帧来跟踪当前的普通帧
                // 通过BoW的方式在参考帧中找当前帧特征点的匹配点
                // 优化每个特征点都对应3D点重投影误差即可得到位姿
                bOK = TrackWithReferenceKeyFrame();
            } else {
                // 用最近的普通帧来跟踪当前的普通帧
                // 根据恒速模型设定当前帧的初始位姿
                // 通过投影的方式在参考帧中找当前帧特征点的匹配点
                // 优化每个特征点所对应3D点的投影误差即可得到位姿
                bOK = TrackWithMotionModel();
                if (!bOK) {
                    //根据恒速模型失败了，只能根据参考关键帧来跟踪
                    bOK = TrackWithReferenceKeyFrame();
                }
            }
        }
        else {//state != Init or OK
            bOK = false;
            if(!mSim3VoGps.empty()) {
                bOK = TrackWithGpsTranslation(vKFNearest);
                if(bOK)
                    cout<<"TrackWithGpsTranslation bOK "<<endl;
            }
            // 如果跟踪状态不成功,那么就只能重定位了
            // BOW搜索，EPnP求解位姿
            if (!bOK) {
                bOK = Relocalization();
                if(bOK)
                    cout<<"Relocalization bOK "<<endl;
            }
        }


        // If we have an initial estimation of the camera pose and matching. InitialOrDoORBTrack the local map.
        // Step 3：在跟踪得到当前帧初始姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
        // 前面只是跟踪一帧得到初始位姿，这里搜索局部关键帧、局部地图点，和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
        if(bOK){
            bOK = TrackWithLocalMap();
        }

        //根据上面的操作来判断是否追踪成功
        if(bOK)
            mState = OK;
        else
            mState = LOST;

        // Step 4：更新显示线程中的图像、特征点、地图点等信息
        ;
        if(mpFrameDrawer->UpdateImgKPMPState(this)<mnGoodMMOrRKFVOThreshold){
            mbBadVO = true;
        }



        // If tracking were good, check if we insert a keyframe
        //只有在成功追踪时才考虑生成关键帧的问题
        if(bOK){
            // UpdateImgKPMPState motion model
            // Step 5：跟踪成功，更新恒速运动模型
            if(!mLastFrame.mTcw.empty()){
                // 更新恒速运动模型 TrackWithMotionModel 中的mVelocity
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                // mVelocity = Tcl = Tcw * Twl,表示上一帧到当前帧的变换， 其中 Twl = LastTwc
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else{
                //否则速度为空
                mVelocity = cv::Mat();
            }

            //更新显示中的位姿
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
            // Clean VO matches
            // Step 6：清除观测不到的地图点   
            for(int i=0; i<mCurrentFrame.mnKeyPointNum; i++){
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP){
                    if(pMP->GetObservations() < 1){
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
                }
            }

            // Check if we need to insert a new keyframe
            // Step 8：检测并插入关键帧，对于双目或RGB-D会产生新的地图点
            if(NeedNewKeyFrame()){
                CreateNewKeyFrame();
            }

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            // 作者这里说允许在BA中被Huber核函数判断为外点的传入新的关键帧中，让后续的BA来审判他们是不是真正的外点
            // 但是估计下一帧位姿的时候我们不想用这些外点，所以删掉

            //  Step 9 删除那些在bundle adjustment中检测为outlier的地图点
            for(int i=0; i<mCurrentFrame.mnKeyPointNum; i++){
                // 这里第一个条件还要执行判断是因为, 前面的操作中可能删除了其中的地图点
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        // Step 10 如果初始化后不久就跟踪失败，并且relocation也没有搞定，只能重新Reset
        if(mState==LOST){
            //如果地图中的关键帧信息过少的话,直接重新进行初始化了
            if(mpMap->GetKeyFramesNumInMap() <= 4){
                cout << "InitialOrDoORBTrack lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return cv::Mat();
            }
        }

//        //确保已经设置了参考关键帧
//        if(!mCurrentFrame.mpReferenceKF)
//            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // 保存上一帧的数据,当前帧变上一帧
        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    // Step 11：记录位姿信息，用于最后保存所有的轨迹
    if(!mCurrentFrame.mTcw.empty()){
        // 计算相对姿态Tcr = Tcw * Twr, Twr = Trw^-1
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        //保存各种状态
        mlRelativeFramePoses.emplace_back(Tcr);
        mlpReferences.emplace_back(mpReferenceKF);
        mlFrameTimes.emplace_back(mCurrentFrame.mTimeStamp);
        mlbLost.emplace_back(mState==LOST);
    }
    else{
        // This can happen if tracking is lost
        // 如果跟踪失败，则相对位姿使用上一次值
        mlRelativeFramePoses.emplace_back(mlRelativeFramePoses.back());
        mlpReferences.emplace_back(mlpReferences.back());
        mlFrameTimes.emplace_back(mlFrameTimes.back());
        mlbLost.emplace_back(mState==LOST);
    }
    return mCurrentFrame.mTcw.clone();
}// Tracking




/*
 * @brief 单目的地图初始化
 *
 * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 * 得到初始两帧的匹配、相对运动、初始MapPoints
 * 
 * Step 1：（未创建）得到用于初始化的第一帧，初始化需要两帧
 * Step 2：（已创建）如果当前帧特征点数大于100，则得到用于单目初始化的第二帧
 * Step 3：在mInitialFrame与mCurrentFrame中找匹配的特征点对
 * Step 4：如果初始化的两帧之间的匹配点太少，重新初始化
 * Step 5：通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
 * Step 6：删除那些无法进行三角化的匹配点
 * Step 7：将三角化得到的3D点包装成MapPoints
 */
void Tracking::MonocularInitialization(){
    // Step 1 如果单目初始器还没有被创建，则创建。后面如果重新初始化时会清掉这个
    if(!mpInitializer){
        // Set Reference Frame
        // 单目初始帧的特征点数必须大于100
        if(mCurrentFrame.mvKeys.size()>100){
            // 初始化需要两帧，分别是mInitialFrame，mCurrentFrame
            mInitialFrame = Frame(mCurrentFrame);
            // 用当前帧更新上一帧
            mLastFrame = Frame(mCurrentFrame);
            // mvbPrevMatched  记录"上一帧"所有特征点
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            // 由当前帧构造初始器 sigma:1.0 iterations:200
            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            // 初始化为-1 表示没有任何匹配。这里面存储的是匹配的点的id
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
        }
    }
    else{
        // Try to initialize
        // Step 2 如果当前帧特征点数太少（不超过100），则重新构造初始器
        // NOTICE 只有连续两帧的特征点个数都大于100时，才能继续进行初始化过程
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        // Step 3 在mInitialFrame与mCurrentFrame中找匹配的特征点对
        ORBmatcher matcher(
            0.9,        //最佳的和次佳特征点评分的比值阈值，这里是比较宽松的，跟踪时一般是0.7
            true);      //检查特征点的方向

        // 对 mInitialFrame,mCurrentFrame 进行特征点匹配
        // mvbPrevMatched为参考帧的特征点坐标，初始化存储的是mInitialFrame中特征点坐标，匹配后存储的是匹配好的当前帧的特征点坐标
        // mvIniMatches 保存参考帧F1中特征点是否匹配上，index保存是F1对应特征点索引，值保存的是匹配好的F2特征点索引
        int nmatches = matcher.SearchMatchKPInInitialization(
                mInitialFrame, mCurrentFrame,    //初始化时的参考帧和当前帧
                mvbPrevMatched,                 //在初始化参考帧中提取得到的特征点
                mvIniMatches,                   //保存匹配关系
                300);                    //搜索窗口大小

        // Check if there are enough correspondences
        // Step 4 验证匹配结果，如果初始化的两帧之间的匹配点太少，重新初始化
        if(nmatches<100){
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        // Step 5 通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
        if(mpInitializer->GetInitializationMatrixRTAndMPs(
                mCurrentFrame,      //当前帧
                mvIniMatches,       //当前帧和参考帧的特征点的匹配关系
                Rcw, tcw,           //初始化得到的相机的位姿
                mvIniP3D,           //进行三角化得到的空间点集合
                vbTriangulated))    //以及对应于mvIniMatches来讲,其中哪些点被三角化了
        {
            // Step 6 初始化成功后，删除那些无法进行三角化的匹配点
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++){
                if(mvIniMatches[i]>=0 && !vbTriangulated[i]){
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            // Step 7 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            mInitialFrame.SetTcwPose(Tcw);
            // 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到相机坐标系的变换矩阵
            Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetTcwPose(Tcw);

            // Step 8 创建初始化地图点MapPoints
            // Initialize函数会得到mvIniP3D，
            // mvIniP3D是cv::Point3f类型的一个容器，是个存放3D点的临时变量，
            // CreateInitialMapMonocular将3D点包装成MapPoint类型存入KeyFrame和Map中
            CreateInitialMapMonocular();
        }//当初始化成功的时候进行
    }//如果单目初始化器已经被创建
}

/**
 * @brief 单目相机成功初始化后用三角化得到的点生成MapPoints
 * 
 */
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames 认为单目初始化时候的参考帧和当前帧都是关键帧
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);  // 第一帧
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);  // 第二帧

    // Step 1 将初始关键帧,当前关键帧的描述子转为BoW
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    // Step 2 将关键帧插入到地图
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    // Step 3 用初始化得到的3D点来生成地图点MapPoints
    //  mvIniMatches[i] 表示初始化两帧特征点匹配关系。
    //  具体解释：i表示帧1中关键点的索引值，vMatches12[i]的值为帧2的关键点索引值,没有匹配关系的话，vMatches12[i]值为 -1
    for(size_t i=0; i<mvIniMatches.size();i++){
        // 没有匹配，跳过
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        // 用三角化点初始化为空间点的世界坐标
        cv::Mat worldPos(mvIniP3D[i]);

        // Step 3.1 用3D点构造MapPoint
        MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);

        // Step 3.2 为该MapPoint添加属性：
        // a.观测到该MapPoint的关键帧
        // b.该MapPoint的描述子
        // c.该MapPoint的平均观测方向和深度范围

        // 表示该KeyFrame的2D特征点和对应的3D地图点
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        // a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        // b.从众多观测到该MapPoint的特征点中挑选最有代表性的描述子
        pMP->ComputeDistinctiveDescriptors();
        // c.更新该MapPoint平均观测方向以及观测距离的范围
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        //mvIniMatches下标i表示在初始化参考帧中的特征点的序号
        //mvIniMatches[i]是初始化当前帧中的特征点的序号
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // UpdateImgKPMPState Connections
    // Step 3.3 更新关键帧间的连接关系
    // 在3D点和关键帧之间建立边，每个边有一个权重，边的权重是该关键帧与当前帧公共3D点的个数
    pKFini->UpdateConnectedKeyFrameAndWeights();
    pKFcur->UpdateConnectedKeyFrameAndWeights();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->GetMapPointsNumInMap() << " points" << endl;

    // Step 4 全局BA优化，同时优化所有位姿和三维点
    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    // Step 5 取场景的中值深度，用于尺度归一化 
    // 为什么是 pKFini 而不是 pKCur ? 答：都可以的，内部做了位姿变换了
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;
    
    //两个条件,一个是平均深度要大于0,另外一个是在当前帧中被观测到的地图点的数目应该大于100
    if(medianDepth<0 || pKFcur->GetNumMapPointsBigObs(1) < 100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Step 6 将两帧之间的变换归一化到平均深度1的尺度下
    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    // x/z y/z 将z归一化到1
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    // Step 7 把3D点的尺度也归一化到1
    // 为什么是pKFini? 是不是就算是使用 pKFcur 得到的结果也是相同的? 答：是的，因为是同样的三维点
    vector<MapPoint*> vpAllMapPoints = pKFini->GetAllMapPointVectorInKF();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++){
        if(vpAllMapPoints[iMP]){
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    //  Step 8 将关键帧插入局部地图，更新归一化后的位姿、局部地图点
    mpLocalMapper->InsertKeyFrameInQueue(pKFini);
    mpLocalMapper->InsertKeyFrameInQueue(pKFcur);

    mCurrentFrame.SetTcwPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.emplace_back(pKFcur);
    mvpLocalKeyFrames.emplace_back(pKFini);
    // 单目初始化之后，得到的初始地图中的所有点都是局部地图点
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    //也只能这样子设置了,毕竟是最近的关键帧，bug, first line is original second line is writed by wang
    //    mCurrentFrame.mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFini;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.emplace_back(pKFini);

    mState=OK;// 初始化成功，至此，初始化过程完成
}

/*
 * @brief 检查上一帧中的地图点是否需要被替换
 * 
 * Local Mapping线程可能会将关键帧中某些地图点进行替换，由于tracking中需要用到上一帧地图点，所以这里检查并更新上一帧中被替换的地图点
 * @see LocalMapping::FuseMapPointsAndObjectsByNeighbors()
 */
void Tracking::CheckReplacedInLastFrame(){
    for(int i =0; i<mLastFrame.mnKeyPointNum; i++){
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        //如果这个地图点存在
        if(pMP){
            // 获取其是否被替换,以及替换后的点
            // 这也是程序不直接删除这个地图点删除的原因
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep){
                //然后替换一下
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

/**
 * @brief 用参考关键帧的地图点来对当前普通帧进行跟踪
 * 
 * Step 1：将当前普通帧的描述子转化为BoW向量
 * Step 2：通过词袋BoW加速当前帧与参考帧之间的特征点匹配
 * Step 3: 将上一帧的位姿态作为当前帧位姿的初始值
 * Step 4: 通过优化3D-2D的重投影误差来获得位姿
 * Step 5：剔除优化后的匹配点中的外点
 * @return 如果匹配数超10，返回true
 * 
 */
bool Tracking::TrackWithReferenceKeyFrame()
{
    // Compute Bag of Words vector
    // Step 1：将当前帧的描述子转化为BoW向量
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    // Step 2：通过词袋BoW加速当前帧与参考帧之间的特征点匹配
    int nmatches = matcher.SearchFMatchPointByKFBoW(
            mpReferenceKF,          //参考关键帧
            mCurrentFrame,          //当前帧
            vpMapPointMatches);     //存储匹配关系
    // 匹配数目小于15，认为跟踪失败
    if(nmatches<mnLossMMOrRKFVOThreshold){
        return false;
    }

    // Step 3:将上一帧的位姿态作为当前帧位姿的初始值
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetTcwPose(mLastFrame.mTcw); // 用上一次的Tcw设置初值，在PoseOptimization可以收敛快一些

    // Step 4:通过优化3D-2D的重投影误差来获得位姿
    Optimizer::OptimizeFramePose(&mCurrentFrame);

    // Discard outliers
    // Step 5：剔除优化后的匹配点中的外点
    //之所以在优化之后才剔除外点，是因为在优化的过程中就有了对这些外点的标记
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.mnKeyPointNum; i++){
        if(mCurrentFrame.mvpMapPoints[i]){
            //如果对应到的某个特征点是外点
            if(mCurrentFrame.mvbOutlier[i]){
                //清除它在当前帧中存在过的痕迹
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->GetObservations() > 0)
                //匹配的内点计数++
                nmatchesMap++;
        }
    }
    // 跟踪成功的数目超过10才认为跟踪成功，否则跟踪失败
    if(nmatchesMap >= mnLossMMOrRKFVOThreshold){
        mnMotionOrGpsOrRefKFTrackOK = 3;
    }
    else{
        mnMotionOrGpsOrRefKFTrackOK = 0;
    }
    return mnMotionOrGpsOrRefKFTrackOK;
}

bool Tracking::TrackWithGpsTranslation(vector<KeyFrame*> &vKFNearest){
    // 最小距离 < 0.9*次小距离 匹配成功，检查旋转
    ORBmatcher matcher(0.7, false);
    cv::Mat tFrameGps,tFrameVo, Sim3VG, sRVG, tVG;
    cv::Mat TFrameVo = cv::Mat::eye(4,4,CV_32F);
    mCurrentFrame.mtFrameGps.copyTo(tFrameGps);
    mSim3VoGps.copyTo(Sim3VG);
    Sim3VG.rowRange(0, 3).colRange(0, 3).copyTo(sRVG);
    Sim3VG.rowRange(0, 3).col(3).copyTo(tVG);
    tFrameVo= sRVG * tFrameGps + tVG;
    tFrameVo.copyTo(TFrameVo.rowRange(0, 3).col(3));
    mLastFrame.mTcw.rowRange(0, 3).colRange(0, 3).copyTo(TFrameVo.rowRange(0, 3).colRange(0, 3));
    // Step 2：用rtk得到当前帧的初始位姿, Assuming that the rotation matrix does not change
    mCurrentFrame.SetTcwPose(TFrameVo);
    // 清空当前帧的地图点
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
    // 设置特征匹配过程中的搜索半径
    int th = mnMotionProjectRadius*2;

    vector<MapPoint*> vpMapPoints;
    vpMapPoints.reserve(400);
    // Step 2：遍历局部关键帧 mvpLocalKeyFrames
    for(vector<KeyFrame*>::const_iterator itKF=vKFNearest.begin(), itEndKF=vKFNearest.end(); itKF!=itEndKF; itKF++){
        KeyFrame* pKF = *itKF;
        if(!pKF){
            continue;
        }
        if(pKF->isBad()){
            continue;
        }
        const vector<MapPoint*> vpMPs = pKF->GetAllMapPointVectorInKF(false);
        // step 2：将局部关键帧的地图点添加到mvpLocalMapPoints
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++){
            MapPoint* pMP = *itMP;
            if(!pMP){
                continue;
            }
            if(!pMP->GetbBad()){
                if(pMP->mnFrameIdForGpsOrMotion == mCurrentFrame.mnId){
                    continue;
                }
                vpMapPoints.emplace_back(pMP);
                pMP->mnFrameIdForGpsOrMotion=mCurrentFrame.mnId;
            }
        }
    }
    // Step 3：用上一帧地图点进行投影匹配，如果匹配点不够，则扩大搜索半径再来一次
    int nmatches = matcher.SearchFMatchPointByProjectMapPoint(mCurrentFrame, vpMapPoints, th, false);
    // If few matches, uses a wider window search
    // 如果匹配点太少，则扩大搜索半径再来一次
    if (nmatches < mnGoodMMOrRKFVOThreshold) {
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        nmatches = matcher.SearchFMatchPointByProjectMapPoint(mCurrentFrame, vpMapPoints, 2 * th, false); // 2*th
    }
//    printf("nmatches<mnLossMMOrRKFVOThreshold:%d<%d\n",nmatches,mnLossMMOrRKFVOThreshold);
    // 如果不能够获得足够的匹配点, use constant velocity
    if(nmatches<mnLossMMOrRKFVOThreshold){
        return false;
    }

    // Optimize frame pose with all matches
    // Step 4：利用3D-2D投影关系，优化当前帧位姿
    Optimizer::OptimizeFramePose(&mCurrentFrame,2);

    // Discard outliers
    // Step 5：剔除地图点中外点
    int nmatchesMap = 0, KKK=0;
    for(int i =0; i<mCurrentFrame.mnKeyPointNum; i++){
        if(mCurrentFrame.mvpMapPoints[i]){
            if(mCurrentFrame.mvpMapPoints[i]->GetbBad()){
                KKK++;
                continue;
            }
            if(mCurrentFrame.mvbOutlier[i]){
                // 如果优化后判断某个地图点是外点，清除它的所有关系
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->GetObservations() > 0){
                nmatchesMap++;
            }
        }
    }

    // Step 6：匹配超过10个点就认为跟踪成功
    if(nmatchesMap >= mnLossMMOrRKFVOThreshold){
        mnMotionOrGpsOrRefKFTrackOK = 2;
    }
    else{
        mnMotionOrGpsOrRefKFTrackOK = 0;
    }
    return mnMotionOrGpsOrRefKFTrackOK;
}


/**
 * @brief 根据恒定速度模型用上一帧地图点来对当前帧进行跟踪
 * Step 1：更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
 * Step 2：根据上一帧特征点对应地图点进行投影匹配
 * Step 3：优化当前帧位姿
 * Step 4：剔除地图点中外点
 * @return 如果匹配数大于10，认为跟踪成功，返回true
 */
bool Tracking::TrackWithMotionModel()
{
    // 最小距离 < 0.9*次小距离 匹配成功，检查旋转
    ORBmatcher matcher(0.8,true);

    // UpdateImgKPMPState last frame pose according to its reference keyframe
    // Create "visual odometry" points
    // Step 1：更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
//    UpdateLastFrame();

    mCurrentFrame.SetTcwPose(mVelocity*mLastFrame.mTcw);
    // 清空当前帧的地图点
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
    // Project points seen in previous frame
    // 设置特征匹配过程中的搜索半径
    int th = mnMotionProjectRadius;


    vector<MapPoint*> vpMapPoints;
    vpMapPoints.reserve(400);

    for(int i=0; i<mLastFrame.mnKeyPointNum; i++) {
        MapPoint *pMP = mLastFrame.mvpMapPoints[i];
        if (pMP) {
            if (pMP->GetbBad()) {
                continue;
            }
            if (pMP->GetObjectId() > 0) {
                continue;
            }
            if (pMP->mnFrameIdForGpsOrMotion == mCurrentFrame.mnId) {
                continue;
            }
            vpMapPoints.emplace_back(pMP);
            pMP->mnFrameIdForGpsOrMotion = mCurrentFrame.mnId;
        }
    }
//    auto ExactStart = std::chrono::steady_clock::now();

    // Step 3：用上一帧地图点进行投影匹配，如果匹配点不够，则扩大搜索半径再来一次
    int nmatches = matcher.SearchFMatchPointByProjectMapPoint(mCurrentFrame, vpMapPoints, th, false);

//    auto ExactEnd = std::chrono::steady_clock::now();
//    std::chrono::duration<double> spent = ExactEnd - ExactStart;
//    std::cout <<mCurrentFrame.mvKeys.size() <<"," << vpMapPoints.size() << ","  <<nmatches <<","<<spent.count() << " sec \n";

    if (nmatches < mnGoodMMOrRKFVOThreshold) {
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        nmatches = matcher.SearchFMatchPointByProjectMapPoint(mCurrentFrame, vpMapPoints, 2*th, false);
    }
    if(nmatches<mnLossMMOrRKFVOThreshold){
        return false;
    }

    // Optimize frame pose with all matches
    // Step 4：利用3D-2D投影关系，优化当前帧位姿
    Optimizer::OptimizeFramePose(&mCurrentFrame);

    // Discard outliers
    // Step 5：剔除地图点中外点
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.mnKeyPointNum; i++){
        if(mCurrentFrame.mvpMapPoints[i]){
            if(mCurrentFrame.mvbOutlier[i]){
                // 如果优化后判断某个地图点是外点，清除它的所有关系
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->GetObservations() > 0)
                // 累加成功匹配到的地图点数目
                nmatchesMap++;
        }
    }
    // Step 6：匹配超过10个点就认为跟踪成功
    if(nmatchesMap >= mnLossMMOrRKFVOThreshold){
        mnMotionOrGpsOrRefKFTrackOK = 1;
    }
    else{
        mnMotionOrGpsOrRefKFTrackOK = 0;
    }
    return mnMotionOrGpsOrRefKFTrackOK;
}

/**
 * @brief 用局部地图进行跟踪，进一步优化位姿
 * 
 * 1. 更新局部地图，包括局部关键帧和关键点
 * 2. 对局部MapPoints进行投影匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return true if success
 * 
 * Step 1：更新局部关键帧mvpLocalKeyFrames和局部地图点mvpLocalMapPoints 
 * Step 2：在局部地图中查找与当前帧匹配的MapPoints, 其实也就是对局部地图点进行跟踪
 * Step 3：更新局部所有MapPoints后对位姿再次优化
 * Step 4：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
 * Step 5：决定是否跟踪成功
 */
bool Tracking::TrackWithLocalMap(){
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    // UpdateImgKPMPState Local KeyFrames and Local Points
    // Step 1：更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
    // UpdateImgKPMPState
    // 用共视图来更新局部关键帧和局部地图点
    RefreshLocalKeyFrames();
    RefreshLocalMapPoints();

    // This is for visualization
    // 设置参考地图点用于绘图显示局部地图点（红色）
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Step 2：筛选局部地图中新增的在视野范围内的地图点，投影到当前帧搜索匹配，得到更多的匹配关系
    SearchNewMatchesByLocalMapPoints();

    // Optimize Pose
    // 在这个函数之前，在 Relocalization、TrackWithReferenceKeyFrame、TrackWithMotionModel 中都有位姿优化，
    // Step 3：前面新增了更多的匹配关系，BA优化得到更准确的位姿
    Optimizer::OptimizeFramePose(&mCurrentFrame);
    mnMatchesInliers = 0;
    // UpdateImgKPMPState MapPoints Statistics
    // Step 4：更新当前帧的地图点被观测程度，并统计跟踪局部地图后匹配数目
    for(int i=0; i<mCurrentFrame.mnKeyPointNum; i++){
        if(mCurrentFrame.mvpMapPoints[i]){
            // 由于当前帧的地图点可以被当前帧观测到，其被观测统计量加1
            if(!mCurrentFrame.mvbOutlier[i]){
                // 找到该点的帧数mnFound 加 1
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                //查看当前是否是在纯定位过程
                // 如果该地图点被相机观测数目nObs大于0，匹配内点计数+1
                // mnObserve： 被观测到的相机数目，单目+1，双目或RGB-D则+2
                if(mCurrentFrame.mvpMapPoints[i]->GetObservations() > 0){
                    mnMatchesInliers++;
                }
            }
            // 如果这个地图点是外点,并且当前相机输入还是双目的时候,就删除这个点
            else {
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
            }
        }
    }
    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    // Step 5：根据跟踪匹配数目及重定位情况决定是否跟踪成功
    // 如果最近刚刚发生了重定位,那么至少成功匹配50个点才认为是成功跟踪
    // 如果是正常的状态话只要跟踪的地图点大于30个就认为成功了
    if(mnMatchesInliers < mnLossLMVOThreshold||(mCurrentFrame.mnId< mnLastRelocFrameId + mnfpsByCfgFile && mnMatchesInliers < mnGoodLMVOThreshold)){
        return false;
    }
    else{
        mbBadVO=mnMatchesInliers<mnGoodLMVOThreshold;
        return true;
    }
}



/**
 * @brief 判断当前帧是否需要插入关键帧
 * 
 * Step 1：纯VO模式下不插入关键帧，如果局部地图被闭环检测使用，则不插入关键帧
 * Step 2：如果距离上一次重定位比较近，或者关键帧数目超出最大限制，不插入关键帧
 * Step 3：得到参考关键帧跟踪到的地图点数量
 * Step 4：查询局部地图管理器是否繁忙,也就是当前能否接受新的关键帧
 * Step 5：对于双目或RGBD摄像头，统计可以添加的有效地图点总数 和 跟踪到的地图点数量
 * Step 6：决策是否需要插入关键帧
 * @return true         需要
 * @return false        不需要
 */
bool Tracking::NeedNewKeyFrame(){
    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // Step 2：如果局部地图线程被闭环检测使用，则不插入关键帧
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    // 获取当前地图中的关键帧数目
    const int nKFs = mpMap->GetKeyFramesNumInMap();
    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // mCurrentFrame.mnId是当前帧的ID
    // mnLastRelocFrameId是最近一次重定位帧的ID
    // mMaxFrames等于图像输入的帧率
    //  Step 3：如果距离上一次重定位比较近，并且关键帧数目超出最大限制，不插入关键帧
    if( mCurrentFrame.mnId < mnLastRelocFrameId + mnfpsByCfgFile && nKFs > mnfpsByCfgFile)
        return false;

    // Tracked MapPoints in the reference keyframe
    // Step 4：得到参考关键帧跟踪到的地图点数量
    // RefreshLocalKeyFrames 函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
    // 地图点的最小观测次数
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    // 参考关键帧地图点中观测的数目>= nMinObs的地图点数目
    int nRefMatches = mpReferenceKF->GetNumMapPointsBigObs(nMinObs);

    // Local Mapping accept keyframes?
    // Step 5：查询局部地图线程是否繁忙，当前能否接受新的关键帧
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();


    // Step 7：决策是否需要插入关键帧
    // Thresholds
    // Step 7.1：设定比例阈值，当前帧和参考关键帧跟踪到点的比例，比例越大，越倾向于增加关键帧
    float thRefRatio = 0.7f;
    if(mbBadVO){
        thRefRatio=1.0f;
    }
    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    // Step 7.2：很长时间没有插入关键帧，可以插入
    const bool c1 = mCurrentFrame.mnId>= mnLastKeyFrameId + mnfpsByCfgFile;

    // Local Mapping is idle
    // Step 7.3：localMapper处于空闲状态，可以插入
    const bool c2 = bLocalMappingIdle;

    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    // Step 7.5：和参考帧相比当前跟踪到的点太少 或者满足bNeedToInsertClose；同时跟踪到的内点还不能太少
    const bool c3 = mnMatchesInliers<nRefMatches*thRefRatio;

    if(c2&&(c1||c3)){
        return true;
    }
    else{
        return false;
    }
}

/**
 * @brief 创建新的关键帧
 * 对于非单目的情况，同时创建新的MapPoints
 * 
 * Step 1：将当前帧构造成关键帧
 * Step 2：将当前关键帧设置为当前帧的参考关键帧
 * Step 3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
 */
void Tracking::CreateNewKeyFrame()
{
    // 如果局部建图线程关闭了,就无法插入关键帧
    if(!mpLocalMapper->SetNotStop(true))
        return;

    // Step 1：将当前帧构造成关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    // Step 2：将当前关键帧设置为当前帧的参考关键帧
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    // Step 4：插入关键帧
    // 关键帧插入到列表 mlNewKeyFrames中，等待local mapping线程临幸
    mpLocalMapper->InsertKeyFrameInQueue(pKF);

    // 插入好了，允许局部建图停止
    mpLocalMapper->SetNotStop(false);

    // 当前帧成为新的关键帧，更新
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}


/**
 * @brief 用局部地图点进行投影匹配，得到更多的匹配关系
 * 注意：局部地图点中已经是当前帧地图点的不需要再投影，只需要将此外的并且在视野范围内的点和当前帧进行投影匹配
 */
void Tracking::SearchNewMatchesByLocalMapPoints(){
    // Do not search map points already matched
    // Step 1：遍历当前帧的地图点，标记这些地图点不参与之后的投影搜索匹配
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++){
        MapPoint* pMP = *vit;
        if(pMP){
            if(pMP->GetbBad()){
                *vit = static_cast<MapPoint*>(NULL);
            }
            else{
                // 更新能观测到该点的帧数加1(被当前帧观测了)
                pMP->IncreaseVisible();
                // 标记该点被当前帧观测到
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
            }
        }
    }

    // 准备进行投影匹配的点的数目
    int nToMatch=0;

    // Project points in frame and check its visibility
    // Step 2：判断所有局部地图点中除当前帧地图点外的点，是否在当前帧视野范围内
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++){
        MapPoint* pMP = *vit;
        // 跳过坏点
        if(pMP->GetbBad())
            continue;
        // 已经被当前帧观测到的地图点肯定在视野范围内，跳过
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        // Project (this fills MapPoint variables for matching)
        // 判断地图点是否在在当前帧视野内
        if(mCurrentFrame.isInFrustum(pMP,0.5)){
        	// 观测到该点的帧数加1
            pMP->IncreaseVisible();
            // 只有在视野范围内的地图点才参与之后的投影匹配
            nToMatch++;
        }
    }

    // Step 3：如果需要进行投影匹配的点的数目大于0，就进行投影匹配，增加更多的匹配关系
    if(nToMatch>0){
        ORBmatcher matcher(0.8);
        int th = 5;
        // If the camera has been relocalised recently, perform a coarser search
        // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2){
            th=+5;
        }
        // 投影匹配得到更多的匹配关系
        matcher.SearchFMatchPointByProjectMapPoint(mCurrentFrame, mvpLocalMapPoints, th, false);
    }
}

/*
 * @brief 更新局部关键点。先把局部地图清空，然后将局部关键帧的有效地图点添加到局部地图中
 */
void Tracking::RefreshLocalMapPoints(){
    // Step 1：清空局部地图点
    mvpLocalMapPoints.clear();
    // Step 2：遍历局部关键帧 mvpLocalKeyFrames
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++){
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetAllMapPointVectorInKF(false);
        // step 2：将局部关键帧的地图点添加到mvpLocalMapPoints
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++){
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            // 用该地图点的成员变量mnTrackReferenceForFrame 记录当前帧的id
            // 表示它已经是当前帧的局部地图点了，可以防止重复添加局部地图点
            if(pMP->mnFrameIdForLocalMp == mCurrentFrame.mnId)
                continue;
            if(!pMP->GetbBad()){
                mvpLocalMapPoints.emplace_back(pMP);
                pMP->mnFrameIdForLocalMp=mCurrentFrame.mnId;
            }
        }
    }
}

/**
 * @brief 跟踪局部地图函数里，更新局部关键帧
 * 方法是遍历当前帧的地图点，将观测到这些地图点的关键帧和相邻的关键帧及其父子关键帧，作为mvpLocalKeyFrames
 * Step 1：遍历当前帧的地图点，记录所有能观测到当前帧地图点的关键帧 
 * Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧包括以下3种类型
 *      类型1：能观测到当前帧地图点的关键帧，也称一级共视关键帧
 *      类型2：一级共视关键帧的共视关键帧，称为二级共视关键帧
 *      类型3：一级共视关键帧的子关键帧、父关键帧
 * Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
 */
void Tracking::RefreshLocalKeyFrames(){
    // Each map point vote for the keyframes in which it has been observed
    // Step 1：遍历当前帧的地图点，记录所有能观测到当前帧地图点的关键帧
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.mnKeyPointNum; i++){
        if(mCurrentFrame.mvpMapPoints[i]){
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->GetbBad()){
                // 得到观测到该地图点的关键帧和该地图点在关键帧中的索引
                const map<KeyFrame*,size_t> observations = pMP->GetObservationsKFAndMPIdx();
                // 由于一个地图点可以被多个关键帧观测到,因此对于每一次观测,都对观测到这个地图点的关键帧进行累计投票
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    // map[key] = value，当要插入的键存在时，会覆盖键对应的原来的值。如果键不存在，则添加一组键值对
                    // it->first 是地图点看到的关键帧，同一个关键帧看到的地图点会累加到该关键帧计数
                    // 所以最后keyframeCounter 第一个参数表示某个关键帧，第2个参数表示该关键帧看到了多少当前帧(mCurrentFrame)的地图点，也就是共视程度
                    keyframeCounter[it->first]++;
            }
            else{
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }
    }

    // 没有当前帧没有共视关键帧，返回
    if(keyframeCounter.empty())
        return;

    // 存储具有最多观测次数（max）的关键帧
    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    // Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有3种类型
    // 先清空局部关键帧
    mvpLocalKeyFrames.clear();
    // 先申请3倍内存，不够后面再加
    mvpLocalKeyFrames.reserve(keyframeCounter.size()*5);

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    // Step 2.1 类型1：能观测到当前帧地图点的关键帧作为局部关键帧 （将邻居拉拢入伙）（一级共视关键帧） 
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++){
        KeyFrame* pKF = it->first;
        // 如果设定为要删除的，跳过
        if(pKF->isBad())
            continue;
        // 寻找具有最大观测数目的关键帧
        if(it->second>max){
            max=it->second;
            pKFmax=pKF;
        }
        // 添加到局部关键帧的列表里
        mvpLocalKeyFrames.emplace_back(it->first);
        // 用该关键帧的成员变量mnTrackReferenceForFrame 记录当前帧的id
        // 表示它已经是当前帧的局部关键帧了，可以防止重复添加局部关键帧
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    // Step 2.2 遍历一级共视关键帧，寻找更多的局部关键帧 
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++){
        // Limit the number of keyframes
        // 处理的局部关键帧不超过80帧
        if(mvpLocalKeyFrames.size()>40)
            break;
        KeyFrame* pKF = *itKF;
        // 类型2:一级共视关键帧的共视（前10个）关键帧，称为二级共视关键帧（将邻居的邻居拉拢入伙）
        // 如果共视帧不足10帧,那么就返回所有具有共视关系的关键帧
        const vector<KeyFrame*> vNeighs = pKF->GetCovisibilityForefrontKeyFrames(3);
        // vNeighs 是按照共视程度从大到小排列
        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++){
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad()){
                // mnTrackReferenceForFrame防止重复添加局部关键帧
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId){
                    mvpLocalKeyFrames.emplace_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                }
            }
        }

        // 类型3:将一级共视关键帧的子关键帧作为局部关键帧（将邻居的孩子们拉拢入伙）
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++){
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad()){
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId){
                    mvpLocalKeyFrames.emplace_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    //? 找到一个就直接跳出for循环？
                    break;
                }
            }
        }

        // 类型3:将一级共视关键帧的父关键帧（将邻居的父母们拉拢入伙）
        KeyFrame* pParent = pKF->GetParent();
        if(pParent){
            // mnTrackReferenceForFrame防止重复添加局部关键帧
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId){
                mvpLocalKeyFrames.emplace_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }

    // Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
    if(pKFmax){
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

/**
 * @details 重定位过程
 * @return true 
 * @return false 
 * 
 * Step 1：计算当前帧特征点的词袋向量
 * Step 2：找到与当前帧相似的候选关键帧
 * Step 3：通过BoW进行匹配
 * Step 4：通过EPnP算法估计姿态
 * Step 5：通过PoseOptimization对姿态进行优化求解
 * Step 6：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
 */
bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    // Step 1：计算当前帧特征点的词袋向量
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // InitialOrDoORBTrack Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    // Step 2：用词袋找到与当前帧相似的候选关键帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
    
    // 如果没有候选关键帧，则退出
    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);
    //每个关键帧的解算器
    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    //每个关键帧和当前帧中特征点的匹配关系
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);
    
    //放弃某个关键帧的标记
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    //有效的候选关键帧数目
    int nCandidates=0;

    // Step 3：遍历所有的候选关键帧，通过词袋进行快速匹配，用匹配结果初始化PnP Solver
    for(int i=0; i<nKFs; i++){
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else{
            // 当前帧和候选关键帧用BoW进行快速匹配，匹配结果记录在vvpMapPointMatches，nmatches表示匹配的数目
            int nmatches = matcher.SearchFMatchPointByKFBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            // 如果和当前帧的匹配数小于15,那么只能放弃这个关键帧
            if(nmatches<15){
                vbDiscarded[i] = true;
                continue;
            }
            else{
                // 如果匹配数目够用，用匹配结果初始化EPnPsolver
                // 为什么用EPnP? 因为计算复杂度低，精度高
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(
                    0.99,   //用于计算RANSAC迭代次数理论值的概率
                    10,     //最小内点数, 但是要注意在程序中实际上是min(给定最小内点数,最小集,内点数理论值),不一定使用这个
                    300,    //最大迭代次数
                    4,      //最小集(求解这个问题在一次采样中所需要采样的最少的点的个数,对于Sim3是3,EPnP是4),参与到最小内点数的确定过程中
                    0.5,    //这个是表示(最小内点数/样本总数);实际上的RANSAC正常退出的时候所需要的最小内点数其实是根据这个量来计算得到的
                    5.991); // 自由度为2的卡方检验的阈值,程序中还会根据特征点所在的图层对这个阈值进行缩放
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    // 这里的 P4P RANSAC是Epnp，每次迭代需要4个点
    // 是否已经找到相匹配的关键帧的标志
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    // Step 4: 通过一系列操作,直到找到能够匹配上的关键帧
    // 为什么搞这么复杂？答：是担心误闭环
    while(nCandidates>0 && !bMatch){
        //遍历当前所有的候选关键帧
        for(int i=0; i<nKFs; i++){
            // 忽略放弃的
            if(vbDiscarded[i])
                continue;
            //内点标记
            vector<bool> vbInliers;
            //内点数
            int nInliers;
            // 表示RANSAC已经没有更多的迭代次数可用 -- 也就是说数据不够好，RANSAC也已经尽力了。。。
            bool bNoMore;

            // Step 4.1：通过EPnP算法估计姿态，迭代5次
            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            // bNoMore 为true 表示已经超过了RANSAC最大迭代次数，就放弃当前关键帧
            if(bNoMore){
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty()){
                //  Step 4.2：如果EPnP 计算出了位姿，对内点进行BA优化
                Tcw.copyTo(mCurrentFrame.mTcw);
                
                // EPnP 里RANSAC后的内点的集合
                set<MapPoint*> sFound;

                const int np = vbInliers.size();
                //遍历所有内点
                for(int j=0; j<np; j++){
                    if(vbInliers[j]){
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else{
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                    }
                }

                // 只优化位姿,不优化地图点的坐标，返回的是内点的数量
                int nGood = Optimizer::OptimizeFramePose(&mCurrentFrame);

                // 如果优化之后的内点数目不多，跳过了当前候选关键帧,但是却没有放弃当前帧的重定位
                if(nGood<10)
                    continue;

                // 删除外点对应的地图点
                for(int io =0; io<mCurrentFrame.mnKeyPointNum; io++)
                    if(mCurrentFrame.mvbOutlier[io]){
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);
                    }

                // If few inliers, search by projection in a coarse window and optimize again
                // Step 4.3：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
                // 前面的匹配关系是用词袋匹配过程得到的
                if(nGood<50)
                {
                    // 通过投影的方式将关键帧中未匹配的地图点投影到当前帧中, 生成新的匹配
                    int nadditional = matcher2.SearchFMatchPointByProjectKeyFrame(
                            mCurrentFrame,          //当前帧
                            vpCandidateKFs[i],      //关键帧
                            sFound,                 //已经找到的地图点集合，不会用于PNP
                            10,                     //窗口阈值，会乘以金字塔尺度
                            100);                   //匹配的ORB描述子距离应该小于这个阈值

                    // 如果通过投影过程新增了比较多的匹配特征点对
                    if(nadditional+nGood>=50){
                        // 根据投影匹配的结果，再次采用3D-2D pnp BA优化位姿
                        nGood = Optimizer::OptimizeFramePose(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        // Step 4.4：如果BA后内点数还是比较少(<50)但是还不至于太少(>30)，可以挽救一下, 最后垂死挣扎 
                        // 重新执行上一步 4.3的过程，只不过使用更小的搜索窗口
                        // 这里的位姿已经使用了更多的点进行了优化,应该更准，所以使用更小的窗口搜索
                        if(nGood>30 && nGood<50){
                            // 用更小窗口、更严格的描述子阈值，重新进行投影搜索匹配
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.mnKeyPointNum; ip++) {
                                if (mCurrentFrame.mvpMapPoints[ip]){
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                }
                            }
                            nadditional = matcher2.SearchFMatchPointByProjectKeyFrame(
                                    mCurrentFrame,          //当前帧
                                    vpCandidateKFs[i],      //候选的关键帧
                                    sFound,                 //已经找到的地图点，不会用于PNP
                                    3,                      //新的窗口阈值，会乘以金字塔尺度
                                    64);                    //匹配的ORB描述子距离应该小于这个阈值

                            // Final optimization
                            // 如果成功挽救回来，匹配数目达到要求，最后BA优化一下
                            if(nGood+nadditional>=50){
                                nGood = Optimizer::OptimizeFramePose(&mCurrentFrame);
                                //更新地图点
                                for(int io =0; io<mCurrentFrame.mnKeyPointNum; io++){
                                    if(mCurrentFrame.mvbOutlier[io]){
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                                    }
                                }
                            }
                            //如果还是不能够满足就放弃了
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                // 如果对于当前的候选关键帧已经有足够的内点(50个)了,那么就认为重定位成功
                if(nGood>=50){
                    bMatch = true;
                    // 只要有一个候选关键帧重定位成功，就退出循环，不考虑其他候选关键帧了
                    break;
                }
            }
        }//一直运行,知道已经没有足够的关键帧,或者是已经有成功匹配上的关键帧
    }

    // 折腾了这么久还是没有匹配上，重定位失败
    if(!bMatch){
        return false;
    }
    else{
        // 如果匹配上了,说明当前帧重定位成功了(当前帧已经有了自己的位姿)
        // 记录成功重定位帧的id，防止短时间多次重定位
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}

//整个追踪线程执行复位操作
void Tracking::Reset()
{
    //基本上是挨个请求各个线程终止

    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    cout << "System Reseting" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clearAllKFinDB();
    cout << " done" << endl;



    // Clear Map (this eraseKFromDB MapPoints and KeyFrames)
    mpMap->clear();
    mSim3VoGps=cv::Mat();
    mnSim3Inliers=0;
    //然后复位各种变量
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

} //namespace ORB_SLAM
