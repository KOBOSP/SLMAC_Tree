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

namespace ORB_SLAM2 {

///构造函数
    Tracking::Tracking(
            System *pSys,                       //系统实例
            ORBVocabulary *pVoc,                //BOW字典
            FrameDrawer *pFrameDrawer,          //帧绘制器
            MapDrawer *pMapDrawer,              //地图点绘制器
            Map *pMap,                          //地图句柄
            KeyFrameDatabase *pKFDB,            //关键帧产生的词袋数据库
            const string &strSettingPath,       //配置文件路径
            const int sensor) :                  //传感器类型
            mState(NO_IMAGES_YET),                              //当前系统还没有准备好
            mSensor(sensor),
            mbOnlyTracking(false),                              //处于SLAM模式
            mpORBVocabulary(pVoc),
            mpKeyFrameDB(pKFDB),
            mpInitializer(static_cast<Initializer *>(NULL)),     //暂时给地图初始化器设置为空指针
            mpSystem(pSys),
            mpViewer(
                    NULL),                                     //注意可视化的查看器是可选的，因为ORB-SLAM2最后是被编译成为一个库，所以对方人拿过来用的时候也应该有权力说我不要可视化界面（何况可视化界面也要占用不少的CPU资源）
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

        //     |fx  0   cx|
        // K = |0   fy  cy|
        //     |0   0   1 |
        //构造相机内参矩阵
        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        // 图像矫正系数
        // [k1 k2 p1 p2 k3]
        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        //有些相机的畸变系数中会没有k3项
        if (k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        // 双目摄像头baseline * fx 50
        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if (fps == 0)
            fps = 30;

        // Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 0;
        mMaxFrames = fps;

        //输出
        cout << endl << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if (DistCoef.rows == 5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;

        // 1:RGB 0:BGR
        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
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

        // tracking过程都会用到mpORBextractorLeft作为特征点提取器
        mpORBextractorLeft = new ORBextractor(
                nFeatures,      //参数的含义还是看上面的注释吧
                fScaleFactor,
                nLevels,
                fIniThFAST,
                fMinThFAST);


        mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        cout << endl << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    }

//设置局部建图器
    void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) {
        mpLocalMapper = pLocalMapper;
    }

//设置回环检测器
    void Tracking::SetLoopClosing(LoopClosing *pLoopClosing) {
        mpLoopClosing = pLoopClosing;
    }

//设置可视化查看器
    void Tracking::SetViewer(Viewer *pViewer) {
        mpViewer = pViewer;
    }

/**
 * @brief 
 * 输入左目RGB或RGBA图像，输出世界坐标系到该帧相机坐标系的变换矩阵
 * 
 * @param[in] im 单目图像
 * @param[in] timestamp 时间戳
 * @return cv::Mat 
 * 
 * Step 1 ：将彩色图像转为灰度图像
 * Step 2 ：构造Frame
 * Step 3 ：跟踪
 */
    cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp) {
        mImGray = im;

        // Step 1 ：将彩色图像转为灰度图像
        //若图片是3、4通道的，还需要转化成灰度图
        if (mImGray.channels() == 3) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        } else if (mImGray.channels() == 4) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET) //没有成功初始化的前一个状态就是NO_IMAGES_YET
            mCurrentFrame = Frame(
                    mImGray,
                    timestamp,
                    mpIniORBextractor,      //初始化ORB特征点提取器会提取2倍的指定特征点数目
                    mpORBVocabulary,
                    mK,
                    mDistCoef,
                    mbf,
                    mThDepth);
        else
            mCurrentFrame = Frame(
                    mImGray,
                    timestamp,
                    mpORBextractorLeft,     //正常运行的时的ORB特征点提取器，提取指定数目特征点
                    mpORBVocabulary,
                    mK,
                    mDistCoef,
                    mbf,
                    mThDepth);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        mdExtraFps = 1.0 / std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        t1 = std::chrono::steady_clock::now();
        Track();
        t2 = std::chrono::steady_clock::now();
        mdTrackFps = 1.0 / std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        //返回当前帧的位姿
        return mCurrentFrame.mTcw.clone();
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
    void Tracking::Track() {
        // track包含两部分：估计运动、跟踪局部地图

        // mState为tracking的状态，包括 SYSTME_NOT_READY, NO_IMAGE_YET, NOT_INITIALIZED, OK, LOST
        // 如果图像复位过、或者第一次运行，则为NO_IMAGE_YET状态
        if (mState == NO_IMAGES_YET) {
            mState = NOT_INITIALIZED;
        }

        mLastProcessedState = mState;


        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        if (mState == NOT_INITIALIZED) {
            MonocularInitialization();
            mpFrameDrawer->Update(this);
            if (mState != OK)
                return;
        } else {
            bool bOK = false;
            mnTrackMethod = 0;
            if (mState == OK) {
                CheckReplacedInLastFrame();
                if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                    bOK = TrackReferenceKeyFrame();
                } else {
                    bOK = TrackWithMotionModel();
                    if (!bOK) {
                        bOK = TrackReferenceKeyFrame();
                    } else if (mnTrackMethod == 0) {
                        mnTrackMethod = 2;//Motion
                    }
                }
                if (bOK && mnTrackMethod == 0) {
                    mnTrackMethod = 3;//RefKF
                }
            } else {
                bOK = Relocalization();
                if (bOK) {
                    mnTrackMethod = 4;//Reloc
                }
            }

            mCurrentFrame.mpReferenceKF = mpReferenceKF;
            if (bOK)
                bOK = TrackLocalMap();

            if (bOK)
                mState = OK;
            else
                mState = LOST;

            mpFrameDrawer->Update(this);

            if (bOK) {
                // Update motion model
                if (!mLastFrame.mTcw.empty()) {
                    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                    mVelocity = mCurrentFrame.mTcw * LastTwc;
                } else {
                    mVelocity = cv::Mat();
                }

                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

                // Clean VO matches
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (pMP)
                        if (pMP->Observations() < 1) {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                        }
                }

                // Delete temporal MapPoints
                // Step 7：清除恒速模型跟踪中 UpdateLastFrame中为当前帧临时添加的MapPoints（仅双目和rgbd）
                // 步骤6中只是在当前帧中将这些MapPoints剔除，这里从MapPoints数据库中删除
                // 临时地图点仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中
                for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end();
                     lit != lend; lit++) {
                    MapPoint *pMP = *lit;
                    delete pMP;
                }

                // 这里不仅仅是清除mlpTemporalPoints，通过delete pMP还删除了指针指向的MapPoint
                // 不能够直接执行这个是因为其中存储的都是指针,之前的操作都是为了避免内存泄露
                mlpTemporalPoints.clear();

                // Check if we need to insert a new keyframe
                // Step 8：检测并插入关键帧，对于双目或RGB-D会产生新的地图点
                if (NeedNewKeyFrame())
                    CreateNewKeyFrame();

                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame.
                // 作者这里说允许在BA中被Huber核函数判断为外点的传入新的关键帧中，让后续的BA来审判他们是不是真正的外点
                // 但是估计下一帧位姿的时候我们不想用这些外点，所以删掉

                //  Step 9 删除那些在bundle adjustment中检测为outlier的地图点
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    // 这里第一个条件还要执行判断是因为, 前面的操作中可能删除了其中的地图点
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }
            }

            // Reset if the camera get lost soon after initialization
            // Step 10 如果初始化后不久就跟踪失败，并且relocation也没有搞定，只能重新Reset
            if (mState == LOST && mpMap->KeyFramesInMap() <= 5) {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            mLastFrame = Frame(mCurrentFrame);
        }

        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        // Step 11：记录位姿信息，用于最后保存所有的轨迹
        if (!mCurrentFrame.mTcw.empty()) {
            // 计算相对姿态Tcr = Tcw * Twr, Twr = Trw^-1
            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            //保存各种状态
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST);
        } else {
            // This can happen if tracking is lost
            // 如果跟踪失败，则相对位姿使用上一次值
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState == LOST);
        }

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
    void Tracking::MonocularInitialization() {
        // Step 1 如果单目初始器还没有被创建，则创建。后面如果重新初始化时会清掉这个
        if (!mpInitializer) {
            // Set Reference Frame
            // 单目初始帧的特征点数必须大于100
            if (mCurrentFrame.mvKeys.size() > 100) {
                // 初始化需要两帧，分别是mInitialFrame，mCurrentFrame
                mInitialFrame = Frame(mCurrentFrame);
                // 用当前帧更新上一帧
                mLastFrame = Frame(mCurrentFrame);
                // mvbPrevMatched  记录"上一帧"所有特征点
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
                for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                    mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

                // 删除前判断一下，来避免出现段错误。不过在这里是多余的判断
                // 不过在这里是多余的判断，因为前面已经判断过了
                if (mpInitializer)
                    delete mpInitializer;

                // 由当前帧构造初始器 sigma:1.0 iterations:200
                mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

                // 初始化为-1 表示没有任何匹配。这里面存储的是匹配的点的id
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

                return;
            }
        } else    //如果单目初始化器已经被创建
        {
            // Try to initialize
            // Step 2 如果当前帧特征点数太少（不超过100），则重新构造初始器
            // NOTICE 只有连续两帧的特征点个数都大于100时，才能继续进行初始化过程
            if ((int) mCurrentFrame.mvKeys.size() <= 100) {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
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
            int nmatches = matcher.SearchForInitialization(
                    mInitialFrame, mCurrentFrame,    //初始化时的参考帧和当前帧
                    mvbPrevMatched,                 //在初始化参考帧中提取得到的特征点
                    mvIniMatches,                   //保存匹配关系
                    100);                           //搜索窗口大小

            // Check if there are enough correspondences
            // Step 4 验证匹配结果，如果初始化的两帧之间的匹配点太少，重新初始化
            if (nmatches < 100) {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                return;
            }

            cv::Mat Rcw; // Current Camera Rotation
            cv::Mat tcw; // Current Camera Translation
            vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

            // Step 5 通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
            if (mpInitializer->Initialize(
                    mCurrentFrame,      //当前帧
                    mvIniMatches,       //当前帧和参考帧的特征点的匹配关系
                    Rcw, tcw,           //初始化得到的相机的位姿
                    mvIniP3D,           //进行三角化得到的空间点集合
                    vbTriangulated))    //以及对应于mvIniMatches来讲,其中哪些点被三角化了
            {
                // Step 6 初始化成功后，删除那些无法进行三角化的匹配点
                for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
                    if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
                        mvIniMatches[i] = -1;
                        nmatches--;
                    }
                }

                // Set Frame Poses
                // Step 7 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
                mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
                // 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到相机坐标系的变换矩阵
                cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
                Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
                tcw.copyTo(Tcw.rowRange(0, 3).col(3));
                mCurrentFrame.SetPose(Tcw);

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
    void Tracking::CreateInitialMapMonocular() {
        // Create KeyFrames 认为单目初始化时候的参考帧和当前帧都是关键帧
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);  // 第一帧
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);  // 第二帧

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
        for (size_t i = 0; i < mvIniMatches.size(); i++) {
            // 没有匹配，跳过
            if (mvIniMatches[i] < 0)
                continue;

            //Create MapPoint.
            // 用三角化点初始化为空间点的世界坐标
            cv::Mat worldPos(mvIniP3D[i]);

            // Step 3.1 用3D点构造MapPoint
            MapPoint *pMP = new MapPoint(
                    worldPos,
                    pKFcur,
                    mpMap);

            // Step 3.2 为该MapPoint添加属性：
            // a.观测到该MapPoint的关键帧
            // b.该MapPoint的描述子
            // c.该MapPoint的平均观测方向和深度范围

            // 表示该KeyFrame的2D特征点和对应的3D地图点
            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            // a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

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

        // Update Connections
        // Step 3.3 更新关键帧间的连接关系
        // 在3D点和关键帧之间建立边，每个边有一个权重，边的权重是该关键帧与当前帧公共3D点的个数
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        // Bundle Adjustment
        cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

        // Step 4 全局BA优化，同时优化所有位姿和三维点
        Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

        // Set median depth to 1
        // Step 5 取场景的中值深度，用于尺度归一化
        // 为什么是 pKFini 而不是 pKCur ? 答：都可以的，内部做了位姿变换了
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f / medianDepth;

        //两个条件,一个是平均深度要大于0,另外一个是在当前帧中被观测到的地图点的数目应该大于100
        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100) {
            cout << "Wrong initialization, reseting..." << endl;
            Reset();
            return;
        }

        // Step 6 将两帧之间的变换归一化到平均深度1的尺度下
        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        // x/z y/z 将z归一化到1
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale points
        // Step 7 把3D点的尺度也归一化到1
        // 为什么是pKFini? 是不是就算是使用 pKFcur 得到的结果也是相同的? 答：是的，因为是同样的三维点
        vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
            if (vpAllMapPoints[iMP]) {
                MapPoint *pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            }
        }

        //  Step 8 将关键帧插入局部地图，更新归一化后的位姿、局部地图点
        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        // 单目初始化之后，得到的初始地图中的所有点都是局部地图点
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        //也只能这样子设置了,毕竟是最近的关键帧
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame);

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;// 初始化成功，至此，初始化过程完成
    }

/*
 * @brief 检查上一帧中的地图点是否需要被替换
 * 
 * Local Mapping线程可能会将关键帧中某些地图点进行替换，由于tracking中需要用到上一帧地图点，所以这里检查并更新上一帧中被替换的地图点
 * @see LocalMapping::FuseMapPointsInNeighbors()
 */
    void Tracking::CheckReplacedInLastFrame() {
        for (int i = 0; i < mLastFrame.N; i++) {
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];
            //如果这个地图点存在
            if (pMP) {
                // 获取其是否被替换,以及替换后的点
                // 这也是程序不直接删除这个地图点删除的原因
                MapPoint *pRep = pMP->GetReplaced();
                if (pRep) {
                    //然后替换一下
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }
    }

/*
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
    bool Tracking::TrackReferenceKeyFrame() {
        // Compute Bag of Words vector
        // Step 1：将当前帧的描述子转化为BoW向量
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);
        vector<MapPoint *> vpMapPointMatches;

        // Step 2：通过词袋BoW加速当前帧与参考帧之间的特征点匹配
        int nmatches = matcher.SearchMatchFrameAndKFByBoW(
                mpReferenceKF,          //参考关键帧
                mCurrentFrame,          //当前帧
                vpMapPointMatches);     //存储匹配关系

        // 匹配数目小于15，认为跟踪失败
        if (nmatches < 15)
            return false;

        // Step 3:将上一帧的位姿态作为当前帧位姿的初始值
        mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        mCurrentFrame.SetPose(mLastFrame.mTcw); // 用上一次的Tcw设置初值，在PoseOptimization可以收敛快一些

        // Step 4:通过优化3D-2D的重投影误差来获得位姿
        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        // Step 5：剔除优化后的匹配点中的外点
        //之所以在优化之后才剔除外点，是因为在优化的过程中就有了对这些外点的标记
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                //如果对应到的某个特征点是外点
                if (mCurrentFrame.mvbOutlier[i]) {
                    //清除它在当前帧中存在过的痕迹
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    //匹配的内点计数++
                    nmatchesMap++;
            }
        }
        // 跟踪成功的数目超过10才认为跟踪成功，否则跟踪失败
        return nmatchesMap >= 10;
    }

/**
 * @brief 更新上一帧位姿，在上一帧中生成临时地图点
 * 单目情况：只计算了上一帧的世界坐标系位姿
 * 双目和rgbd情况：选取有有深度值的并且没有被选为地图点的点生成新的临时地图点，提高跟踪鲁棒性
 */
    void Tracking::UpdateLastFrame() {
        // Update pose according to reference keyframe
        // Step 1：利用参考关键帧更新上一帧在世界坐标系下的位姿
        // 上一普通帧的参考关键帧，注意这里用的是参考关键帧（位姿准）而不是上上一帧的普通帧
        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        // ref_keyframe 到 lastframe的位姿变换
        cv::Mat Tlr = mlRelativeFramePoses.back();

        // 将上一帧的世界坐标系下的位姿计算出来
        // l:last, r:reference, w:world
        // Tlw = Tlr*Trw
        mLastFrame.SetPose(Tlr * pRef->GetPose());

        // 如果上一帧为关键帧，或者单目的情况，则退出
        if (mnLastKeyFrameId == mLastFrame.mnId)
            return;

    }

/**
 * @brief 根据恒定速度模型用上一帧地图点来对当前帧进行跟踪
 * Step 1：更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
 * Step 2：根据上一帧特征点对应地图点进行投影匹配
 * Step 3：优化当前帧位姿
 * Step 4：剔除地图点中外点
 * @return 如果匹配数大于10，认为跟踪成功，返回true
 */
    bool Tracking::TrackWithMotionModel() {
        // 最小距离 < 0.9*次小距离 匹配成功，检查旋转
        ORBmatcher matcher(0.9, true);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points
        // Step 1：更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
        UpdateLastFrame();

        // Step 2：根据之前估计的速度，用恒速模型得到当前帧的初始位姿。
        mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

        // 清空当前帧的地图点
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

        // Project points seen in previous frame
        // 设置特征匹配过程中的搜索半径
        int th = 15;//单目


        // Step 3：用上一帧地图点进行投影匹配，如果匹配点不够，则扩大搜索半径再来一次
        int nmatches = matcher.SearchFrameAndFrameByProject(mCurrentFrame, mLastFrame, th, true);

        // If few matches, uses a wider window search
        // 如果匹配点太少，则扩大搜索半径再来一次
        if (nmatches < 20) {
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
            nmatches = matcher.SearchFrameAndFrameByProject(mCurrentFrame, mLastFrame, 2 * th,
                                                            true); // 2*th
        }

        // 如果还是不能够获得足够的匹配点,那么就认为跟踪失败
        if (nmatches < 20)
            return false;

        // Optimize frame pose with all matches
        // Step 4：利用3D-2D投影关系，优化当前帧位姿
        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        // Step 5：剔除地图点中外点
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    // 如果优化后判断某个地图点是外点，清除它的所有关系
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    // 累加成功匹配到的地图点数目
                    nmatchesMap++;
            }
        }

        mnTrackedMotion=nmatchesMap;
        // Step 6：匹配超过10个点就认为跟踪成功
        return nmatchesMap >= 10;
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
    bool Tracking::TrackLocalMap() {
        // We have an estimation of the camera pose and some map points tracked in the frame.
        // We retrieve the local map and try to find matches to points in the local map.

        // Update Local KeyFrames and Local Points
        // Step 1：更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
        UpdateLocalMap();

        // Step 2：筛选局部地图中新增的在视野范围内的地图点，投影到当前帧搜索匹配，得到更多的匹配关系
        SearchLocalPoints();

        // Optimize Pose
        // 在这个函数之前，在 Relocalization、TrackReferenceKeyFrame、TrackWithMotionModel 中都有位姿优化，
        // Step 3：前面新增了更多的匹配关系，BA优化得到更准确的位姿
        Optimizer::PoseOptimization(&mCurrentFrame);
        mnMatchesInliers = 0;

        // Update MapPoints Statistics
        // Step 4：更新当前帧的地图点被观测程度，并统计跟踪局部地图后匹配数目
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                // 由于当前帧的地图点可以被当前帧观测到，其被观测统计量加1
                if (!mCurrentFrame.mvbOutlier[i]) {
                    // 找到该点的帧数mnFound 加 1
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    //查看当前是否是在纯定位过程
                    if (!mbOnlyTracking) {
                        // 如果该地图点被相机观测数目nObs大于0，匹配内点计数+1
                        // nObs： 被观测到的相机数目，单目+1，双目或RGB-D则+2
                        if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                            mnMatchesInliers++;
                    } else
                        // 记录当前帧跟踪到的地图点数目，用于统计跟踪效果
                        mnMatchesInliers++;
                } else {
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }
            }
        }

        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        // Step 5：根据跟踪匹配数目及重定位情况决定是否跟踪成功
        // 如果最近刚刚发生了重定位,那么至少成功匹配50个点才认为是成功跟踪
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
            return false;

        //如果是正常的状态话只要跟踪的地图点大于30个就认为成功了
        if (mnMatchesInliers < 30)
            return false;
        else
            return true;
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
    bool Tracking::NeedNewKeyFrame() {
        // Step 1：纯VO模式下不插入关键帧
        if (mbOnlyTracking)
            return false;

        // If Local Mapping is freezed by a Loop Closure do not insert keyframes
        // Step 2：如果局部地图线程被闭环检测使用，则不插入关键帧
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
            return false;
        // 获取当前地图中的关键帧数目
        const int nKFs = mpMap->KeyFramesInMap();

        // Do not insert keyframes if not enough frames have passed from last relocalisation
        // mCurrentFrame.mnId是当前帧的ID
        // mnLastRelocFrameId是最近一次重定位帧的ID
        // mMaxFrames等于图像输入的帧率
        //  Step 3：如果距离上一次重定位比较近，并且关键帧数目超出最大限制，不插入关键帧
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
            return false;

        // Tracked MapPoints in the reference keyframe
        // Step 4：得到参考关键帧跟踪到的地图点数量
        // UpdateLocalKeyFrames 函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧

        // 地图点的最小观测次数
        int nMinObs = 3;
        if (nKFs <= 2)
            nMinObs = 2;
        // 参考关键帧地图点中观测的数目>= nMinObs的地图点数目
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

        // Local Mapping accept keyframes?
        // Step 5：查询局部地图线程是否繁忙，当前能否接受新的关键帧
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

        // Check how many "close" points are being tracked and how many could be potentially created.
        // Step 6：对于双目或RGBD摄像头，统计成功跟踪的近点的数量，如果跟踪到的近点太少，没有跟踪到的近点较多，可以插入关键帧
        int nNonTrackedClose = 0;  //双目或RGB-D中没有跟踪到的近点
        int nTrackedClose = 0;       //双目或RGB-D中成功跟踪的近点（三维点）

        // 双目或RGBD情况下：跟踪到的地图点中近点太少 同时 没有跟踪到的三维点太多，可以插入关键帧了
        // 单目时，为false
        bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

        // Step 7：决策是否需要插入关键帧
        // Thresholds
        // Step 7.1：设定比例阈值，当前帧和参考关键帧跟踪到点的比例，比例越大，越倾向于增加关键帧
        float thRefRatio = 0.75f;

        // 关键帧只有一帧，那么插入关键帧的阈值设置的低一点，插入频率较低
        if (nKFs < 2)
            thRefRatio = 0.4f;

        thRefRatio = 0.9f;

        // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        // Step 7.2：很长时间没有插入关键帧，可以插入
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;

        // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        // Step 7.3：满足插入关键帧的最小间隔并且localMapper处于空闲状态，可以插入
        const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);

        // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        // Step 7.5：和参考帧相比当前跟踪到的点太少 或者满足bNeedToInsertClose；同时跟踪到的内点还不能太少
        const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15);

        if ((c1a || c1b) && c2) {
            // If the mapping accepts keyframes, insert keyframe.
            // Otherwise send a signal to interrupt BA
            // Step 7.6：local mapping空闲时可以直接插入，不空闲的时候要根据情况插入
            if (bLocalMappingIdle) {
                //可以插入关键帧
                return true;
            } else {
                mpLocalMapper->InterruptBA();
                return false;
            }
        } else
            //不满足上面的条件,自然不能插入关键帧
            return false;
    }

/**
 * @brief 创建新的关键帧
 * 对于非单目的情况，同时创建新的MapPoints
 * 
 * Step 1：将当前帧构造成关键帧
 * Step 2：将当前关键帧设置为当前帧的参考关键帧
 * Step 3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
 */
    void Tracking::CreateNewKeyFrame() {
        // 如果局部建图线程关闭了,就无法插入关键帧
        if (!mpLocalMapper->SetNotStop(true))
            return;

        // Step 1：将当前帧构造成关键帧
        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);
        // Step 2：将当前关键帧设置为当前帧的参考关键帧
        // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        // Step 4：插入关键帧
        // 关键帧插入到列表 mlNewKeyFrames中，等待local mapping线程临幸
        mpLocalMapper->InsertKeyFrame(pKF);

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
    void Tracking::SearchLocalPoints() {
        // Do not search map points already matched
        // Step 1：遍历当前帧的地图点，标记这些地图点不参与之后的投影搜索匹配
        for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if (pMP) {
                if (pMP->isBad()) {
                    *vit = static_cast<MapPoint *>(NULL);
                } else {
                    // 更新能观测到该点的帧数加1(被当前帧观测了)
                    pMP->IncreaseVisible();
                    // 标记该点被当前帧观测到
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    // 标记该点在后面搜索匹配时不被投影，因为已经有匹配了
                    pMP->mbTrackInView = false;
                }
            }
        }

        // 准备进行投影匹配的点的数目
        int nToMatch = 0;

        // Project points in frame and check its visibility
        // Step 2：判断所有局部地图点中除当前帧地图点外的点，是否在当前帧视野范围内
        for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;

            // 已经被当前帧观测到的地图点肯定在视野范围内，跳过
            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            // 跳过坏点
            if (pMP->isBad())
                continue;

            // Project (this fills MapPoint variables for matching)
            // 判断地图点是否在在当前帧视野内
            if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
                // 观测到该点的帧数加1
                pMP->IncreaseVisible();
                // 只有在视野范围内的地图点才参与之后的投影匹配
                nToMatch++;
            }
        }

        // Step 3：如果需要进行投影匹配的点的数目大于0，就进行投影匹配，增加更多的匹配关系
        if (nToMatch > 0) {
            ORBmatcher matcher(0.8);
            int th = 1;

            // If the camera has been relocalised recently, perform a coarser search
            // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;

            // 投影匹配得到更多的匹配关系
            matcher.SearchReplaceFrameAndMPsByProject(mCurrentFrame, mvpLocalMapPoints, th);
        }
    }

/**
 * @brief 更新LocalMap
 *
 * 局部地图包括： 
 * 1、K1个关键帧、K2个临近关键帧和参考关键帧
 * 2、由这些关键帧观测到的MapPoints
 */
    void Tracking::UpdateLocalMap() {
        // This is for visualization
        // 设置参考地图点用于绘图显示局部地图点（红色）
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        // Update
        // 用共视图来更新局部关键帧和局部地图点
        UpdateLocalKeyFrames();
        UpdateLocalPoints();
    }

/*
 * @brief 更新局部关键点。先把局部地图清空，然后将局部关键帧的有效地图点添加到局部地图中
 */
    void Tracking::UpdateLocalPoints() {
        // Step 1：清空局部地图点
        mvpLocalMapPoints.clear();

        // Step 2：遍历局部关键帧 mvpLocalKeyFrames
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            KeyFrame *pKF = *itKF;
            const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

            // step 2：将局部关键帧的地图点添加到mvpLocalMapPoints
            for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end();
                 itMP != itEndMP; itMP++) {
                MapPoint *pMP = *itMP;
                if (!pMP)
                    continue;
                // 用该地图点的成员变量mnTrackReferenceForFrame 记录当前帧的id
                // 表示它已经是当前帧的局部地图点了，可以防止重复添加局部地图点
                if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pMP->isBad()) {
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
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
    void Tracking::UpdateLocalKeyFrames() {
        // Each map point vote for the keyframes in which it has been observed
        // Step 1：遍历当前帧的地图点，记录所有能观测到当前帧地图点的关键帧
        map<KeyFrame *, int> keyframeCounter;
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP->isBad()) {
                    // 得到观测到该地图点的关键帧和该地图点在关键帧中的索引
                    const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                    // 由于一个地图点可以被多个关键帧观测到,因此对于每一次观测,都对观测到这个地图点的关键帧进行累计投票
                    for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end();
                         it != itend; it++)
                        // 这里的操作非常精彩！
                        // map[key] = value，当要插入的键存在时，会覆盖键对应的原来的值。如果键不存在，则添加一组键值对
                        // it->first 是地图点看到的关键帧，同一个关键帧看到的地图点会累加到该关键帧计数
                        // 所以最后keyframeCounter 第一个参数表示某个关键帧，第2个参数表示该关键帧看到了多少当前帧(mCurrentFrame)的地图点，也就是共视程度
                        keyframeCounter[it->first]++;
                } else {
                    mCurrentFrame.mvpMapPoints[i] = NULL;
                }
            }
        }

        // 没有当前帧没有共视关键帧，返回
        if (keyframeCounter.empty())
            return;

        // 存储具有最多观测次数（max）的关键帧
        int max = 0;
        KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

        // Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有3种类型
        // 先清空局部关键帧
        mvpLocalKeyFrames.clear();
        // 先申请3倍内存，不够后面再加
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

        // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        // Step 2.1 类型1：能观测到当前帧地图点的关键帧作为局部关键帧 （将邻居拉拢入伙）（一级共视关键帧）
        for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end();
             it != itEnd; it++) {
            KeyFrame *pKF = it->first;

            // 如果设定为要删除的，跳过
            if (pKF->isBad())
                continue;

            // 寻找具有最大观测数目的关键帧
            if (it->second > max) {
                max = it->second;
                pKFmax = pKF;
            }

            // 添加到局部关键帧的列表里
            mvpLocalKeyFrames.push_back(it->first);

            // 用该关键帧的成员变量mnTrackReferenceForFrame 记录当前帧的id
            // 表示它已经是当前帧的局部关键帧了，可以防止重复添加局部关键帧
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }


        // Include also some not-already-included keyframes that are neighbors to already-included keyframes
        // Step 2.2 遍历一级共视关键帧，寻找更多的局部关键帧
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            // Limit the number of keyframes
            // 处理的局部关键帧不超过80帧
            if (mvpLocalKeyFrames.size() > 80)
                break;

            KeyFrame *pKF = *itKF;

            // 类型2:一级共视关键帧的共视（前10个）关键帧，称为二级共视关键帧（将邻居的邻居拉拢入伙）
            // 如果共视帧不足10帧,那么就返回所有具有共视关系的关键帧
            const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
            // vNeighs 是按照共视程度从大到小排列
            for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end();
                 itNeighKF != itEndNeighKF; itNeighKF++) {
                KeyFrame *pNeighKF = *itNeighKF;
                if (!pNeighKF->isBad()) {
                    // mnTrackReferenceForFrame防止重复添加局部关键帧
                    if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pNeighKF);
                        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        //? 找到一个就直接跳出for循环？
                        break;
                    }
                }
            }

            // 类型3:将一级共视关键帧的子关键帧作为局部关键帧（将邻居的孩子们拉拢入伙）
            const set<KeyFrame *> spChilds = pKF->GetChilds();
            for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++) {
                KeyFrame *pChildKF = *sit;
                if (!pChildKF->isBad()) {
                    if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pChildKF);
                        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        //? 找到一个就直接跳出for循环？
                        break;
                    }
                }
            }

            // 类型3:将一级共视关键帧的父关键帧（将邻居的父母们拉拢入伙）
            KeyFrame *pParent = pKF->GetParent();
            if (pParent) {
                // mnTrackReferenceForFrame防止重复添加局部关键帧
                if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pParent);
                    pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    //! 感觉是个bug！如果找到父关键帧会直接跳出整个循环
                    break;
                }
            }

        }

        // Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
        if (pKFmax) {
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
    bool Tracking::Relocalization() {
        // Compute Bag of Words Vector
        // Step 1：计算当前帧特征点的词袋向量
        mCurrentFrame.ComputeBoW();

        // Relocalization is performed when tracking is lost
        // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
        // Step 2：用词袋找到与当前帧相似的候选关键帧
        vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

        // 如果没有候选关键帧，则退出
        if (vpCandidateKFs.empty())
            return false;

        const int nKFs = vpCandidateKFs.size();

        // We perform first an ORB matching with each candidate
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.75, true);
        //每个关键帧的解算器
        vector<PnPsolver *> vpPnPsolvers;
        vpPnPsolvers.resize(nKFs);

        //每个关键帧和当前帧中特征点的匹配关系
        vector<vector<MapPoint *> > vvpMapPointMatches;
        vvpMapPointMatches.resize(nKFs);

        //放弃某个关键帧的标记
        vector<bool> vbDiscarded;
        vbDiscarded.resize(nKFs);

        //有效的候选关键帧数目
        int nCandidates = 0;

        // Step 3：遍历所有的候选关键帧，通过词袋进行快速匹配，用匹配结果初始化PnP Solver
        for (int i = 0; i < nKFs; i++) {
            KeyFrame *pKF = vpCandidateKFs[i];
            if (pKF->isBad())
                vbDiscarded[i] = true;
            else {
                // 当前帧和候选关键帧用BoW进行快速匹配，匹配结果记录在vvpMapPointMatches，nmatches表示匹配的数目
                int nmatches = matcher.SearchMatchFrameAndKFByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
                // 如果和当前帧的匹配数小于15,那么只能放弃这个关键帧
                if (nmatches < 15) {
                    vbDiscarded[i] = true;
                    continue;
                } else {
                    // 如果匹配数目够用，用匹配结果初始化EPnPsolver
                    // 为什么用EPnP? 因为计算复杂度低，精度高
                    PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
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
        ORBmatcher matcher2(0.9, true);

        // Step 4: 通过一系列操作,直到找到能够匹配上的关键帧
        // 为什么搞这么复杂？答：是担心误闭环
        while (nCandidates > 0 && !bMatch) {
            //遍历当前所有的候选关键帧
            for (int i = 0; i < nKFs; i++) {
                // 忽略放弃的
                if (vbDiscarded[i])
                    continue;

                //内点标记
                vector<bool> vbInliers;

                //内点数
                int nInliers;

                // 表示RANSAC已经没有更多的迭代次数可用 -- 也就是说数据不够好，RANSAC也已经尽力了。。。
                bool bNoMore;

                // Step 4.1：通过EPnP算法估计姿态，迭代5次
                PnPsolver *pSolver = vpPnPsolvers[i];
                cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                // If Ransac reachs max. iterations discard keyframe
                // bNoMore 为true 表示已经超过了RANSAC最大迭代次数，就放弃当前关键帧
                if (bNoMore) {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If a Camera Pose is computed, optimize
                if (!Tcw.empty()) {
                    //  Step 4.2：如果EPnP 计算出了位姿，对内点进行BA优化
                    Tcw.copyTo(mCurrentFrame.mTcw);

                    // EPnP 里RANSAC后的内点的集合
                    set<MapPoint *> sFound;

                    const int np = vbInliers.size();
                    //遍历所有内点
                    for (int j = 0; j < np; j++) {
                        if (vbInliers[j]) {
                            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        } else
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                    }

                    // 只优化位姿,不优化地图点的坐标，返回的是内点的数量
                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                    // 如果优化之后的内点数目不多，跳过了当前候选关键帧,但是却没有放弃当前帧的重定位
                    if (nGood < 10)
                        continue;

                    // 删除外点对应的地图点
                    for (int io = 0; io < mCurrentFrame.N; io++)
                        if (mCurrentFrame.mvbOutlier[io])
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                    // If few inliers, search by projection in a coarse window and optimize again
                    // Step 4.3：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
                    // 前面的匹配关系是用词袋匹配过程得到的
                    if (nGood < 50) {
                        // 通过投影的方式将关键帧中未匹配的地图点投影到当前帧中, 生成新的匹配
                        int nadditional = matcher2.SearchFrameAndKFByProject(
                                mCurrentFrame,          //当前帧
                                vpCandidateKFs[i],      //关键帧
                                sFound,                 //已经找到的地图点集合，不会用于PNP
                                10,                     //窗口阈值，会乘以金字塔尺度
                                100);                   //匹配的ORB描述子距离应该小于这个阈值

                        // 如果通过投影过程新增了比较多的匹配特征点对
                        if (nadditional + nGood >= 50) {
                            // 根据投影匹配的结果，再次采用3D-2D pnp BA优化位姿
                            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            // Step 4.4：如果BA后内点数还是比较少(<50)但是还不至于太少(>30)，可以挽救一下, 最后垂死挣扎
                            // 重新执行上一步 4.3的过程，只不过使用更小的搜索窗口
                            // 这里的位姿已经使用了更多的点进行了优化,应该更准，所以使用更小的窗口搜索
                            if (nGood > 30 && nGood < 50) {
                                // 用更小窗口、更严格的描述子阈值，重新进行投影搜索匹配
                                sFound.clear();
                                for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                    if (mCurrentFrame.mvpMapPoints[ip])
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                nadditional = matcher2.SearchFrameAndKFByProject(
                                        mCurrentFrame,          //当前帧
                                        vpCandidateKFs[i],      //候选的关键帧
                                        sFound,                 //已经找到的地图点，不会用于PNP
                                        3,                      //新的窗口阈值，会乘以金字塔尺度
                                        64);                    //匹配的ORB描述子距离应该小于这个阈值

                                // Final optimization
                                // 如果成功挽救回来，匹配数目达到要求，最后BA优化一下
                                if (nGood + nadditional >= 50) {
                                    nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                                    //更新地图点
                                    for (int io = 0; io < mCurrentFrame.N; io++)
                                        if (mCurrentFrame.mvbOutlier[io])
                                            mCurrentFrame.mvpMapPoints[io] = NULL;
                                }
                                //如果还是不能够满足就放弃了
                            }
                        }
                    }

                    // If the pose is supported by enough inliers stop ransacs and continue
                    // 如果对于当前的候选关键帧已经有足够的内点(50个)了,那么就认为重定位成功
                    if (nGood >= 50) {
                        bMatch = true;
                        // 只要有一个候选关键帧重定位成功，就退出循环，不考虑其他候选关键帧了
                        break;
                    }
                }
            }//一直运行,知道已经没有足够的关键帧,或者是已经有成功匹配上的关键帧
        }

        // 折腾了这么久还是没有匹配上，重定位失败
        if (!bMatch) {
            return false;
        } else {
            // 如果匹配上了,说明当前帧重定位成功了(当前帧已经有了自己的位姿)
            // 记录成功重定位帧的id，防止短时间多次重定位
            mnLastRelocFrameId = mCurrentFrame.mnId;
            return true;
        }
    }

//整个追踪线程执行复位操作
    void Tracking::Reset() {
        //基本上是挨个请求各个线程终止

        if (mpViewer) {
            mpViewer->RequestStop();
            while (!mpViewer->isStopped())
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        cout << "System Reseting" << endl;

        // Reset Local Mapping
        cout << "Reseting Local Mapper...";
        mpLocalMapper->RequestReset();
        cout << " done" << endl;

        // Reset Loop Closing
        cout << "Reseting Loop Closing...";
        mpLoopClosing->RequestReset();
        cout << " done" << endl;

        // Clear BoW Database
        cout << "Reseting Database...";
        mpKeyFrameDB->clear();
        cout << " done" << endl;

        // Clear Map (this erase MapPoints and KeyFrames)
        mpMap->clear();

        //然后复位各种变量
        KeyFrame::nNextId = 0;
        Frame::nNextId = 0;
        mState = NO_IMAGES_YET;

        if (mpInitializer) {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
        }

        mlRelativeFramePoses.clear();
        mlpReferences.clear();
        mlFrameTimes.clear();
        mlbLost.clear();

        if (mpViewer)
            mpViewer->Release();
    }

//? 目测是根据配置文件中的参数重新改变已经设置在系统中的参数,但是当前文件中没有找到对它的调用
    void Tracking::ChangeCalibration(const string &strSettingPath) {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        //做标记,表示在初始化帧的时候将会是第一个帧,要对它进行一些特殊的初始化操作
        Frame::mbInitialComputations = true;
    }

    void Tracking::InformOnlyTracking(const bool &flag) {
        mbOnlyTracking = flag;
    }

} //namespace ORB_SLAM
