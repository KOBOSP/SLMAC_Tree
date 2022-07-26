/**
 * @file LocalMapping.cc
 * @author guoqing (1337841346@qq.com)
 * @brief 局部建图线程
 * @version 0.1
 * @date 2019-04-29
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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

// 构造函数
LocalMapping::LocalMapping(Map *pMap, const float bMonocular, float fCullKFRedundantMPRate):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true), mfCullKFRedundantMPRate(fCullKFRedundantMPRate)
{
    /*
     * mbStopRequested：    外部线程调用，为true，表示外部线程请求停止 local mapping
     * mbStopped：          为true表示可以并终止localmapping 线程
     * mbNotStop：          true，表示不要停止 localmapping 线程，因为要插入关键帧了。需要和 mbStopped 结合使用
     * mbAcceptKeyFrames：  true，允许接受关键帧。tracking 和local mapping 之间的关键帧调度
     * mbAbortBA：          是否流产BA优化的标志位
     * mbFinishRequested：  请求终止当前线程的标志。注意只是请求，不一定终止。终止要看 mbFinished
     * mbResetRequested：   请求当前线程复位的标志。true，表示一直请求复位，但复位还未完成；表示复位完成为false
     * mbFinished：         判断最终LocalMapping::Run() 是否完成的标志。
     */
}

// 设置回环检测线程句柄
void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

// 线程主函数
void LocalMapping::Run()
{
    // 标记状态，表示当前run函数正在运行，尚未结束
    mbFinished = false;
    // 主循环

    while(true){
        // Tracking will see that Local Mapping is busy
        // Step 1 告诉Tracking，LocalMapping正处于繁忙状态，请不要给我发送关键帧打扰我
        // LocalMapping线程处理的关键帧都是Tracking线程发来的
        SetAcceptKeyFrames(false);
        // Check if there are keyframes in the queue
        // 等待处理的关键帧列表不为空
        if(HaveNewKeyFrameInQueue()){
            // BoW conversion and insertion in Map
            // Step 2 处理列表中的关键帧，包括计算BoW、更新观测、描述子、共视图，插入到地图等
            ProcessNewKeyFrame();
            // Check recent MapPoints
            // Step 3 根据地图点的观测情况剔除质量不好的地图点
            CullRecentAddedMapPoints();
            // GetTriangulMapPoint new MapPoints
            // Step 4 当前关键帧与相邻关键帧通过三角化产生新的地图点，使得跟踪更稳
            CreateNewMapPointsAndObjectsByNerborKFs();
            // 已经处理完队列中的最后的一个关键帧
            if(!HaveNewKeyFrameInQueue()){
                // Find more matches in neighbor keyframes and fuse point duplications
                //  Step 5 检查并融合当前关键帧与相邻关键帧帧（两级相邻）中重复的地图点
                FuseMapPointsAndObjectsByNeighbors();
            }
            // 终止BA的标志
            mbAbortBA = false;
            // 已经处理完队列中的最后的一个关键帧，并且闭环检测没有请求停止LocalMapping
            if(!HaveNewKeyFrameInQueue() && !stopRequested()){
                // Local BA
                // Step 6 当局部地图中的关键帧大于2个的时候进行局部地图的BA
                if(mpMap->GetKeyFramesNumInMap() > 3){
                    // 注意这里的第二个参数是按地址传递的,当这里的 mbAbortBA 状态发生变化时，能够及时执行/停止BA
                    Optimizer::OptimizeLocalMapPoint(mpCurrentKeyFrame, &mbAbortBA, mpMap);
                }
                // Check redundant local Keyframes
                // Step 7 检测并剔除当前帧相邻的关键帧中冗余的关键帧
                // 冗余的判定：该关键帧的90%的地图点可以被其它关键帧观测到
                CullRedundantKeyFrame();
            }
            // Step 8 将当前帧加入到闭环检测队列中
            // 注意这里的关键帧被设置成为了bad的情况,这个需要注意
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())     // 当要终止当前线程的时候
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish()){
                // 如果还没有结束利索,那么等
                // usleep(3000);
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
            // 然后确定终止了就跳出这个线程的主循环
            if(CheckFinish())
                break;
        }
        // 查看是否有复位线程的请求
        ResetIfRequested();
        // Tracking will see that Local Mapping is not busy
        SetAcceptKeyFrames(true);

        // 如果当前线程已经结束了就跳出主循环
        if(CheckFinish())
            break;

        //usleep(3000);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));

    }
    // 设置线程已经终止
    SetFinish();
}

// 插入关键帧,由外部（Tracking）线程调用;这里只是插入到列表中,等待线程主函数对其进行处理
void LocalMapping::InsertKeyFrameInQueue(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    // 将关键帧插入到列表中
    mlNewKeyFrames.emplace_back(pKF);
    mbAbortBA=true;
}

// 查看列表中是否有等待被插入的关键帧,
bool LocalMapping::HaveNewKeyFrameInQueue()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

/**
 * @brief 处理列表中的关键帧，包括计算BoW、更新观测、描述子、共视图，插入到地图等
 * 
 */
void LocalMapping::ProcessNewKeyFrame()
{
    // Step 1：从缓冲队列中取出一帧关键帧
    // 该关键帧队列是Tracking线程向LocalMapping中插入的关键帧组成
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        // 取出列表中最前面的关键帧，作为当前要处理的关键帧
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        // 取出最前面的关键帧后，在原来的列表里删掉该关键帧
        mlNewKeyFrames.pop_front();
    }
    // Compute Bags of Words structures
    // Step 2：计算该关键帧特征点的词袋向量
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    // Step 3：当前处理关键帧中有效的地图点，更新normal，描述子等信息
    // TrackLocalMap中和当前帧新匹配上的地图点和当前关键帧进行关联绑定
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetAllMapPointVectorInKF();
    // 对当前处理的这个关键帧中的所有的地图点展开遍历
    for(size_t i=0; i<vpMapPointMatches.size(); i++){
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP){
            if(!pMP->GetbBad()){
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame)){
                    // 如果地图点不是来自当前帧的观测（比如来自局部地图点TrackWithLocalMap），为当前地图点添加观测
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    // 获得该点的平均观测方向和观测距离范围
                    pMP->UpdateNormalAndDepth();
                    // 更新地图点的最佳描述子
                    pMP->ComputeDistinctiveDescriptors();
                }
                if(pMP->GetObjectId()<0){
                    mlpRecentAddedMapAndObjPoints.emplace_back(pMP);
                }
            }
        }
    }

    // UpdateImgKPMPState links in the Covisibility Graph
    // Step 4：更新关键帧间的连接关系（共视图）
    mpCurrentKeyFrame->UpdateConnectedKeyFrameAndWeights();

    // Insert Keyframe in Map
    // Step 5：将该关键帧插入到地图中
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

/**
 * @brief 检查新增地图点，根据地图点的观测情况剔除质量不好的新增的地图点
 * mlpRecentAddedMapAndObjPoints：存储新增的地图点，这里是要删除其中不靠谱的
 */
void LocalMapping::CullRecentAddedMapPoints()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapAndObjPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    // Step 1：根据相机类型设置不同的观测阈值
    int nThObs = 2;
	// Step 2：遍历检查新添加的地图点
    while(lit != mlpRecentAddedMapAndObjPoints.end()){
        MapPoint* pMP = *lit;
        if(!pMP){
            continue;
        }
        if(pMP->GetbBad()){
            // Step 2.1：已经是坏点的地图点仅从队列中删除
            lit = mlpRecentAddedMapAndObjPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f){
            // Step 2.2：跟踪到该地图点的帧数相比预计可观测到该地图点的帧数的比例小于25%，从地图中删除
            // (mnFound/mnVisible） < 25%
            // mnFound ：地图点被多少帧（包括普通帧）看到，次数越多越好
            // mnVisible：地图点应该被看到的次数
            // (mnFound/mnVisible）：对于大FOV镜头这个比例会高，对于窄FOV镜头这个比例会低
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapAndObjPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>= 3 && pMP->GetObservations() <= nThObs){
            // Step 2.3：从该点建立开始，到现在已经过了不小于2个关键帧
            // 但是观测到该点的相机数却不超过阈值cnThObs，从地图中删除
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapAndObjPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3){
            // Step 2.4：从建立该点开始，已经过了3个关键帧而没有被剔除，则认为是质量高的点
            // 因此没有SetBadFlag()，仅从队列中删除
            lit = mlpRecentAddedMapAndObjPoints.erase(lit);
        }
        else{
            lit++;
        }
    }
}




/**
 * @brief 用当前关键帧与相邻关键帧通过三角化产生新的地图点，使得跟踪更稳
 * 
 */
void LocalMapping::CreateNewMapPointsAndObjectsByNerborKFs()
{
    // Retrieve neighbor keyframes in covisibility graph
    // nn表示搜索最佳共视关键帧的数目
    // 不同传感器下要求不一样,单目的时候需要有更多的具有较好共视关系的关键帧来建立地图
    int nn = 10;
    mvpNeighKFs.clear();
    // Step 1：在当前关键帧的共视关键帧中找到共视程度最高的nn帧相邻关键帧
    mvpNeighKFs = mpCurrentKeyFrame->GetCovisibilityForefrontKeyFrames(nn);
    mpMap->GetNearestKeyFramesByGps(nn, mvpNeighKFs, mpCurrentKeyFrame->mTgpsFrame);
    // 特征点匹配配置 最佳距离 < 0.6*次佳距离，比较苛刻了。不检查旋转
    ORBmatcher matcher(0.6,false); 

    // 取出当前帧从世界坐标系到相机坐标系的变换矩阵
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));

    // 得到当前关键帧（左目）光心在世界坐标系中的坐标、内参
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    // 用于后面的点深度的验证;这里的1.5是经验值
    // mfScaleFactor = 1.2
    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    // Search matches with epipolar restriction and triangulate
    // Step 2：遍历相邻关键帧，搜索匹配并用极线约束剔除误匹配，最终三角化
    for(size_t i=0; i<mvpNeighKFs.size(); i++){
        KeyFrame* pKF2 = mvpNeighKFs[i];
        // Check first that baseline is not too short
        // 相邻的关键帧光心在世界坐标系中的坐标
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        // 基线向量，两个关键帧间的相机位移
        cv::Mat vBaseline = Ow2-Ow1;
        // 基线长度
        const float baseline = cv::norm(vBaseline);

        // Step 3：判断相机运动的基线是不是足够长
        // 单目相机情况
        // 相邻关键帧的场景深度中值
        const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
        // 基线与景深的比例
        const float ratioBaselineDepth = baseline/medianDepthKF2;
        // 如果比例特别小，基线太短恢复3D点不准，那么跳过当前邻接的关键帧，不生成3D点
        if(ratioBaselineDepth<0.16)//5m/30m=0.1666
            continue;

        // Compute Fundamental Matrix
        // Step 4：根据两个关键帧的位姿计算它们之间的基础矩阵
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        // Step 5：通过词袋对两关键帧的未匹配的特征点快速匹配，用极线约束抑制离群点，生成新的匹配点对
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchNewKFMatchPointByKFBoWAndF12(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // GetTriangulMapPoint each match
        // Step 6：对每对匹配通过三角化生成3D点,和 Triangulate函数差不多
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++){
            // Step 6.1：取出匹配特征点
            // 当前匹配对在当前关键帧中的索引
            const int &idx1 = vMatchedIndices[ikp].first;
            // 当前匹配对在邻接关键帧中的索引
            const int &idx2 = vMatchedIndices[ikp].second;
            // 当前匹配在当前关键帧中的特征点
            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            // 当前匹配在邻接关键帧中的特征点
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            if(kp1.class_id!=kp2.class_id){
                continue;
            }

            // Check parallax between rays
            // Step 6.2：利用匹配点反投影得到视差角
            // 特征点反投影,其实得到的是在各自相机坐标系下的一个非归一化的方向向量,和这个点的反投影射线重合
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            // 由相机坐标系转到世界坐标系(得到的是那条反投影射线的一个同向向量在世界坐标系下的表示,还是只能够表示方向)，得到视差角余弦值
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            // 匹配点射线夹角余弦值
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            // Step 6.4：三角化恢复3D点
            cv::Mat x3D;
            // cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998)表明视差角正常,0.9998 对应1°， 0.9848=10, 0.9659=cos(15d) 0.9397=cos(20d)
            // cosParallaxRays < cosParallaxStereo 表明匹配点对夹角大于双目本身观察三维点夹角
            // 匹配点对夹角大，用三角法恢复3D点
            // 参考：https://github.com/raulmur/ORB_SLAM2/issues/345
            if(cosParallaxRays>0 && (kp1.class_id<0 && cosParallaxRays<0.9848) or (kp1.class_id>0 && cosParallaxRays<0.9659)){//tan-1(5/30)=9.46
                //(kp1.class_id<0 && cosParallaxRays>0 && cosParallaxRays<0.9960) || ((kp1.class_id>0 &&
                // Linear Triangulation Method
                // 见Initializer.cc的 GetTriangulMapPoint 函数,实现是一样的,顶多就是把投影矩阵换成了变换矩阵
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();
                // 归一化之前的检查
                if(x3D.at<float>(3)==0)
                    continue;
                // 归一化成为齐次坐标,然后提取前面三个维度作为欧式坐标
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
            }
            else{
                continue; //No stereo and very low parallax, 放弃
            }

            // 为方便后续计算，转换成为了行向量
            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            // Step 6.5：检测生成的3D点是否在相机前方,不在的话就放弃这个点
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0){
                continue;
            }

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0){
                continue;
            }

            //Check reprojection error in first keyframe
            // Step 6.6：计算3D点在当前关键帧下的重投影误差
            const float sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;
            float u1 = fx1*x1*invz1+cx1;
            float v1 = fy1*y1*invz1+cy1;
            float errX1 = u1 - kp1.pt.x;
            float errY1 = v1 - kp1.pt.y;
            // 假设测量有一个像素的偏差，2自由度卡方检验阈值是5.991
            if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                continue;

            //Check reprojection error in second keyframe
            // 计算3D点在另一个关键帧下的重投影误差，操作同上
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            float u2 = fx2*x2*invz2+cx2;
            float v2 = fy2*y2*invz2+cy2;
            float errX2 = u2 - kp2.pt.x;
            float errY2 = v2 - kp2.pt.y;
            if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                continue;

            //Check scale consistency
            // Step 6.7：检查尺度连续性

            // 世界坐标系下，3D点与相机间的向量，方向由相机指向3D点
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            // ratioDist是不考虑金字塔尺度下的距离比例
            const float ratioDist = dist2/dist1;
            // 金字塔尺度因子的比例
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];
            // 距离的比例和图像金字塔的比例不应该差太多，否则就跳过
            if(kp1.class_id<0&&(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)){
                continue;
            }
            if(kp1.class_id>0&&(dist1>2*medianDepthKF2||dist2>2*medianDepthKF2||dist1<medianDepthKF2/2||dist2<medianDepthKF2/2)){
                continue;
            }

            // Triangulation is succesfull
            // Step 6.8：三角化生成3D点成功，构造成MapPoint
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap,kp1.class_id,kp1.size);
            // Step 6.9：为该MapPoint添加属性：
            // a.观测到该MapPoint的关键帧
            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);
            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);
            // b.该MapPoint的描述子
            pMP->ComputeDistinctiveDescriptors();
            // c.该MapPoint的平均观测方向和深度范围
            pMP->UpdateNormalAndDepth();
            mpMap->AddMapPoint(pMP);
            // Step 6.10：将新产生的点放入检测队列
            // 这些MapPoints都会经过MapPointCulling函数的检验
            if(pMP->GetObjectId()<0){
                mlpRecentAddedMapAndObjPoints.emplace_back(pMP);
            }
        }
    }
}

/**
 * @brief 检查并融合当前关键帧与相邻帧（两级相邻）重复的地图点
 * 
 */
void LocalMapping::FuseMapPointsAndObjectsByNeighbors()
{
    // Retrieve neighbor keyframes
    // Step 1：获得当前关键帧在共视图中权重排名前nn的邻接关键帧
    // 开始之前先定义几个概念
    // 当前关键帧的邻接关键帧，称为一级相邻关键帧，也就是邻居
    // 与一级相邻关键帧相邻的关键帧，称为二级相邻关键帧，也就是邻居的邻居

    // Step 2：存储一级相邻关键帧及其二级相邻关键帧
    vector<KeyFrame*> vpTargetKFs;
    // 开始对所有候选的一级关键帧展开遍历：
    for(vector<KeyFrame*>::const_iterator vit=mvpNeighKFs.begin(), vend=mvpNeighKFs.end(); vit!=vend; vit++){
        KeyFrame* pKFi = *vit;
        // 没有和当前帧进行过融合的操作
        if(pKFi->isBad() || pKFi->mnFuseCandidateInLM == mpCurrentKeyFrame->mnId)
            continue;
        // 加入一级相邻关键帧    
        vpTargetKFs.emplace_back(pKFi);
        // 标记已经加入
        pKFi->mnFuseCandidateInLM = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        // 以一级相邻关键帧的共视关系最好的5个相邻关键帧 作为二级相邻关键帧
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetCovisibilityForefrontKeyFrames(5);
        // 遍历得到的二级相邻关键帧
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(),
                vend2=vpSecondNeighKFs.end();vit2!=vend2; vit2++){
            KeyFrame* pKFi2 = *vit2;
            // 当然这个二级相邻关键帧要求没有和当前关键帧发生融合,并且这个二级相邻关键帧也不是当前关键帧
            if(pKFi2->isBad() || pKFi2->mnFuseCandidateInLM == mpCurrentKeyFrame->mnId || pKFi2->mnId == mpCurrentKeyFrame->mnId)
                continue;
            // 存入二级相邻关键帧    
            vpTargetKFs.emplace_back(pKFi2);
            pKFi2->mnFuseCandidateInLM = mpCurrentKeyFrame->mnId;
        }
    }

    // Search matches by projection from current KF in target KFs
    // 使用默认参数, 最优和次优比例0.6,匹配时检查特征点的旋转
    ORBmatcher matcher(0.6,true,mpMap);
    vector<MapPoint*> vpCandidateObjPoints;//this same id mappoint project into current KF
    vpCandidateObjPoints.reserve(vpTargetKFs.size() * 50);
    // Step 3：将当前帧的地图点分别投影到两级相邻关键帧，寻找匹配点对应的地图点进行融合，称为正向投影融合
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetAllMapPointVectorInKF(true);
    for(vector<MapPoint*>::iterator vitMP=vpMapPointMatches.begin(), vendMP=vpMapPointMatches.end(); vitMP!=vendMP; vitMP++){
        if(!(*vitMP))
            continue;
        // 如果地图点是坏点，或者已经加进集合vpFuseCandidates，跳过
        if((*vitMP)->GetbBad()){
            continue;
        }
        if((*vitMP)->GetObjectId()>0){
            vpCandidateObjPoints.emplace_back((*vitMP));
        }
    }
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++){
        KeyFrame* pKFi = *vit;
//        cout<<"-2 vpTargetKFs.size(): "<<vpTargetKFs.size()<<" pKFi->mnId "<<pKFi->mnId<<endl;
        // 将地图点投影到关键帧中进行匹配和融合；融合策略如下
        // 1.如果地图点能匹配关键帧的特征点，并且该点有对应的地图点，那么选择观测数目多的替换两个地图点
        // 2.如果地图点能匹配关键帧的特征点，并且该点没有对应的地图点，那么为该点添加该投影地图点
        // 注意这个时候对地图点融合的操作是立即生效的
        matcher.FuseRedundantMapPointAndObjectByProjectInLocalMap(pKFi, vpMapPointMatches,2);
    }
    // Search matches by projection from target KFs in current KF
    // Step 4：将两级相邻关键帧地图点分别投影到当前关键帧，寻找匹配点对应的地图点进行融合，称为反向投影融合
    // 用于进行存储要融合的一级邻接和二级邻接关键帧所有MapPoints的集合
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());
    
    //  Step 4.1：遍历每一个一级邻接和二级邻接关键帧，收集他们的地图点存储到 vpFuseCandidates
    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++){
        KeyFrame* pKFi = *vitKF;
        vector<MapPoint*> vpMapPointsKFi = pKFi->GetAllMapPointVectorInKF(true);
        // 遍历当前一级邻接和二级邻接关键帧中所有的MapPoints,找出需要进行融合的并且加入到集合中
        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++){
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            // 如果地图点是坏点，或者已经加进集合vpFuseCandidates，跳过
            if(pMP->GetbBad() || pMP->mnFuseCandidateInLM == mpCurrentKeyFrame->mnId)
                continue;
            if(pMP->GetObjectId()>0){
                vpCandidateObjPoints.emplace_back(pMP);
            }
            // 加入集合，并标记已经加入
            pMP->mnFuseCandidateInLM = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.emplace_back(pMP);
        }
    }
//    cout<<"-1 vpFuseCandidates.size(): "<<vpFuseCandidates.size()<<endl;
    // Step 4.2：进行地图点投影融合,和正向融合操作是完全相同的
    // 不同的是正向操作是"每个关键帧和当前关键帧的地图点进行融合",而这里的是"当前关键帧和所有邻接关键帧的地图点进行融合"
    matcher.FuseRedundantMapPointAndObjectByProjectInLocalMap(mpCurrentKeyFrame, vpFuseCandidates,2);
    mpMap->SetReferenceObjects(vpCandidateObjPoints);

    // UpdateImgKPMPState points
    // Step 5：更新当前帧地图点的描述子、深度、平均观测方向等属性
    vpMapPointMatches = mpCurrentKeyFrame->GetAllMapPointVectorInKF(true);
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++){
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP){
            if(!pMP->GetbBad()){
                // 在所有找到pMP的关键帧中，获得最佳的描述子
                pMP->ComputeDistinctiveDescriptors();
                // 更新平均观测方向和观测距离
                pMP->UpdateNormalAndDepth();
            }
        }
    }
    // UpdateImgKPMPState connections in covisibility graph
    // Step 6：更新当前帧与其它帧的共视连接关系
    mpCurrentKeyFrame->UpdateConnectedKeyFrameAndWeights();
}


// 根据两关键帧的姿态计算两个关键帧之间的基本矩阵
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    // 先构造两帧之间的R12,t12
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    // 得到 t12 的反对称矩阵
    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Essential Matrix: t12叉乘R12
    // Fundamental Matrix: inv(K1)*E*inv(K2)
    return K1.t().inv()*t12x*R12*K2.inv();
}

// 外部线程调用,请求停止当前线程的工作; 其实是回环检测线程调用,来避免在进行全局优化的过程中局部建图线程添加新的关键帧
void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

// 检查是否要把当前的局部建图线程停止工作,运行的时候要检查是否有终止请求,如果有就执行. 由run函数调用
bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    // 如果当前线程还没有准备停止,但是已经有终止请求了,那么就准备停止当前线程
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

// 检查mbStopped是否为true，为true表示可以并终止localmapping 线程
bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

// 求外部线程调用，为true，表示外部线程请求停止 local mapping
bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

// 释放当前还在缓冲区中的关键帧指针
void LocalMapping::ReleaseNewKFinList()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

// 查看当前是否允许接受关键帧
bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

// 设置"允许接受关键帧"的状态标志
void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

// 设置 mbnotStop标志的状态
bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    //已经处于!flag的状态了
    // 就是我希望线程先不要停止,但是经过检查这个时候线程已经停止了...
    if(flag && mbStopped)
        //设置失败
        return false;

    //设置为要设置的状态
    mbNotStop = flag;
    //设置成功
    return true;
}



/**
 * @brief 检测当前关键帧在共视图中的关键帧，根据地图点在共视图中的冗余程度剔除该共视关键帧
 * 冗余关键帧的判定：90%以上的地图点能被其他关键帧（至少3个）观测到
 */
void LocalMapping::CullRedundantKeyFrame()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points

    // 该函数里变量层层深入，这里列一下：
    // mpCurrentKeyFrame：当前关键帧，本程序就是判断它是否需要删除
    // pKF： mpCurrentKeyFrame的某一个共视关键帧
    // vpMapPoints：pKF对应的所有地图点
    // pMP：vpMapPoints中的某个地图点
    // observations：所有能观测到pMP的关键帧
    // pKFi：observations中的某个关键帧
    // scaleLeveli：pKFi的金字塔尺度
    // scaleLevel：pKF的金字塔尺度

    // Step 1：根据共视图提取当前关键帧的所有共视关键帧
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    // 对所有的共视关键帧进行遍历
    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        // 第1个关键帧不能删除，跳过
        if(pKF->mnId == 0)
            continue;
        // Step 2：提取每个共视关键帧的地图点
        const vector<MapPoint*> vpMapPoints = pKF->GetAllMapPointVectorInKF();

        // 观测次数阈值，默认为3
        const int thObs=3;
        // 记录冗余观测点的数目
        int nRedundantObservations=0;
        int nMPs=0;

        // Step 3：遍历该共视关键帧的所有地图点，其中能被其它至少3个关键帧观测到的地图点为冗余地图点
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++){
            MapPoint* pMP = vpMapPoints[i];
            if(pMP){
                if(!pMP->GetbBad()){
                    nMPs++;
                    // pMP->GetObservations() 是观测到该地图点的相机总数目（单目1，双目2）
                    if(pMP->GetObservations() > thObs){
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        // Observation存储的是可以看到该地图点的所有关键帧的集合
                        const map<KeyFrame*, size_t> observations = pMP->GetObservationsKFAndMPIdx();

                        int nObs=0;
                        // 遍历观测到该地图点的关键帧
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(),
                                mend=observations.end(); mit!=mend; mit++){
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            // 尺度约束：为什么pKF 尺度+1 要大于等于 pKFi 尺度？
                            // 回答：因为同样或更低金字塔层级的地图点更准确
                            if(scaleLeveli<=scaleLevel+1){
                                nObs++;
                                // 已经找到3个满足条件的关键帧，就停止不找了
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        // 地图点至少被3个关键帧观测到，就记录为冗余点，更新冗余点计数数目
                        if(nObs>=thObs){
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }
        // Step 4：如果该关键帧90%以上的有效地图点被判断为冗余的，则认为该关键帧是冗余的，需要删除该关键帧
        if(nRedundantObservations>mfCullKFRedundantMPRate*nMPs){
            pKF->SetBadFlag();
        }
    }
}

// 计算三维向量v的反对称矩阵
cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<
            0,              -v.at<float>(2),     v.at<float>(1),
            v.at<float>(2),               0,    -v.at<float>(0),
           -v.at<float>(1),  v.at<float>(0),                 0);
}

// 请求当前线程复位,由外部线程调用,堵塞的
void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    // 一直等到局部建图线程响应之后才可以退出
    while(1){{
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        //usleep(3000);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));

    }
}

// 检查是否有复位线程的请求
void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    // 执行复位操作:清空关键帧缓冲区,清空待cull的地图点缓冲
    
    if(mbResetRequested){
        mlNewKeyFrames.clear();
        mlpRecentAddedMapAndObjPoints.clear();
        // 恢复为false表示复位过程完成
        mbResetRequested=false;
    }
}

// 请求终止当前线程
void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

// 检查是否已经有外部线程请求终止当前线程
bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

// 设置当前线程已经真正地结束了
void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    // 线程已经被结束
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;     //既然已经都结束了,那么当前线程也已经停止工作了
}

// 当前线程的run函数是否已经终止
bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
