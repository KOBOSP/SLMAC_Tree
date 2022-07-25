/**
 * @file LoopClosing.cc
 * @author guoqing (1337841346@qq.com)
 * @brief 回环检测线程
 * @version 0.1
 * @date 2019-05-05
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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM2
{

// 构造函数
LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, int nMaxObjectID, const string &strSettingPath, const bool bFixScale):
        mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap), mbRunningGBA(false), mnMaxObjectID(nMaxObjectID)
{
    // 连续性阈值
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    int temp = fSettings["LoopClose.CovisibilityConsistencyTh"];
    mnCovisibilityConsistencyTh = temp;
    temp = fSettings["LoopClose.SystemWithoutLoopClose"];
    mbSystemWithoutLoopClose=bool(temp);
    mnSingleMatchKeyPoint = fSettings["LoopClose.SingleMatchKeyPoint"];
    mnTotalMatchKeyPoint = fSettings["LoopClose.TotalMatchKeyPoint"];
    mnSingleMatchObject = fSettings["LoopClose.SingleMatchObject"];
    mnTotalMatchObject = fSettings["LoopClose.TotalMatchObject"];
    mnFuseMPByObjectSceneTh = fSettings["LoopClose.FuseMPByObjectSceneThreshold"];
    mnDetectLoopByFusedMPNumTh = fSettings["LoopClose.DetectLoopByFusedMPNumThreshold"];
    cout << "- LoopClose.CovisibilityConsistencyTh: " << mnCovisibilityConsistencyTh << endl;
    cout << "- LoopClose.SystemWithoutLoopClose: " << mbSystemWithoutLoopClose << endl;
    cout << "- LoopClose.SingleMatchKeyPoint " << mnSingleMatchKeyPoint << endl;
    cout << "- LoopClose.TotalMatchKeyPoint: " << mnTotalMatchKeyPoint << endl;
    cout << "- LoopClose.SingleMatchObject " << mnSingleMatchObject << endl;
    cout << "- LoopClose.TotalMatchObject: " << mnTotalMatchObject << endl;
    cout << "- LoopClose.FuseMPByObjectSceneTh: " << mnFuseMPByObjectSceneTh << endl;
    cout << "- LoopClose.DetectLoopByFusedMPNumTh: " << mnDetectLoopByFusedMPNumTh << endl;
    cout << "- Track.MaxObjectID: " << nMaxObjectID << endl;

}

// 设置追踪线程句柄
void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}


// 回环线程主函数
void LoopClosing::Run()
{
    mbFinished =false;

    // 线程主循环
    while(true){
        if(!mbSystemWithoutLoopClose){        // Check if there are keyframes in the queue
            if (CheckNewKeyFrames()) {//have new KF
                UpdataGPSToVOSim3InLoopClosing();
                LinkObjectIdBySearchGpsLocation();
            }
            LinkObjectIdByProjectGlobalMapToAllKF();
            FuseSameIdObjectInGlobalMap();
        }
        // 查看是否有外部线程请求复位当前线程
        ResetIfRequested();
        // 查看外部线程是否有终止当前线程的请求,如果有的话就跳出这个线程的主函数的主循环
        if(CheckFinish())
            break;
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // 运行到这里说明有外部线程请求终止当前线程,在这个函数中执行终止当前线程的一些操作
    SetFinish();
}

/**
 * @brief use LocalKeyFrames to refresh the sim3 matrix from gps to VO
 *
 */

void LoopClosing::UpdataGPSToVOSim3InLoopClosing(){
    if(mpTracker->mState!=Tracking::OK){
        return;
    }
    Sim3Solver* pSim3Solvers = new Sim3Solver();
    vector<KeyFrame*> vAllKFInMap = mpMap->GetAllKeyFrames();
    std::vector<cv::Mat> vP1s;
    std::vector<cv::Mat> vP2s;
    vector<bool> vbInliers;
    int nInliers;
    for(vector<KeyFrame*>::iterator itKF=vAllKFInMap.begin();itKF!=vAllKFInMap.end();itKF++){
        if(!(*itKF)){
            continue;
        }
        if((*itKF)->isBad()){
            continue;
        }
        cv::Mat temp;
        (*itKF)->mTgpsFrame.copyTo(temp);
        vP1s.emplace_back(temp.clone());
        (*itKF)->GetTranslation().copyTo(temp);
        vP2s.emplace_back(temp.clone());
    }
    cv::Mat Sim3VoGps = pSim3Solvers->KFsiterate(vP1s, vP2s, 100, vbInliers, nInliers);//P1:Gps, P2Vo
    if(!Sim3VoGps.empty()&&nInliers>mpTracker->mnSim3Inliers){
        Sim3VoGps.copyTo(mpTracker->mSim3VoGps);
        mpTracker->mnSim3Inliers=nInliers;
    }
}

void LoopClosing::LinkObjectIdBySearchGpsLocation() {
    if(!mpTracker){
        return;
    }
    if(mpTracker->mSim3VoGps.empty()){
        return;
    }

    double dCentmpDis;
    int root1,root2;
    vector<MapPoint*> ObjectsInGlobalMap;
    mpMap->GetAllObjectsInMap(ObjectsInGlobalMap);


    cv::Mat tObjVo,tObjGps, Sim3GV, sRGV, tGV;
    Sim3GV = mpTracker->mSim3VoGps.inv();
    for(vector<MapPoint*>::iterator vit=ObjectsInGlobalMap.begin(); vit != ObjectsInGlobalMap.end(); vit++) {
        if (!(*vit)) {
            continue;
        }
        if ((*vit)->GetbBad()) {
            continue;
        }
        MapPoint* pMPObj = *vit;
        tObjVo=pMPObj->GetWorldPos();
        Sim3GV.rowRange(0, 3).colRange(0, 3).copyTo(sRGV);
        Sim3GV.rowRange(0, 3).col(3).copyTo(tGV);
        tObjGps= sRGV * tObjVo + tGV;
        pMPObj->SetGpsPos(tObjGps);
    }

    for(vector<MapPoint*>::iterator vit1=ObjectsInGlobalMap.begin(); vit1 != ObjectsInGlobalMap.end(); vit1++) {
        if (!(*vit1)) {
            continue;
        }
        if ((*vit1)->GetbBad()) {
            continue;
        }
        for (vector<MapPoint *>::iterator vit2 = vit1+1; vit2 != ObjectsInGlobalMap.end(); vit2++) {
            if (!(*vit2)) {
                continue;
            }
            if ((*vit2)->GetbBad()) {
                continue;
            }
            root1=mpMap->GetRootIdxToSameObjectIdMap((*vit1)->GetObjectId());
            root2=mpMap->GetRootIdxToSameObjectIdMap((*vit2)->GetObjectId());
            if(root1==root2){
                continue;
            }
            dCentmpDis = cv::norm((*vit1)->GetGpsPos(), (*vit2)->GetGpsPos(), cv::NORM_L2);
            if(dCentmpDis<1.0) {//less than 1m
                mpMap->SetRootIdxToSameObjectIdMap(root1,root2);
            }
        }
    }
}


void LoopClosing::LinkObjectIdByProjectGlobalMapToAllKF(){
    ORBmatcher matcher(0.6, false, mpMap);
    vector<MapPoint*> ObjectsInGlobalMap;
    mpMap->GetAllObjectsInMap(ObjectsInGlobalMap);
    for(vector<MapPoint*>::iterator vit=ObjectsInGlobalMap.begin(); vit != ObjectsInGlobalMap.end(); vit++) {
        if(!(*vit)){
            vit=ObjectsInGlobalMap.erase(vit);
            vit--;
            continue;
        }
        if((*vit)->GetbBad()){
            vit=ObjectsInGlobalMap.erase(vit);
            vit--;
            continue;
        }
        MapPoint* pMPObj = *vit;
        vector<KeyFrame*> vKFNearest;
        mpMap->GetNearestKeyFramesByGps(20, vKFNearest, pMPObj->GetWorldPos());
        for(vector<KeyFrame*>::iterator vit=vKFNearest.begin(), vend=vKFNearest.end(); vit!=vend; vit++){
            KeyFrame* pKFi = *vit;
            if(!pKFi){
                continue;
            }
            if(pKFi->isBad()){
                continue;
            }
            vector<MapPoint *> vpMPOne;
            vpMPOne.emplace_back(pMPObj);
            // 将地图点投影到关键帧中进行匹配和融合；融合策略如下
            // 1.如果地图点能匹配关键帧的特征点，并且该点有对应的地图点，那么选择观测数目多的替换两个地图点
            // 2.如果地图点能匹配关键帧的特征点，并且该点没有对应的地图点，那么为该点添加该投影地图点
            // 注意这个时候对地图点融合的操作是立即生效的
            matcher.FuseRedundantMapPointAndObjectByProjectInLocalMap(pKFi, vpMPOne, 2);
        }
    }
}



void LoopClosing::FuseSameIdObjectInGlobalMap(){
    int nMaxObjectID=-1,nMinObjectID=mnMaxObjectID+1;
    vector<MapPoint*> ObjectsInGlobalMap;
    mpMap->GetAllObjectsInMap(ObjectsInGlobalMap);
    vector<vector<MapPoint*> > vvpSameIDObject;
    vvpSameIDObject.resize(nMinObjectID);

    for(vector<MapPoint*>::iterator vit=ObjectsInGlobalMap.begin(); vit != ObjectsInGlobalMap.end(); vit++) {
        if(!(*vit))
            continue;
        if((*vit)->GetbBad())
            continue;
        MapPoint* pMPObj = (*vit);
        int tmpId=mpMap->GetRootIdxToSameObjectIdMap(pMPObj->GetObjectId());
        pMPObj->SetObjectId(tmpId);
        vvpSameIDObject[tmpId].emplace_back(pMPObj);
        nMaxObjectID=(tmpId > nMaxObjectID)?tmpId : nMaxObjectID;
        nMinObjectID=(tmpId < nMinObjectID)?tmpId : nMinObjectID;
    }

    int nMP3DNum, nMPId, nMPOk;
    cv::Mat MP3DTot, MP3DMean;
    double dCenDisTot, dCenDisMean, dCentmpDis, dCenMinDis;

    for(int i=nMinObjectID;i<=nMaxObjectID;i++){
        if(vvpSameIDObject[i].size()<2){
            continue;
        }
        vector<double> vdCenDis(100,-1.0);
        vector<bool> vbMPisBad(100,false);
        bool bHaveOuterInVec = true;
        MapPoint *pMPMinDis;

        while(bHaveOuterInVec){
            bHaveOuterInVec = false;
            MP3DTot = vvpSameIDObject[i][0]->GetWorldPos()-vvpSameIDObject[i][0]->GetWorldPos();
            nMP3DNum=0;
            nMPId=0;

            //get mean pos in vvpSameIDObject[i]
            for(vector<MapPoint*>::iterator vit=vvpSameIDObject[i].begin(); vit != vvpSameIDObject[i].end(); vit++,nMPId++) {
                if(vbMPisBad[nMPId]){
                    continue;
                }
                MapPoint *pMPtmp=(*vit);
                if(!pMPtmp){
                    vbMPisBad[nMPId]=true;
                    continue;
                }
                if(pMPtmp->GetbBad()){
                    vbMPisBad[nMPId]=true;
                    continue;
                }
                MP3DTot += pMPtmp->GetWorldPos() * pMPtmp->mObjectsReplaceWeight;
                nMP3DNum += pMPtmp->mObjectsReplaceWeight;
            }
            MP3DMean= MP3DTot / nMP3DNum;
            dCenDisTot=0;
            dCenMinDis=999999;
            nMPOk=0;
            nMPId=0;

            //get MinDis MapPoint and Distance, besides total distance
            for(vector<MapPoint*>::iterator vit=vvpSameIDObject[i].begin(); vit != vvpSameIDObject[i].end(); vit++,nMPId++) {
                if(vbMPisBad[nMPId]){
                    continue;
                }
                MapPoint *pMPtmp=(*vit);
                cv::Mat pMpPos=pMPtmp->GetWorldPos();
                dCentmpDis = cv::norm(pMpPos, MP3DMean, cv::NORM_L2);
                vdCenDis[nMPId]=dCentmpDis;
                dCenDisTot+=dCentmpDis;
                nMPOk++;
                if(dCentmpDis<dCenMinDis){
                    dCenMinDis=dCentmpDis;
                    pMPMinDis=pMPtmp;
                }
            }
            dCenDisMean= dCenDisTot / nMPOk;
            nMPId=0;

            for(vector<MapPoint*>::iterator vit=vvpSameIDObject[i].begin(); vit != vvpSameIDObject[i].end(); vit++,nMPId++) {
                if(vbMPisBad[nMPId]){
                    continue;
                }
                MapPoint *pMPtmp=(*vit);
                if(dCenDisMean*5<vdCenDis[nMPId]){
                    bHaveOuterInVec= true;
                    vbMPisBad[nMPId] = true;
                    pMPtmp->Replace(pMPMinDis,true);
                }
            }
        }
        pMPMinDis->SetWorldPos(MP3DMean);
        nMPId=0;
        for(vector<MapPoint*>::iterator vit=vvpSameIDObject[i].begin(); vit != vvpSameIDObject[i].end(); vit++,nMPId++) {
            if(vbMPisBad[nMPId]==true){
                continue;
            }
            MapPoint *pMPtmp=(*vit);
            if(pMPtmp->mnId==pMPMinDis->mnId){
                continue;
            }
            pMPtmp->Replace(pMPMinDis, true);
        }
    }
}

// 将某个关键帧加入到回环检测的过程中,由局部建图线程调用
void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    // 注意：这里第0个关键帧不能够参与到回环检测的过程中,因为第0关键帧定义了整个地图的世界坐标系
    if(pKF->mnId != 0)
        mlpLoopKeyFrameQueue.emplace_back(pKF);
}

/*
 * 查看列表中是否有等待被插入的关键帧
 * @return 如果存在，返回true
 */
bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

// 由外部线程调用,请求复位当前线程
void LoopClosing::RequestReset(){
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }
    // 堵塞,直到回环检测线程复位完成
    while(1){
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
		//usleep(5000);
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// 当前线程调用,检查是否有外部线程请求复位当前线程,如果有的话就复位回环检测线程
void LoopClosing::ResetIfRequested(){
    unique_lock<mutex> lock(mMutexReset);
    // 如果有来自于外部的线程的复位请求,那么就复位当前线程
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();   // 清空参与和进行回环检测的关键帧队列
        mbResetRequested=false;         // 复位请求标志复位
    }
}

// 由外部线程调用,请求终止当前线程
void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

// 当前线程调用,查看是否有外部线程请求当前线程
bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

// 有当前线程调用,执行完成该函数之后线程主函数退出,线程销毁
void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

// 由外部线程调用,判断当前回环检测线程是否已经正确终止了
bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
