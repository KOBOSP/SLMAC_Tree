/**
 * @file Map.cc
 * @author guoqing (1337841346@qq.com)
 * @brief 地图的实现
 * @version 0.1
 * @date 2019-02-26
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


#include "Map.h"

#include<mutex>

namespace ORB_SLAM2
{

//构造函数,地图点中最大关键帧id归0
Map::Map(int nMaxObjectID): mnMaxKeyFrameID(0)
{
    mnMaxObjectID=nMaxObjectID;
    mvnObjectNumByID.resize(nMaxObjectID,0);
    mvnSameObjectIdMap.resize(nMaxObjectID,0);
    for(int i=0;i<nMaxObjectID;i++){
        mvnSameObjectIdMap[i]=i;
    }
}

int Map::GetRootIdxToSameObjectIdMap(int idx){
    if(idx<0){
        return -1;
    }
    unique_lock<mutex> lock(mMutexDsu);
    while(mvnSameObjectIdMap[idx]!=idx){
//        cout<<"+++++++"<<idx<<" "<<mvnSameObjectIdMap[idx];
        idx=mvnSameObjectIdMap[idx];
    }
    return idx;
}

int Map::SetRootIdxToSameObjectIdMap(int idx, int root) {
    unique_lock<mutex> lock(mMutexDsu);
    if (idx < 0 || root < 0 || idx > mnMaxObjectID || root > mnMaxObjectID) {
        return -1;
    }
    mvnSameObjectIdMap[idx]=root;
    return 0;
}

/*
 * @brief Insert KeyFrame in the map
 * @param pKF KeyFrame
 */
//在地图中插入关键帧,同时更新关键帧的最大id
void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->mnId > mnMaxKeyFrameID)
        mnMaxKeyFrameID=pKF->mnId;
}

/*
 * @brief Insert MapPoint in the map
 * @param pMP MapPoint
 */
//向地图中插入地图点
void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

/**
 * @brief 从地图中删除地图点,但是其实这个地图点所占用的内存空间并没有被释放
 * 
 * @param[in] pMP 
 */
void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);
    //下面是作者加入的注释. 实际上只是从std::set中删除了地图点的指针, 原先地图点
    //占用的内存区域并没有得到释
    //bug: where new where delete, because it may be used soon.
    // TODO: This only eraseKFromDB the pointer.
    // Delete the MapPoint
}

/**
 * @brief Erase KeyFrame from the map
 * @param pKF KeyFrame
 */
void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    //是的,根据值来删除地图点
    mspKeyFrames.erase(pKF);
    //bug: where new where delete, because it may be used soon.
    // TODO: This only eraseKFromDB the pointer.
    // Delete the MapPoint
}

/*
 * @brief 设置参考MapPoints，将用于DrawMapPoints函数画图
 * @param vpMPs Local MapPoints
 */
// 设置参考地图点用于绘图显示局部地图点（红色）
void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints.clear();
    mvpReferenceMapPoints = vpMPs;
}

void Map::SetReferenceObjects(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceObjects.clear();
    mvpReferenceObjects = vpMPs;
}


//获取地图中的所有关键帧
vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}
//获取地图中的nNum nearest queue关键帧
void Map::GetNearestKeyFramesByGps(int nNum, vector<KeyFrame*> &vKFNearest, cv::Mat t0)
{
    vector<KeyFrame*> vKFAll=GetAllKeyFrames();
    vector<pair<double,KeyFrame*> >vpdpDistanceAndKF;
    vpdpDistanceAndKF.reserve(vKFAll.size());
    {
        unique_lock<mutex> lock(mMutexMap);
        for (vector<KeyFrame *>::iterator vit = vKFAll.begin(), vend = vKFAll.end(); vit != vend; vit++) {
            vpdpDistanceAndKF.emplace_back(make_pair(cv::norm((*vit)->mTgpsFrame - t0), (*vit)));
        }
    }
    sort(vpdpDistanceAndKF.begin(), vpdpDistanceAndKF.end());
    for (vector<pair<double, KeyFrame *> >::iterator vit = vpdpDistanceAndKF.begin(), vend = vpdpDistanceAndKF.end();
         vit != vend && nNum > 0; vit++, nNum--) {
        vKFNearest.emplace_back((*vit).second);
    }
}

//获取地图中的所有地图点
vector<MapPoint*> Map::GetAllMapPoints(bool bNeedObject)
{
    unique_lock<mutex> lock(mMutexMap);
    vector<MapPoint*> tmp;
    for(set<MapPoint*>::iterator ipMP=mspMapPoints.begin();ipMP!=mspMapPoints.end();ipMP++){
        if((*ipMP)->GetObjectId()>0 && !bNeedObject){
            continue;
        }
        tmp.emplace_back((*ipMP));
    }
    return tmp;
}

void Map::GetAllObjectsInMap(vector<MapPoint*> &vMPs){
    unique_lock<mutex> lock(mMutexMap);
    for(set<MapPoint*>::iterator ipMP=mspMapPoints.begin();ipMP!=mspMapPoints.end();ipMP++){
        if((*ipMP)->GetObjectId()>0){
            vMPs.emplace_back((*ipMP));
        }
    }
}

long unsigned int Map::GetObcjectsNumInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    long unsigned int num=0;
    for(set<MapPoint*>::iterator ipMP=mspMapPoints.begin();ipMP!=mspMapPoints.end();ipMP++){
        if((*ipMP)->GetObjectId()>0){
            num++;
        }
    }
    return num;
}

//获取地图点数目
long unsigned int Map::GetMapPointsNumInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

//获取地图中的关键帧数目
long unsigned int Map::GetKeyFramesNumInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}


//获取参考地图点
vector<MapPoint*> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

vector<MapPoint*> Map::GetReferenceObjects()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceObjects;
}

//获取地图中最大的关键帧id
long unsigned int Map::GetMaxKeyFrameID()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKeyFrameID;
}


//清空地图中的数据
void Map::clear()
{
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++){
        delete *sit;
    }

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++){
        delete *sit;
    }

    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKeyFrameID = 0;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
}

} //namespace ORB_SLAM
