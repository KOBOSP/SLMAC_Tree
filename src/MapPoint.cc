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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM2
{

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

/**
 * @brief Construct a new Map Point:: Map Point object
 * 
 * @param[in] Pos           MapPoint的坐标（世界坐标系）
 * @param[in] pRefKF        关键帧
 * @param[in] pMap          地图
 */
MapPoint::MapPoint(const cv::Mat &Pos,  //地图点的世界坐标
                   KeyFrame *pRefKF,    //生成地图点的关键帧
                   Map* pMap,           //地图点所存在的地图
                   int nObjectID,
                   float fsize):
        mnFirstKFid(pRefKF->mnId),              //第一次观测/生成它的关键帧 id
    mnObserve(0),                                //被观测次数
    mnFrameIdForLocalMp(0),            //放置被重复添加到局部地图点的标记
    mnFrameIdForGpsOrMotion(0),
        mnLastFrameSeen(0),                     //是否决定判断在某个帧视野中的变量
    mnBALocalForKF(0),                      //
    mnFuseCandidateInLM(0),                //
    mnCorrectedByKF(0),                     //
    mnCorrectedReference(0),                //
    mpRefKF(pRefKF),                        //
    mnVisible(1),                           //在帧中的可视次数
    mnFound(1),                             //被找到的次数 和上面的相比要求能够匹配上
    mbBad(false),                           //坏点标记
    mpReplaced(static_cast<MapPoint*>(NULL)), //替换掉当前地图点的点
    mfMinDistance(0),                       //当前地图点在某帧下,可信赖的被找到时其到关键帧光心距离的下界
    mfMaxDistance(0),                       //上界
    mnObjectID(nObjectID),
    mfObjectRadius(fsize),
    mpMap(pMap)                             //从属地图
{
    Pos.copyTo(mWorldPos);
    Pos.copyTo(mGpsPos);
    mObjectsReplaceWeight=1;
    //平均观测方向初始化为0
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);
    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}


//设置地图点在世界坐标系下的坐标
void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    //TODO 为什么这里多了个线程锁
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}
//获取地图点在世界坐标系下的坐标
cv::Mat MapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}
void MapPoint::SetGpsPos(const cv::Mat &Pos)
{
    //TODO 为什么这里多了个线程锁
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mGpsPos);
}
cv::Mat MapPoint::GetGpsPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mGpsPos.clone();
}

int MapPoint::GetObjectId(){
    unique_lock<mutex> lock(mMutexObjId);
    return mnObjectID;
}
int MapPoint::SetObjectId(int nObjId){
    unique_lock<mutex> lock(mMutexObjId);
    mnObjectID=nObjId;
}

//世界坐标系下地图点被多个相机观测的平均观测方向
cv::Mat MapPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}
//获取地图点的参考关键帧
KeyFrame* MapPoint::GetReferenceKeyFrame()
{
     unique_lock<mutex> lock(mMutexFeatures);
     return mpRefKF;
}

/**
 * @brief 给地图点添加观测
 *
 * 记录哪些 KeyFrame 的那个特征点能观测到该 地图点
 * 并增加观测的相机数目nObs，单目+1，双目或者rgbd+2
 * 这个函数是建立关键帧共视关系的核心函数，能共同观测到某些地图点的关键帧是共视关键帧
 * @param pKF KeyFrame
 * @param idx MapPoint在KeyFrame中的索引
 */
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    // mmObservationsKFAndMPidx:观测到该MapPoint的关键帧KF和该MapPoint在KF中的索引
    // 如果已经添加过观测，返回
    if(mmObservationsKFAndMPidx.count(pKF))
        return;
    // 如果没有添加过观测，记录下能观测到该MapPoint的KF和该MapPoint在KF中的索引
    mmObservationsKFAndMPidx[pKF]=idx;
    mnObserve++; // 单目
}


// 删除某个关键帧对当前地图点的观测
void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        // 查找这个要删除的观测,根据单目和双目类型的不同从其中删除当前地图点的被观测次数
        if(mmObservationsKFAndMPidx.count(pKF)){
            mmObservationsKFAndMPidx.erase(pKF);
            mnObserve--;
            // 如果该keyFrame是参考帧，该Frame被删除后重新指定RefFrame
            if(mpRefKF==pKF)
                mpRefKF=mmObservationsKFAndMPidx.begin()->first;
            // If only 2 observations or less, discard point
            // 当观测到该点的相机数目少于2时，丢弃该点
            if(mnObserve <= 1)
                bBad=true;
        }
    }

    if(bBad){
        // 告知可以观测到该MapPoint的Frame，该MapPoint已被删除
        SetBadFlag();
    }
}

// 能够观测到当前地图点的所有关键帧及该地图点在KF中的索引
map<KeyFrame*, size_t> MapPoint::GetObservationsKFAndMPIdx()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mmObservationsKFAndMPidx;
}

// 被观测到的相机数目，单目+1，双目或RGB-D则+2
int MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mnObserve;
}

/**
 * @brief 告知可以观测到该MapPoint的Frame，该MapPoint已被删除
 * 
 */
void MapPoint::SetBadFlag(){
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        // 把mObservations转存到obs，obs和mObservations里存的是指针，赋值过程为浅拷贝
        obs = mmObservationsKFAndMPidx;
        // 把mObservations指向的内存释放，obs作为局部变量之后自动删除
        mmObservationsKFAndMPidx.clear();
    }
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++){
        KeyFrame* pKF = mit->first;
        // 告诉可以观测到该MapPoint的KeyFrame，该MapPoint被删了
        pKF->EraseMapPointMatch(mit->second);
    }
    // 擦除该MapPoint申请的内存
    mpMap->EraseMapPoint(this);
}

MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

/**
 * @brief 替换地图点，更新观测关系
 * 
 * @param[in] pMP       用该地图点来替换当前地图点
 */
void MapPoint::Replace(MapPoint* pMP, bool bUpdataWeight)
{
    // 同一个地图点则跳过
    if(pMP->mnId==this->mnId){
        return;
    }
    //要替换当前地图点,有两个工作:
    // 1. 将当前地图点的观测数据等其他数据都"叠加"到新的地图点上
    // 2. 将观测到当前地图点的关键帧的信息进行更新
    // 清除当前地图点的信息，这一段和SetBadFlag函数相同
    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        swap(obs,mmObservationsKFAndMPidx);
//        obs=mmObservationsKFAndMPidx;
        //清除当前地图点的原有观测
        mmObservationsKFAndMPidx.clear();
        //当前的地图点被删除了
        mbBad=true;
        //暂存当前地图点的可视次数和被找到的次数
        nvisible = mnVisible;
        nfound = mnFound;
        //指明当前地图点已经被指定的地图点替换了
        mpReplaced = pMP;
    }

    // 所有能观测到原地图点的关键帧都要复制到替换的地图点上
    //- 将观测到当前地图的的关键帧的信息进行更新
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++){
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;
        if(!pMP->IsInKeyFrame(pKF)){
            // 该关键帧中没有对"要替换本地图点的地图点"的观测
            pKF->ReplaceMapPointMatch(mit->second, pMP);// 让KeyFrame用pMP替换掉原来的MapPoint
            pMP->AddObservation(pKF,mit->second);// 让MapPoint替换掉对应的KeyFrame
        }
        else{
            // 这个关键帧对当前的地图点和"要替换本地图点的地图点"都具有观测
            // 产生冲突，即pKF中有两个特征点a,b（这两个特征点的描述子是近似相同的），这两个特征点对应两个 MapPoint 为this,pMP
            // 然而在fuse的过程中pMP的观测更多，需要替换this，因此保留b与pMP的联系，去掉a与this的联系
            //说白了,既然是让对方的那个地图点来代替当前的地图点,就是说明对方更好,所以删除这个关键帧对当前帧的观测
            pKF->EraseMapPointMatch(mit->second);
        }
    }

    //- 将当前地图点的观测数据等其他数据都"叠加"到新的地图点上
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    //描述子更新
    if(pMP->mnObjectID<0){
        pMP->ComputeDistinctiveDescriptors();
    }
    if(bUpdataWeight){
        pMP->mObjectsReplaceWeight+=this->mObjectsReplaceWeight;
    }
    //告知地图,删掉我
    this->SetBadFlag();
//    next line is bug, must first set bad then erase
//    mpMap->EraseMapPoint(this);
}

// 没有经过 CullRecentAddedMapPoints 检测的MapPoints, 认为是坏掉的点
bool MapPoint::GetbBad()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}


/**
 * @brief Increase Visible
 *
 * Visible表示：
 * 1. 该MapPoint在某些帧的视野范围内，通过Frame::isInFrustum()函数判断
 * 2. 该MapPoint被这些帧观测到，但并不一定能和这些帧的特征点匹配上
 *    例如：有一个MapPoint（记为M），在某一帧F的视野范围内，
 *    但并不表明该点M可以和F这一帧的某个特征点能匹配上
 * TODO  所以说，found 就是表示匹配上了嘛？
 */
void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

/**
 * @brief Increase Found
 *
 * 能找到该点的帧数+n，n默认为1
 * @see Tracking::TrackWithLocalMap()
 */
void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

// 计算被找到的比例
float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

/**
 * @brief 计算地图点最具代表性的描述子
 *
 * 由于一个地图点会被许多相机观测到，因此在插入关键帧后，需要判断是否更新代表当前点的描述子 
 * 先获得当前点的所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值
 */
void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;
    map<KeyFrame*,size_t> observations;

    // Step 1 获取该地图点所有有效的观测关键帧信息
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mmObservationsKFAndMPidx;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    // Step 2 遍历观测到该地图点的所有关键帧，对应的orb描述子，放到向量vDescriptors中
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++){
        // mit->first取观测到该地图点的关键帧
        // mit->second取该地图点在关键帧中的索引
        KeyFrame* pKF = mit->first;
        if(!pKF->isBad())        
            // 取对应的描述子向量                                               
            vDescriptors.emplace_back(pKF->mDescriptors.row(mit->second));
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    // Step 3 计算这些描述子两两之间的距离
    // N表示为一共多少个描述子
    const size_t N = vDescriptors.size();
	
    // 将Distances表述成一个对称的矩阵
    // float Distances[mnKeyPointNum][mnKeyPointNum];
	std::vector<std::vector<float> > Distances;
	Distances.resize(N, vector<float>(N, 0));
	for (size_t i = 0; i<N; i++){
        // 和自己的距离当然是0
        Distances[i][i]=0;
        // 计算并记录不同描述子距离
        for(size_t j=i+1;j<N;j++){
            int distij = ORBmatcher::ComputeDescriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    // Step 4 选择最有代表性的描述子，它与其他描述子应该具有最小的距离中值
    int BestMedian = INT_MAX;   // 记录最小的中值
    int BestIdx = 0;            // 最小中值对应的索引
    for(size_t i=0;i<N;i++){
        // 第i个描述子到其它所有描述子之间的距离
        // vector<int> vDists(Distances[i],Distances[i]+mnKeyPointNum);
		vector<int> vDists(Distances[i].begin(), Distances[i].end());
		sort(vDists.begin(), vDists.end());
        // 获得中值
        int median = vDists[0.5*(N-1)];
        // 寻找最小的中值
        if(median<BestMedian){
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();       
    }
}

// 获取当前地图点的描述子
cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

//获取当前地图点在某个关键帧的观测中，对应的特征点的ID
int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mmObservationsKFAndMPidx.count(pKF))
        return mmObservationsKFAndMPidx[pKF];
    else
        return -1;
}

/**
 * @brief check MapPoint is in keyframe
 * @param  pKF KeyFrame
 * @return     true if in pKF
 */

/**
 * @brief 检查该地图点是否在关键帧中（有对应的二维特征点）
 * 
 * @param[in] pKF       关键帧
 * @return true         如果能够观测到，返回true
 * @return false        如果观测不到，返回false
 */
bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    // 存在返回true，不存在返回false
    // std::map.count 用法见：http://www.cplusplus.com/reference/map/map/count/
    return (mmObservationsKFAndMPidx.count(pKF));
}

/**
 * @brief 更新地图点的平均观测方向、观测距离范围
 *
 */
void MapPoint::UpdateNormalAndDepth()
{
    // Step 1 获得观测到该地图点的所有关键帧、坐标等信息
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;
        observations=mmObservationsKFAndMPidx; // 获得观测到该地图点的所有关键帧
        pRefKF=mpRefKF;             // 观测到该点的参考关键帧（第一次创建时的关键帧）
        Pos = mWorldPos.clone();    // 地图点在世界坐标系中的位置
    }

    if(observations.empty())
        return;

    // Step 2 计算该地图点的平均观测方向
    // 能观测到该地图点的所有关键帧，对该点的观测方向归一化为单位向量，然后进行求和得到该地图点的朝向
    // 初始值为0向量，累加为归一化向量，最后除以总数n
    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        cv::Mat Owi = pKF->GetCameraCenter();
        // 获得地图点和观测到它关键帧的向量并归一化
        cv::Mat normali = Pos - Owi;
        normal = normal + normali/cv::norm(normali);                       
        n++;
    } 

    cv::Mat PC = Pos - pRefKF->GetCameraCenter();                           // 参考关键帧相机指向地图点的向量（在世界坐标系下的表示）
    const float dist = cv::norm(PC);                                        // 该点到参考关键帧相机的距离
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;        // 观测到该地图点的当前帧的特征点在金字塔的第几层
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];          // 当前金字塔层对应的尺度因子，scale^n，scale=1.2，n为层数
    const int nLevels = pRefKF->mnScaleLevels;                              // 金字塔总层数，默认为8

    {
        unique_lock<mutex> lock1(mMutexPos);
        // 使用方法见PredictScale函数前的注释
        mfMaxDistance = dist*levelScaleFactor;                              // 观测到该点的距离上限
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];    // 观测到该点的距离下限
        mNormalVector = normal/n;                                           // 获得地图点平均的观测方向
    }
}

float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}

// 下图中横线的大小表示不同图层图像上的一个像素表示的真实物理空间中的大小
//              ____
// Nearer      /____\     level:n-1 --> dmin
//            /______\                       d/dmin = 1.2^(n-1-m)
//           /________\   level:m   --> d
//          /__________\                     dmax/d = 1.2^m
// Farther /____________\ level:0   --> dmax
//
//           log(dmax/d)
// m = ceil(------------)
//            log(1.2)
// 这个函数的作用:
// 在进行投影匹配的时候会给定特征点的搜索范围,考虑到处于不同尺度(也就是距离相机远近,位于图像金字塔中不同图层)的特征点受到相机旋转的影响不同,
// 因此会希望距离相机近的点的搜索范围更大一点,距离相机更远的点的搜索范围更小一点,所以要在这里,根据点到关键帧/帧的距离来估计它在当前的关键帧/帧中,
// 会大概处于哪个尺度

/**
 * @brief 预测地图点对应特征点所在的图像金字塔尺度层数
 * 
 * @param[in] currentDist   相机光心距离地图点距离
 * @param[in] pKF           关键帧
 * @return int              预测的金字塔尺度
 */
int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        // mfMaxDistance = ref_dist*levelScaleFactor 为参考帧考虑上尺度后的距离
        // ratio = mfMaxDistance/currentDist = ref_dist/cur_dist
        ratio = mfMaxDistance/currentDist;
    }

    // 取对数
    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;
    return nScale;
}

/**
 * @brief 根据地图点到光心的距离来预测一个类似特征金字塔的尺度
 * 
 * @param[in] currentDist       地图点到光心的距离
 * @param[in] pF                当前帧
 * @return int                  尺度
 */
int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}



} //namespace ORB_SLAM
