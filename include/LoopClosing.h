/**
 * @file LoopClosing.h
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

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "KeyFrame.h"
#include "LocalMapping.h"
#include "Map.h"
#include "ORBVocabulary.h"
#include "Tracking.h"

#include "KeyFrameDatabase.h"

#include <thread>
#include <mutex>
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{

class Tracking;
class LocalMapping;
class KeyFrameDatabase;

/// 回环检测线程
class LoopClosing
{
public:
    /// 存储关键帧对象和位姿的键值对,这里是map的完整构造函数
    typedef map<KeyFrame*,                  //键
                g2o::Sim3,                  //值
                std::less<KeyFrame*>,       //排序算法
                Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3> > // 指定分配器,和内存空间开辟有关. 为了能够使用Eigen库中的SSE和AVX指令集加速,需要将传统STL容器中的数据进行对齐处理
                > KeyFrameAndPose;

public:

    /**
     * @brief 构造函数
     * @param[in] pMap          地图指针
     * @param[in] pDB           词袋数据库
     * @param[in] pVoc          词典
     * @param[in] bFixScale     表示sim3中的尺度是否要计算,对于双目和RGBD情况尺度是固定的,s=1,bFixScale=true;而单目下尺度是不确定的,此时bFixScale=false,sim
     * 3中的s需要被计算
     */
    LoopClosing(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc, int nMaxObjectID, const string &strSettingPath, const bool bFixScale);
    /** @brief 设置追踪线程的句柄
     *  @param[in] pTracker 追踪线程的句柄  */
    void SetTracker(Tracking* pTracker);
    /** @brief 设置局部建图线程的句柄
     * @param[in] pLocalMapper   */
    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function
    /** @brief 回环检测线程主函数 */
    void Run();

    /** @brief 将某个关键帧加入到回环检测的过程中,由局部建图线程调用
     *  @param[in] pKF   */
    void InsertKeyFrame(KeyFrame *pKF);

    /** @brief 由外部线程调用,请求复位当前线程.在回环检测复位完成之前,该函数将一直保持堵塞状态 */
    void RequestReset();

    // 在回环纠正的时候调用,查看当前是否已经有一个全局优化的线程在进行
    bool isRunningGBA(){
        unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }

    /** @brief 由外部线程调用,请求终止当前线程 */
    void RequestFinish();

    /** @brief 由外部线程调用,判断当前回环检测线程是否已经正确终止了  */
    bool isFinished();


    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
    void FuseSameIdObjectInGlobalMap();
    void UpdataGPSToVOSim3InLoopClosing();
    void LinkObjectIdByProjectGlobalMapToAllKF();
    void LinkObjectIdBySearchGpsLocation();

    /** @brief 查看列表中是否有等待被插入的关键帧
     *  @return true 如果有
     *  @return false 没有  */
    bool CheckNewKeyFrames();


    /** @brief  当前线程调用,检查是否有外部线程请求复位当前线程,如果有的话就复位回环检测线程 */
    void ResetIfRequested();
    /// 是否有复位当前线程的请求
    bool mbResetRequested;
    /// 和复位当前线程相关的互斥量
    std::mutex mMutexReset;

    /** @brief 当前线程调用,查看是否有外部线程请求当前线程  */
    bool CheckFinish();
    /** @brief 有当前线程调用,执行完成该函数之后线程主函数退出,线程销毁 */
    void SetFinish();
    /// 是否有终止当前线程的请求
    bool mbFinishRequested;
    /// 当前线程是否已经停止工作
    bool mbFinished;
    bool mbSystemWithoutLoopClose;
    /// 和当前线程终止状态操作有关的互斥量
    std::mutex mMutexFinish;

    /// (全局)地图的指针
    Map* mpMap;
    /// 追踪线程句柄
    Tracking* mpTracker;


    /// 一个队列, 其中存储了参与到回环检测的关键帧 (当然这些关键帧也有可能因为各种原因被设置成为bad,这样虽然这个关键帧还是存储在这里但是实际上已经不再实质性地参与到回环检测的过程中去了)
    std::list<KeyFrame*> mlpLoopKeyFrameQueue;

    /// 操作参与到回环检测队列中的关键帧时,使用的互斥量
    std::mutex mMutexLoopQueue;

    // Loop detector parameters
    /// 连续性阈值,构造函数中将其设置成为了3
    int mnCovisibilityConsistencyTh;
    int mnSingleMatchKeyPoint;
    int mnTotalMatchKeyPoint;
    int mnSingleMatchObject;
    int mnTotalMatchObject;
    int mnFuseMPByObjectSceneTh;
    int mnDetectLoopByFusedMPNumTh;

    // Variables related to Global Bundle Adjustment
    /// 全局BA线程是否在进行
    bool mbRunningGBA;
    /// 在对和全局线程标志量有关的操作的时候使用的互斥量
    std::mutex mMutexGBA;
    int mnMaxObjectID;
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H
