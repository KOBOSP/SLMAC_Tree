/**
 * @file ORBmatcher.h
 * @author guoqing (1337841346@qq.com)
 * @brief 处理数据关联问题
 * @version 0.1
 * @date 2019-04-26
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


#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "MapPoint.h"
#include "KeyFrame.h"
#include "Frame.h"
#include "LoopClosing.h"

namespace ORB_SLAM2
{

class ORBmatcher
{    
public:

    /**
     * Constructor
     * @param nnratio  ratio of the best and the second score   最优和次优评分的比例
     * @param checkOri check orientation                        是否检查方向
     */
    ORBmatcher(float nnratio=0.6, bool checkOri=true, Map* pMap=NULL);

    /**
     * @brief Computes the Hamming distance between two ORB descriptors 计算地图点和候选投影点的描述子距离
     * @param[in] a     一个描述子
     * @param[in] b     另外一个描述子
     * @return int      描述子的汉明距离
     */
    static int ComputeDescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    /**
     * @brief 通过投影地图点到当前帧，对Local MapPoint进行跟踪
     * 步骤
     * Step 1 遍历有效的局部地图点
     * Step 2 设定搜索搜索窗口的大小。取决于视角, 若当前视角和平均视角夹角较小时, r取一个较小的值
     * Step 3 通过投影点以及搜索窗口和预测的尺度进行搜索, 找出搜索半径内的候选匹配点索引
     * Step 4 寻找候选匹配点中的最佳和次佳匹配点
     * Step 5 筛选最佳匹配点
     * @param[in] F                         当前帧
     * @param[in] vpMapPoints               局部地图点，来自局部关键帧
     * @param[in] th                        搜索范围
     * @return int                          成功匹配的数目
     */
    int SearchFMatchPointByProjectMapPoint(Frame &CurrentFrame, const vector<MapPoint*> &vpMapPoints, const float th, bool bNeedObject);


    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    /**
     * @brief 通过投影的方式将关键帧中的地图点投影到当前帧中,并且进行匹配
     * 
     * @param[in] CurrentFrame      当前帧
     * @param[in] pKF               关键帧
     * @param[in] sAlreadyFound     已经寻找得到的地图点
     * @param[in] th                //窗口大小的阈值
     * @param[in] ORBdist           //描述子最小距离阈值
     * @return int                  //匹配到的点的数目
     */
    int SearchFMatchPointByProjectKeyFrame(Frame &CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist);


    // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
    // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
    // Used in Relocalisation and Loop Detection
    /*
    * @brief 通过词袋，对关键帧的特征点进行跟踪
    * 步骤
    * Step 1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
    * Step 2：遍历KF中属于该node的特征点
    * Step 3：遍历F中属于该node的特征点，寻找最佳匹配点
    * Step 4：根据阈值 和 角度投票剔除误匹配
    * Step 5：根据方向剔除误匹配的点
    * @param  pKF               KeyFrame
    * @param  F                 Current Frame
    * @param  vpMapPointMatches F中MapPoints对应的匹配，NULL表示未匹配
    * @return                   成功匹配的数量
    */
    int SearchFMatchPointByKFBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);

    // Matching for the Map Initialization (only used in the monocular case)
    /**
     * @brief 单目初始化中用于参考帧和当前帧的特征点匹配
     * 步骤
     * Step 1 构建旋转直方图
     * Step 2 在半径窗口内搜索当前帧F2中所有的候选匹配特征点 
     * Step 3 遍历搜索搜索窗口中的所有潜在的匹配候选点，找到最优的和次优的
     * Step 4 对最优次优结果进行检查，满足阈值、最优/次优比例，删除重复匹配
     * Step 5 计算匹配点旋转角度差所在的直方图
     * Step 6 筛除旋转直方图中“非主流”部分
     * Step 7 将最后通过筛选的匹配好的特征点保存
     * @param[in] F1                        初始化参考帧                  
     * @param[in] F2                        当前帧
     * @param[in & out] vbPrevMatched       本来存储的是参考帧的所有特征点坐标，该函数更新为匹配好的当前帧的特征点坐标
     * @param[in & out] vnMatches12         保存参考帧F1中特征点是否匹配上，index保存是F1对应特征点索引，值保存的是匹配好的F2特征点索引
     * @param[in] windowSize                搜索窗口
     * @return int                          返回成功匹配的特征点数目
     */
    int SearchMatchKPInInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize=10);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    /**
     * @brief 利用基本矩阵F12，在两个关键帧之间未匹配的特征点中产生新的3d点
     * @param pKF1          关键帧1
     * @param pKF2          关键帧2
     * @param F12           基础矩阵
     * @param vMatchedPairs 存储匹配特征点对，特征点用其在关键帧中的索引表示，下标是关键帧1的特征点id，存储的是关键帧2的特征点id
     * @param bOnlyStereo   在双目和rgbd情况下，是否要求特征点在右图存在匹配
     * @return              成功匹配的数量
     */
    int SearchKFMatchObjectByKFTargetId(KeyFrame *pKF1, KeyFrame *pKF2, vector<pair<size_t, size_t> > &vMatchedPairs);
    int SearchNewKFMatchPointByKFBoWAndF12(KeyFrame *pKF1, KeyFrame* pKF2, cv::Mat F12,
                                           std::vector<pair<size_t, size_t> > &vMatchedPairs);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    /**
     * @brief 将地图点投影到关键帧中进行匹配和融合;并且地图点的替换可以在这个函数中进行
     * @param[in] pKF           关键帧
     * @param[in] vpMapPoints   地图点
     * @param[in] th            搜索窗口的阈值
     * @return int 
     */
    int FuseRedundantMapPointAndObjectByProjectInLocalMap(KeyFrame* pKF, const vector<MapPoint *> &vpMapPoints, const float th= 3.0);
    int FuseRedundantDifferIdObjectInLocalMap(KeyFrame* pKF, const vector<MapPoint *> &vpMapPoints, vector<int> &vnSameObjectIdMap, set<int> &sLinkedObjectID, const int nMaxObjectID, bool notRecursion = false, const float th = 1.5);

public:

    // 要用到的一些阈值
    static const int TH_LOW;            ///< 判断描述子距离时比较低的那个阈值,主要用于基于词袋模型加速的匹配过程，可能是感觉使用词袋模型的时候对匹配的效果要更加严格一些
    static const int TH_HIGH;           ///< 判断描述子距离时比较高的那个阈值,用于计算投影后能够匹配上的特征点的数目；如果匹配的函数中没有提供阈值的话，默认就使用这个阈值
    static const int HISTO_LENGTH;      ///< 判断特征点旋转用直方图的长度
    vector<vector<int> >rotHist;


protected:

    /**
     * @brief 检查极线距离
     * @param[in] kp1   特征点1
     * @param[in] kp2   特征点2
     * @param[in] F12   两帧之间的基础矩阵
     * @param[in] pKF   //? 关键帧?
     * @return true 
     * @return false 
     */
    bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF);
    void CheckFrameRotationConsistency(Frame &CurrentFrame, int &nmatches);


    /**
     * @brief 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
     * 
     * @param[in] rotHist         匹配特征点对旋转方向差直方图
     * @param[in] L             直方图尺寸
     * @param[in & out] ind1          bin值第一大对应的索引
     * @param[in & out] ind2          bin值第二大对应的索引
     * @param[in & out] ind3          bin值第三大对应的索引
     */
    void ComputeThreeMaxOrienta(int &ind1, int &ind2, int &ind3);
    Map* mpMap;
    float mfNNratio;            ///< 最优评分和次优评分的比例
    bool mbCheckOrientation;    ///< 是否检查特征点的方向
};

}// namespace ORB_SLAM

#endif // ORBMATCHER_H
