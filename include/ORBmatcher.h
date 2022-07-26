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
#include "sophus/sim3.hpp"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Frame.h"


namespace ORB_SLAM2 {

    class ORBmatcher {
    public:

        /**
         * Constructor
         * @param nnratio  ratio of the best and the second score   最优和次优评分的比例
         * @param checkOri check orientation                        是否检查方向
         */
        ORBmatcher(float nnratio = 0.6, bool checkOri = true);

        /**
         * @brief Computes the Hamming distance between two ORB descriptors 计算地图点和候选投影点的描述子距离
         * @param[in] a     一个描述子
         * @param[in] b     另外一个描述子
         * @return int      描述子的汉明距离
         */
        static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

        //MonocularInitialization()
        int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched,
                                    std::vector<int> &vnMatches12, int windowSize = 10);

        // TrackWithMotionModel()||invzc,uv,bForward,bestDist,GetObsTimes,er,mbCheckOrientation
        int SearchFrameAndFrameByProject(Frame &CurrentFrame, const Frame &LastFrame, const float nThProjRad, const bool bMono);
        // TrackReferenceKeyFrame() and Relocalization()||BestDist,secondDist,mbCheckOrientation
        int SearchMatchFrameAndKFByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint *> &vpMapPointMatches);
        // Relocalization()||uv,Dist3D,bestDist,mbCheckOrientation
        int SearchFrameAndKFByProject(Frame &CurrentFrame, KeyFrame *pKF, const std::set<MapPoint *> &sAlreadyFound, const float nThProjRad, const int ORBdist);
        //MatchLocalMPsToCurFrame()||bFarPoints,bestDist,secondDist,GetObsTimes,er
        int SearchReplaceFrameAndMPsByProject(Frame &F, const std::vector<MapPoint *> &vpMapPoints, const float nThProjRad = 3);

        //CreateNewMapPoints()
        int SearchKFAndKFByTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                         std::vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo);

        //FuseMapPointsInNeighbors() full replace
        int SearchReplaceKFAndMPsByProjectInLocalMap(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float nThProjRad = 3.0);
        //SearchAndFuse() part replace
        int SearchReplaceKFAndMPsByProjectInGlobalMap(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapPoint *> &vpCandidMPs, float nThProjRad, vector<MapPoint *> &vpReplacePoint);
        //ComputeSim3()
        int SearchMatchKFAndMPsByProject(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapPoint *> &vpPoints, std::vector<MapPoint *> &vpMatchedMPs, int nThProjRad);
        // ComputeSim3()
        int SearchMatchKFAndKFByProject(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float nThProjRad);
        // DetectCommonRegionsByBoWSearchAndProjectVerify()
        int SearchMatchKFAndKFByBoW(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12);


    public:

        // 要用到的一些阈值
        static const int TH_LOW;            ///< 判断描述子距离时比较低的那个阈值,主要用于基于词袋模型加速的匹配过程，可能是感觉使用词袋模型的时候对匹配的效果要更加严格一些
        static const int TH_HIGH;           ///< 判断描述子距离时比较高的那个阈值,用于计算投影后能够匹配上的特征点的数目；如果匹配的函数中没有提供阈值的话，默认就使用这个阈值
        static const int HISTO_LENGTH;      ///< 判断特征点旋转用直方图的长度


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
        bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12,
                                   const KeyFrame *pKF);

        /**
         * @brief 根据观察的视角来计算匹配的时的搜索窗口大小
         * @param[in] viewCos   视角的余弦值
         * @return float        搜索窗口的大小
         */
        float RadiusByViewingCos(const float &viewCos);

        /**
         * @brief 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
         *
         * @param[in] histo         匹配特征点对旋转方向差直方图
         * @param[in] L             直方图尺寸
         * @param[in & out] ind1          bin值第一大对应的索引
         * @param[in & out] ind2          bin值第二大对应的索引
         * @param[in & out] ind3          bin值第三大对应的索引
         */
        void ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);

        bool GetBestKPIdxInKFToMP(KeyFrame *pKF, Sophus::SE3f &Tcw, MapPoint *pMP, float th,
                                  int &bestDist, int &bestIdx, bool bProjError, float fScale=1.0);

        float mfNNratio;            ///< 最优评分和次优评分的比例
        bool mbCheckOrientation;    ///< 是否检查特征点的方向
    };

}// namespace ORB_SLAM

#endif // ORBMATCHER_H
