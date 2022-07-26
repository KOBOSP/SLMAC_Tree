/**
 * @file ORBmatcher.cc
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

#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "DBoW2/FeatureVector.h"

#include<stdint.h>

using namespace std;

namespace ORB_SLAM2 {

// 要用到的一些阈值
    const int ORBmatcher::TH_HIGH = 100;
    const int ORBmatcher::TH_LOW = 50;
    const int ORBmatcher::HISTO_LENGTH = 30;

// 构造函数,参数默认值为0.6,true
    ORBmatcher::ORBmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri) {

    }

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
 * @param[in] nThProjRad                        搜索范围
 * @return int                          成功匹配的数目
 */
    int ORBmatcher::SearchReplaceFrameAndMPsByProject(Frame &F, const vector<MapPoint *> &vpMapPoints,
                                                      const float nThProjRad) {
        int nmatches = 0;

        // 如果 nThProjRad！=1 (RGBD 相机或者刚刚进行过重定位), 需要扩大范围搜索
        const bool bFactor = nThProjRad != 1.0;

        // Step 1 遍历有效的局部地图点
        for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++) {
            MapPoint *pMP = vpMapPoints[iMP];
            if (pMP->isBad())
                continue;
            // 判断该点是否要投影
            if (!pMP->mbTrackInView)
                continue;

            // 通过距离预测的金字塔层数，该层数相对于当前的帧
            const int &nPredictedLevel = pMP->mnTrackScaleLevel;

            // The size of the window will depend on the viewing direction
            // Step 2 设定搜索搜索窗口的大小。取决于视角, 若当前视角和平均视角夹角较小时, r取一个较小的值
            float r = RadiusByViewingCos(pMP->mTrackViewCos);

            // 如果需要扩大范围搜索，则乘以阈值th
            if (bFactor)
                r *= nThProjRad;

            // Step 3 通过投影点以及搜索窗口和预测的尺度进行搜索, 找出搜索半径内的候选匹配点索引
            const vector<size_t> vIndices =
                    F.GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY,      // 该地图点投影到一帧上的坐标
                                        r * F.mvScaleFactors[nPredictedLevel],    // 认为搜索窗口的大小和该特征点被追踪到时所处的尺度也有关系
                                        nPredictedLevel - 1, nPredictedLevel);     // 搜索的图层范围

            // 没找到候选的,就放弃对当前点的匹配
            if (vIndices.empty())
                continue;

            const cv::Mat MPdescriptor = pMP->GetDescriptor();

            // 最优的次优的描述子距离和index
            int bestDist = 256;
            int bestLevel = -1;
            int bestDist2 = 256;
            int bestLevel2 = -1;
            int bestIdx = -1;

            // Get best and second matches with near keypoints
            // Step 4 寻找候选匹配点中的最佳和次佳匹配点
            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++) {
                const size_t idx = *vit;

                // 如果Frame中的该兴趣点已经有对应的MapPoint了,则退出该次循环
                if (F.mvpMapPoints[idx])
                    if (F.mvpMapPoints[idx]->Observations() > 0)
                        continue;


                const cv::Mat &d = F.mDescriptors.row(idx);

                // 计算地图点和候选投影点的描述子距离
                const int dist = DescriptorDistance(MPdescriptor, d);

                // 寻找描述子距离最小和次小的特征点和索引
                if (dist < bestDist) {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestLevel2 = bestLevel;
                    bestLevel = F.mvKeysUn[idx].octave;
                    bestIdx = idx;
                } else if (dist < bestDist2) {
                    bestLevel2 = F.mvKeysUn[idx].octave;
                    bestDist2 = dist;
                }
            }

            // Apply ratio to second match (only if best and second are in the same scale level)
            // Step 5 筛选最佳匹配点
            // 最佳匹配距离还需要满足在设定阈值内
            if (bestDist <= TH_HIGH) {
                // 条件1：bestLevel==bestLevel2 表示 最佳和次佳在同一金字塔层级
                // 条件2：bestDist>mfNNratio*bestDist2 表示最佳和次佳距离不满足阈值比例。理论来说 bestDist/bestDist2 越小越好
                if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
                    continue;

                //保存结果: 为Frame中的特征点增加对应的MapPoint
                F.mvpMapPoints[bestIdx] = pMP;
                nmatches++;
            }
        }

        return nmatches;
    }

// 根据观察的视角来计算匹配的时的搜索窗口大小
    float ORBmatcher::RadiusByViewingCos(const float &viewCos) {
        // 当视角相差小于3.6°，对应cos(3.6°)=0.998，搜索范围是2.5，否则是4
        if (viewCos > 0.998)
            return 2.5;
        else
            return 4.0;
    }

/**
 * @brief 用基础矩阵检查极线距离是否符合要求
 *
 * @param[in] kp1               KF1中特征点
 * @param[in] kp2               KF2中特征点
 * @param[in] F12               从KF2到KF1的基础矩阵
 * @param[in] pKF2              关键帧KF2
 * @return true
 * @return false
 */
    bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12,
                                           const KeyFrame *pKF2) {
        // Epipolar line in second image l2 = x1'F12 = [a b c]
        // Step 1 求出kp1在pKF2上对应的极线
        const float a = kp1.pt.x * F12.at<float>(0, 0) + kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
        const float b = kp1.pt.x * F12.at<float>(0, 1) + kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
        const float c = kp1.pt.x * F12.at<float>(0, 2) + kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);

        // Step 2 计算kp2特征点到极线l2的距离
        // 极线l2：ax + by + c = 0
        // (u,v)到l2的距离为： |au+bv+c| / sqrt(a^2+b^2)

        const float num = a * kp2.pt.x + b * kp2.pt.y + c;

        const float den = a * a + b * b;

        // 距离无穷大
        if (den == 0)
            return false;

        // 距离的平方
        const float dsqr = num * num / den;

        // Step 3 判断误差是否满足条件。尺度越大，误差范围应该越大。
        // 金字塔最底层一个像素就占一个像素，在倒数第二层，一个像素等于最底层1.2个像素（假设金字塔尺度为1.2）
        // 3.84 是自由度为1时，服从高斯分布的一个平方项（也就是这里的误差）小于一个像素，这件事发生概率超过95%时的概率 （卡方分布）
        return dsqr < 3.84 * pKF2->mvLevelSigma2[kp2.octave];
    }


    bool
    ORBmatcher::GetBestKPIdxInKFToMP(KeyFrame *pKF, Sophus::SE3f &Tcw, MapPoint *pMP, float nThProjRad, int &bestDist,
                                     int &bestIdx, bool bProjError, float fScale) {

        Sophus::Sim3f soScw = Sophus::Sim3f(Sophus::RxSO3f(fScale, Tcw.rotationMatrix()),
                                            Tcw.translation());
        Eigen::Vector3f Ow = Tcw.inverse().translation();
        Eigen::Vector3f p3Dw = Converter::toVector3f(pMP->GetWorldPos());
        Eigen::Vector3f p3Dc = soScw * p3Dw;

        if (p3Dc(2) < 0.0f)
            return false;

        const Eigen::Vector2f uv = pKF->ProjectMPToKP(p3Dc);
        if (!pKF->IsInImage(uv(0), uv(1)))
            return false;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw - Ow;
        const float Dist3D = PO.norm();
        // Depth must be inside the scale pyramid of the image
        if (Dist3D < minDistance || Dist3D > maxDistance)
            return false;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = Converter::toVector3f(pMP->GetNormal());
        if (PO.dot(Pn) < 0.5 * Dist3D)
            return false;
        int nPredictedLevel = pMP->PredictScale(Dist3D, pKF);
        const float radius = nThProjRad * pKF->mvScaleFactors[nPredictedLevel];
        // Step 5 在投影点附近搜索窗口内找到候选匹配点的索引
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0), uv(1), radius);
        if (vIndices.empty())
            return false;


        const cv::Mat dMP = pMP->GetDescriptor();
        bestDist = 256;
        bestIdx = -1;
        for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end();
             vit != vend; vit++) {
            const size_t Idx = *vit;
            const cv::KeyPoint &kp = pKF->mvKeysUn[Idx];
            const int &kpLevel = kp.octave;
            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;
            if (bProjError) {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = uv(0) - kpx;
                const float ey = uv(1) - kpy;
                const float e2 = ex * ex + ey * ey;
                // 自由度为2的，卡方检验阈值5.99（假设测量有一个像素的偏差）
                if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 5.99)
                    continue;
            }
            const cv::Mat &dKF = pKF->mDescriptors.row(Idx);
            const int Dist = DescriptorDistance(dMP, dKF);
            // 和投影点的描述子距离最小
            if (Dist < bestDist) {
                bestDist = Dist;
                bestIdx = Idx;
            }
        }
        if (bestIdx != -1) {
            return true;
        } else {
            return false;
        }
    }


    /*
 * @brief 通过词袋，对关键帧的特征点进行跟踪
 * 步骤
 * Step 1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
 * Step 2：遍历KF中属于该node的特征点
 * Step 3：遍历F中属于该node的特征点，寻找最佳匹配点
 * Step 4：根据阈值 和 角度投票剔除误匹配
 * Step 5：根据方向剔除误匹配的点
 * @param  pKF               关键帧
 * @param  F                 当前普通帧
 * @param  vpMapPointMatches F中地图点对应的匹配，NULL表示未匹配
 * @return                   成功匹配的数量
 */
    int ORBmatcher::SearchMatchFrameAndKFByBoW(KeyFrame *pKF, Frame &F, vector<MapPoint *> &vpMapPointMatches) {
        // 获取该关键帧的地图点
        const vector<MapPoint *> vpMapPointsKF = pKF->GetMapPointMatches();

        // 和普通帧F特征点的索引一致
        vpMapPointMatches = vector<MapPoint *>(F.N, static_cast<MapPoint *>(NULL));

        // 取出关键帧的词袋特征向量
        const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

        int nmatches = 0;

        // 特征点角度旋转差统计用的直方图
        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = HISTO_LENGTH / 360.0f;

        // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
        // 将属于同一节点(特定层)的ORB特征进行匹配
        DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
        DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
        DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
        DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

        while (KFit != KFend && Fit != Fend) {
            // Step 1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
            // first 元素就是node id，遍历
            if (KFit->first == Fit->first) {
                // second 是该node内存储的feature index
                const vector<unsigned int> vIndicesKF = KFit->second;
                const vector<unsigned int> vIndicesF = Fit->second;

                // Step 2：遍历KF中属于该node的特征点
                for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++) {
                    // 关键帧该节点中特征点的索引
                    const unsigned int realIdxKF = vIndicesKF[iKF];

                    // 取出KF中该特征对应的地图点
                    MapPoint *pMP = vpMapPointsKF[realIdxKF];

                    if (!pMP)
                        continue;

                    if (pMP->isBad())
                        continue;

                    const cv::Mat &dKF = pKF->mDescriptors.row(realIdxKF); // 取出KF中该特征对应的描述子

                    int bestDist1 = 256; // 最好的距离（最小距离）
                    int bestIdxF = -1;
                    int bestDist2 = 256; // 次好距离（倒数第二小距离）

                    // Step 3：遍历F中属于该node的特征点，寻找最佳匹配点
                    for (size_t iF = 0; iF < vIndicesF.size(); iF++) {
                        // 和上面for循环重名了,这里的realIdxF是指普通帧该节点中特征点的索引
                        const unsigned int realIdxF = vIndicesF[iF];

                        // 如果地图点存在，说明这个点已经被匹配过了，不再匹配，加快速度
                        if (vpMapPointMatches[realIdxF])
                            continue;

                        const cv::Mat &dF = F.mDescriptors.row(realIdxF); // 取出F中该特征对应的描述子
                        // 计算描述子的距离
                        const int dist = DescriptorDistance(dKF, dF);

                        // 遍历，记录最佳距离、最佳距离对应的索引、次佳距离等
                        // 如果 dist < bestDist1 < bestDist2，更新bestDist1 bestDist2
                        if (dist < bestDist1) {
                            bestDist2 = bestDist1;
                            bestDist1 = dist;
                            bestIdxF = realIdxF;
                        }
                            // 如果bestDist1 < dist < bestDist2，更新bestDist2
                        else if (dist < bestDist2) {
                            bestDist2 = dist;
                        }
                    }

                    // Step 4：根据阈值 和 角度投票剔除误匹配
                    // Step 4.1：第一关筛选：匹配距离必须小于设定阈值
                    if (bestDist1 <= TH_LOW) {
                        // Step 4.2：第二关筛选：最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱
                        if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2)) {
                            // Step 4.3：记录成功匹配特征点的对应的地图点(来自关键帧)
                            vpMapPointMatches[bestIdxF] = pMP;

                            // 这里的realIdxKF是当前遍历到的关键帧的特征点id
                            const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

                            // Step 4.4：计算匹配点旋转角度差所在的直方图
                            if (mbCheckOrientation) {
                                // angle：每个特征点在提取描述子时的旋转主方向角度，如果图像旋转了，这个角度将发生改变
                                // 所有的特征点的角度变化应该是一致的，通过直方图统计得到最准确的角度变化值
                                float rot = kp.angle - F.mvKeys[bestIdxF].angle;// 该特征点的角度变化值
                                if (rot < 0.0)
                                    rot += 360.0f;
                                int bin = round(rot * factor);// 将rot分配到bin组, 四舍五入, 其实就是离散到对应的直方图组中
                                if (bin == HISTO_LENGTH)
                                    bin = 0;
                                assert(bin >= 0 && bin < HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdxF);       // 直方图统计
                            }
                            nmatches++;
                        }
                    }

                }
                KFit++;
                Fit++;
            } else if (KFit->first < Fit->first) {
                // 对齐
                KFit = vFeatVecKF.lower_bound(Fit->first);
            } else {
                // 对齐
                Fit = F.mFeatVec.lower_bound(KFit->first);
            }
        }

        // Step 5 根据方向剔除误匹配的点
        if (mbCheckOrientation) {
            // index
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            // 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++) {
                // 如果特征点的旋转角度变化量属于这三个组，则保留
                if (i == ind1 || i == ind2 || i == ind3)
                    continue;

                // 剔除掉不在前三的匹配对，因为他们不符合“主流旋转方向”
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                    vpMapPointMatches[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
                    nmatches--;
                }
            }
        }

        return nmatches;
    }


/**
 * @brief 根据Sim3变换，将闭环KF及其共视KF的所有地图点（不考虑当前KF已经匹配的地图点）投影到当前KF，生成新的匹配点对
 *
 * @param[in] pKF               当前KF
 * @param[in] Scw               当前KF和闭环KF之间的Sim3变换
 * @param[in] vpPoints          闭环KF及其共视KF的地图点
 * @param[in] vpMatchedMPs         当前KF的已经匹配的地图点
 * @param[in] nThProjRad                搜索范围
 * @return int                  返回新的成功匹配的点对的数目
 */
    int ORBmatcher::SearchMatchKFAndMPsByProject(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints,
                                                 vector<MapPoint *> &vpMatchedMPs, int nThProjRad) {

        Sophus::Sim3f soseScw = Converter::toSophus(Scw, true);
        Sophus::SE3f Tcw = Sophus::SE3f(soseScw.rotationMatrix(), soseScw.translation() / soseScw.scale());


        // Set of MapPoints already found in the KeyFrame
        set<MapPoint *> spAlreadyFound(vpMatchedMPs.begin(), vpMatchedMPs.end());
        spAlreadyFound.erase(static_cast<MapPoint *>(NULL));
        int nmatches = 0, bestDist, bestIdx;
        for (int iMP = 0, iendMP = vpPoints.size(); iMP < iendMP; iMP++) {
            MapPoint *pMP = vpPoints[iMP];

            if (!pMP) {
                continue;
            }
            if (pMP->isBad()) {
                continue;
            }
            if (spAlreadyFound.count(pMP)) {
                continue;
            }
            if (!GetBestKPIdxInKFToMP(pKF, Tcw, pMP, nThProjRad, bestDist, bestIdx, false)) {
                continue;
            }
//            if (vpMatchedMPs[bestIdx])
//                continue;
            // 该MapPoint与bestIdx对应的特征点匹配成功
            if (bestDist <= TH_LOW) {
                vpMatchedMPs[bestIdx] = pMP;
                nmatches++;
            }
        }
        //  Step 3 返回新的成功匹配的点对的数目
        return nmatches;
    }

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
    int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched,
                                            vector<int> &vnMatches12, int windowSize) {
        int nmatches = 0;
        // F1中特征点和F2中匹配关系，注意是按照F1特征点数目分配空间
        vnMatches12 = vector<int>(F1.mvKeysUn.size(), -1);

        // Step 1 构建旋转直方图，HISTO_LENGTH = 30
        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            // 每个bin里预分配500个，因为使用的是vector不够的话可以自动扩展容量
            rotHist[i].reserve(500);

        const float factor = HISTO_LENGTH / 360.0f;

        // 匹配点对距离，注意是按照F2特征点数目分配空间
        vector<int> vMatchedDistance(F2.mvKeysUn.size(), INT_MAX);
        // 从帧2到帧1的反向匹配，注意是按照F2特征点数目分配空间
        vector<int> vnMatches21(F2.mvKeysUn.size(), -1);

        // 遍历帧1中的所有特征点
        for (size_t i1 = 0, iend1 = F1.mvKeysUn.size(); i1 < iend1; i1++) {
            cv::KeyPoint kp1 = F1.mvKeysUn[i1];
            int level1 = kp1.octave;
            // 只使用原始图像上提取的特征点
            if (level1 > 0)
                continue;

            // Step 2 在半径窗口内搜索当前帧F2中所有的候选匹配特征点
            // vbPrevMatched 输入的是参考帧 F1的特征点
            // windowSize = 100，输入最大最小金字塔层级 均为0
            vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize,
                                                            level1, level1);

            // 没有候选特征点，跳过
            if (vIndices2.empty())
                continue;

            // 取出参考帧F1中当前遍历特征点对应的描述子
            cv::Mat d1 = F1.mDescriptors.row(i1);

            int bestDist = INT_MAX;     //最佳描述子匹配距离，越小越好
            int bestDist2 = INT_MAX;    //次佳描述子匹配距离
            int bestIdx2 = -1;          //最佳候选特征点在F2中的index

            // Step 3 遍历搜索搜索窗口中的所有潜在的匹配候选点，找到最优的和次优的
            for (vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++) {
                size_t i2 = *vit;
                // 取出候选特征点对应的描述子
                cv::Mat d2 = F2.mDescriptors.row(i2);
                // 计算两个特征点描述子距离
                int dist = DescriptorDistance(d1, d2);

                if (vMatchedDistance[i2] <= dist)
                    continue;
                // 如果当前匹配距离更小，更新最佳次佳距离
                if (dist < bestDist) {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestIdx2 = i2;
                } else if (dist < bestDist2) {
                    bestDist2 = dist;
                }
            }

            // Step 4 对最优次优结果进行检查，满足阈值、最优/次优比例，删除重复匹配
            // 即使算出了最佳描述子匹配距离，也不一定保证配对成功。要小于设定阈值
            if (bestDist <= TH_LOW) {
                // 最佳距离比次佳距离要小于设定的比例，这样特征点辨识度更高
                if (bestDist < (float) bestDist2 * mfNNratio) {
                    // 如果找到的候选特征点对应F1中特征点已经匹配过了，说明发生了重复匹配，将原来的匹配也删掉
                    if (vnMatches21[bestIdx2] >= 0) {
                        vnMatches12[vnMatches21[bestIdx2]] = -1;
                        nmatches--;
                    }
                    // 次优的匹配关系，双向建立
                    // vnMatches12保存参考帧F1和F2匹配关系，index保存是F1对应特征点索引，值保存的是匹配好的F2特征点索引
                    vnMatches12[i1] = bestIdx2;
                    vnMatches21[bestIdx2] = i1;
                    vMatchedDistance[bestIdx2] = bestDist;
                    nmatches++;

                    // Step 5 计算匹配点旋转角度差所在的直方图
                    if (mbCheckOrientation) {
                        // 计算匹配特征点的角度差，这里单位是角度°，不是弧度
                        float rot = F1.mvKeysUn[i1].angle - F2.mvKeysUn[bestIdx2].angle;
                        if (rot < 0.0)
                            rot += 360.0f;
                        // 前面factor = HISTO_LENGTH/360.0f
                        // bin = rot / 360.of * HISTO_LENGTH 表示当前rot被分配在第几个直方图bin
                        int bin = round(rot * factor);
                        // 如果bin 满了又是一个轮回
                        if (bin == HISTO_LENGTH)
                            bin = 0;
                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back(i1);
                    }
                }
            }

        }

        // Step 6 筛除旋转直方图中“非主流”部分
        if (mbCheckOrientation) {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;
            // 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++) {
                if (i == ind1 || i == ind2 || i == ind3)
                    continue;
                // 剔除掉不在前三的匹配对，因为他们不符合“主流旋转方向”
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                    int idx1 = rotHist[i][j];
                    if (vnMatches12[idx1] >= 0) {
                        vnMatches12[idx1] = -1;
                        nmatches--;
                    }
                }
            }

        }

        //Update prev matched
        // Step 7 将最后通过筛选的匹配好的特征点保存到vbPrevMatched
        for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
            if (vnMatches12[i1] >= 0)
                vbPrevMatched[i1] = F2.mvKeysUn[vnMatches12[i1]].pt;

        return nmatches;
    }

/*
 * @brief 通过词袋，对关键帧的特征点进行跟踪，该函数用于闭环检测时两个关键帧间的特征点匹配
 * @details 通过bow对pKF和F中的特征点进行快速匹配（不属于同一node的特征点直接跳过匹配）
 * 对属于同一node的特征点通过描述子距离进行匹配
 * 通过距离阈值、比例阈值和角度投票进行剔除误匹配
 * @param  pKF1               KeyFrame1
 * @param  pKF2               KeyFrame2
 * @param  vpMatches12        pKF2中与pKF1匹配的MapPoint，vpMatches12[i]表示匹配的地图点，null表示没有匹配，i表示匹配的pKF1 特征点索引
 * @return                    成功匹配的数量
 */
    int ORBmatcher::SearchMatchKFAndKFByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12) {
        // Step 1 分别取出两个关键帧的特征点、BoW 向量、地图点、描述子
        const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
        const cv::Mat &Descriptors1 = pKF1->mDescriptors;

        const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
        const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
        const cv::Mat &Descriptors2 = pKF2->mDescriptors;

        // 保存匹配结果
        vpMatches12 = vector<MapPoint *>(vpMapPoints1.size(), static_cast<MapPoint *>(NULL));
        vector<bool> vbMatched2(vpMapPoints2.size(), false);

        // Step 2 构建旋转直方图，HISTO_LENGTH = 30
        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);

        const float factor = HISTO_LENGTH / 360.0f;

        int nmatches = 0;

        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while (f1it != f1end && f2it != f2end) {
            // Step 3 开始遍历，分别取出属于同一node的特征点(只有属于同一node，才有可能是匹配点)
            if (f1it->first == f2it->first) {
                // 遍历KF中属于该node的特征点
                for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
                    const size_t idx1 = f1it->second[i1];

                    MapPoint *pMP1 = vpMapPoints1[idx1];
                    if (!pMP1)
                        continue;
                    if (pMP1->isBad())
                        continue;

                    const cv::Mat &d1 = Descriptors1.row(idx1);

                    int bestDist1 = 256;
                    int bestIdx2 = -1;
                    int bestDist2 = 256;

                    // Step 4 遍历KF2中属于该node的特征点，找到了最优及次优匹配点
                    for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++) {
                        const size_t idx2 = f2it->second[i2];

                        MapPoint *pMP2 = vpMapPoints2[idx2];

                        // 如果已经有匹配的点，或者遍历到的特征点对应的地图点无效
                        if (vbMatched2[idx2] || !pMP2)
                            continue;

                        if (pMP2->isBad())
                            continue;

                        const cv::Mat &d2 = Descriptors2.row(idx2);

                        int dist = DescriptorDistance(d1, d2);

                        if (dist < bestDist1) {
                            bestDist2 = bestDist1;
                            bestDist1 = dist;
                            bestIdx2 = idx2;
                        } else if (dist < bestDist2) {
                            bestDist2 = dist;
                        }
                    }

                    // Step 5 对匹配结果进行检查，满足阈值、最优/次优比例，记录旋转直方图信息
                    if (bestDist1 < TH_LOW) {
                        if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2)) {
                            vpMatches12[idx1] = vpMapPoints2[bestIdx2];
                            vbMatched2[bestIdx2] = true;

                            if (mbCheckOrientation) {
                                float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;
                                if (rot < 0.0)
                                    rot += 360.0f;
                                int bin = round(rot * factor);
                                if (bin == HISTO_LENGTH)
                                    bin = 0;
                                assert(bin >= 0 && bin < HISTO_LENGTH);
                                rotHist[bin].push_back(idx1);
                            }
                            nmatches++;
                        }
                    }
                }

                f1it++;
                f2it++;
            } else if (f1it->first < f2it->first) {
                f1it = vFeatVec1.lower_bound(f2it->first);
            } else {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        // Step 6 检查旋转直方图分布，剔除差异较大的匹配
        if (mbCheckOrientation) {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++) {
                if (i == ind1 || i == ind2 || i == ind3)
                    continue;
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                    vpMatches12[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
                    nmatches--;
                }
            }
        }

        return nmatches;
    }

/*
 * @brief 利用基础矩阵F12极线约束，用BoW加速匹配两个关键帧的未匹配的特征点，产生新的匹配点对
 * 具体来说，pKF1图像的每个特征点与pKF2图像同一node节点的所有特征点依次匹配，判断是否满足对极几何约束，满足约束就是匹配的特征点
 * @param pKF1          关键帧1
 * @param pKF2          关键帧2
 * @param F12           从2到1的基础矩阵
 * @param vMatchedPairs 存储匹配特征点对，特征点用其在关键帧中的索引表示
 * @param bOnlyStereo   在双目和rgbd情况下，是否要求特征点在右图存在匹配
 * @return              成功匹配的数量
 */
    int ORBmatcher::SearchKFAndKFByTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                                 vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo) {
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

        // Compute epipole in second image
        // Step 1 计算KF1的相机中心在KF2图像平面的二维像素坐标
        // KF1相机光心在世界坐标系坐标Cw
        cv::Mat Cw = pKF1->GetCameraCenter();
        // KF2相机位姿R2w,t2w,是世界坐标系到相机坐标系
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();
        // KF1的相机光心转化到KF2坐标系中的坐标
        cv::Mat C2 = R2w * Cw + t2w;
        const float invz = 1.0f / C2.at<float>(2);
        // 得到KF1的相机光心在KF2中的坐标，也叫极点，这里是像素坐标
        const float ex = pKF2->fx * C2.at<float>(0) * invz + pKF2->cx;
        const float ey = pKF2->fy * C2.at<float>(1) * invz + pKF2->cy;

        // Find matches between not tracked keypoints
        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node

        int nmatches = 0;
        // 记录匹配是否成功，避免重复匹配
        vector<bool> vbMatched2(pKF2->N, false);
        vector<int> vMatches12(pKF1->N, -1);
        // 用于统计匹配点对旋转差的直方图
        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);

        const float factor = HISTO_LENGTH / 360.0f;

        // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
        // Step 2 利用BoW加速匹配：只对属于同一节点(特定层)的ORB特征进行匹配
        // FeatureVector其实就是一个map类，那就可以直接获取它的迭代器进行遍历
        // FeatureVector的数据结构类似于：{(node1,feature_vector1) (node2,feature_vector2)...}
        // f1it->first对应node编号，f1it->second对应属于该node的所有特特征点编号
        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

        // Step 2.1：遍历pKF1和pKF2中的node节点
        while (f1it != f1end && f2it != f2end) {
            // 如果f1it和f2it属于同一个node节点才会进行匹配，这就是BoW加速匹配原理
            if (f1it->first == f2it->first) {
                // Step 2.2：遍历属于同一node节点(id：f1it->first)下的所有特征点
                for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
                    // 获取pKF1中属于该node节点的所有特征点索引
                    const size_t idx1 = f1it->second[i1];
                    MapPoint *pMP1 = pKF1->GetMapPoint(idx1);
                    // If there is already a MapPoint skip
                    if (pMP1)
                        continue;
                    if (bOnlyStereo)
                        continue;
                    const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
                    const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                    int bestDist = TH_LOW;
                    int bestIdx2 = -1;
                    // Step 2.5：遍历该node节点下(f2it->first)对应KF2中的所有特征点
                    for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++) {
                        size_t idx2 = f2it->second[i2];
                        MapPoint *pMP2 = pKF2->GetMapPoint(idx2);
                        // If we have already matched or there is a MapPoint skip
                        if (vbMatched2[idx2] || pMP2)
                            continue;
                        const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                        const int dist = DescriptorDistance(d1, d2);
                        if (dist > TH_LOW || dist > bestDist)
                            continue;
                        const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
                        const float distex = ex - kp2.pt.x;
                        const float distey = ey - kp2.pt.y;
                        // Step 2.7 极点e2到kp2的像素距离如果小于阈值th,认为kp2对应的MapPoint距离pKF1相机太近，跳过该匹配点对
                        // 作者根据kp2金字塔尺度因子(scale^n，scale=1.2，n为层数)定义阈值th
                        // 金字塔层数从0到7，对应距离 sqrt(100*pKF2->mvScaleFactors[kp2.octave]) 是10-20个像素
                        if (distex * distex + distey * distey < 100 * pKF2->mvScaleFactors[kp2.octave])
                            continue;
                        if (CheckDistEpipolarLine(kp1, kp2, F12, pKF2)) {
                            bestIdx2 = idx2;
                            bestDist = dist;
                        }
                    }
                    if (bestIdx2 >= 0) {
                        const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
                        vMatches12[idx1] = bestIdx2;
                        vbMatched2[bestIdx2] = true;  // !记录已经匹配，避免重复匹配。原作者漏掉！
                        nmatches++;
                        // 记录旋转差直方图信息
                        if (mbCheckOrientation) {
                            // angle：角度，表示匹配点对的方向差。
                            float rot = kp1.angle - kp2.angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                    }
                }
                f1it++;
                f2it++;
            } else if (f1it->first < f2it->first) {
                f1it = vFeatVec1.lower_bound(f2it->first);
            } else {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        // Step 3 用旋转差直方图来筛掉错误匹配对
        if (mbCheckOrientation) {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;
            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
            for (int i = 0; i < HISTO_LENGTH; i++) {
                if (i == ind1 || i == ind2 || i == ind3)
                    continue;
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                    vbMatched2[vMatches12[rotHist[i][j]]] = false;  // !清除匹配关系。原作者漏掉！
                    vMatches12[rotHist[i][j]] = -1;
                    nmatches--;
                }
            }
        }

        // Step 4 存储匹配关系，下标是关键帧1的特征点id，存储的是关键帧2的特征点id
        vMatchedPairs.clear();
        vMatchedPairs.reserve(nmatches);
        for (size_t i = 0, iend = vMatches12.size(); i < iend; i++) {
            if (vMatches12[i] < 0)
                continue;
            vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
        }
        return nmatches;
    }

/**
 * @brief 将地图点投影到关键帧中进行匹配和融合；融合策略如下
 * 1.如果地图点能匹配关键帧的特征点，并且该点有对应的地图点，那么选择观测数目多的替换两个地图点
 * 2.如果地图点能匹配关键帧的特征点，并且该点没有对应的地图点，那么为该点添加该投影地图点

 * @param[in] pKF           关键帧
 * @param[in] vpMapPoints   待投影的地图点
 * @param[in] nThProjRad            搜索窗口的阈值，默认为3
 * @return int              更新地图点的数量
 */
    int ORBmatcher::SearchReplaceKFAndMPsByProjectInLocalMap(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints,
                                                             const float nThProjRad) {
        Sophus::SE3f Tcw = Converter::toSophus(pKF->GetPose());
        cv::Mat Ow = pKF->GetCameraCenter();
        int nMatches = 0, bestDist, bestIdx;
        const int nMPs = vpMapPoints.size();
        for (int i = 0; i < nMPs; i++) {
            MapPoint *pMP = vpMapPoints[i];
            if (!pMP)
                continue;
            if (pMP->isBad()) {
                continue;
            }
            if (pMP->IsInKeyFrame(pKF)) {
                continue;
            }
            if (!GetBestKPIdxInKFToMP(pKF, Tcw, pMP, nThProjRad, bestDist, bestIdx, true)) {
                continue;
            }

            if (bestDist <= TH_LOW) {
                MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
                if (pMPinKF) {
                    // 如果最佳匹配点有对应有效地图点，选择被观测次数最多的那个替换
                    if (!pMPinKF->isBad()) {
                        if (pMPinKF->Observations() > pMP->Observations())
                            pMP->Replace(pMPinKF);
                        else
                            pMPinKF->Replace(pMP);
                    }
                } else {
                    // 如果最佳匹配点没有对应地图点，添加观测信息
                    pMP->AddObservation(pKF, bestIdx);
                    pKF->AddMapPoint(pMP, bestIdx);
                }
                nMatches++;
            }
        }
        return nMatches;
    }


/**
 * @brief 闭环矫正中使用。将当前关键帧闭环匹配上的关键帧及其共视关键帧组成的地图点投影到当前关键帧，融合地图点
 *
 * @param[in] pKF                   当前关键帧
 * @param[in] Scw                   当前关键帧经过闭环Sim3 后的世界到相机坐标系的Sim变换
 * @param[in] vpCandidMPs              与当前关键帧闭环匹配上的关键帧及其共视关键帧组成的地图点
 * @param[in] nThProjRad                    搜索范围系数
 * @param[out] vpReplacePoint       替换的地图点
 * @return int                      融合（替换和新增）的地图点数目
 */
    int ORBmatcher::SearchReplaceKFAndMPsByProjectInGlobalMap(KeyFrame *pKF, cv::Mat Scw,
                                                              const vector<MapPoint *> &vpCandidMPs, float nThProjRad,
                                                              vector<MapPoint *> &vpReplacePoint) {
        Sophus::Sim3f soseScw = Converter::toSophus(Scw, true);
        Sophus::SE3f Tcw = Sophus::SE3f(soseScw.rotationMatrix(), soseScw.translation() / soseScw.scale());
        const set<MapPoint *> spAlreadyFound = pKF->GetMapPoints();
        int nMatches = 0, bestDist, bestIdx;
        const int nPoints = vpCandidMPs.size();
        for (int iMP = 0; iMP < nPoints; iMP++) {
            MapPoint *pMP = vpCandidMPs[iMP];
            if (!pMP) {
                continue;
            }
            if (pMP->isBad()) {
                continue;
            }
            if (spAlreadyFound.count(pMP)) {
                continue;
            }
            if (!GetBestKPIdxInKFToMP(pKF, Tcw, pMP, nThProjRad, bestDist, bestIdx, false)) {
                continue;
            }

            // If there is already a MapPoint replace otherwise add new measurement
            if (bestDist <= TH_LOW) {
                MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
                if (pMPinKF) {
                    // 如果这个地图点已经存在，则记录要替换信息
                    // 这里不能直接替换，原因是需要对地图点加锁后才能替换，否则可能会crash。所以先记录，在加锁后替换
                    if (!pMPinKF->isBad())
                        vpReplacePoint[iMP] = pMPinKF;
                } else {
                    // 如果这个地图点不存在，直接添加
                    pMP->AddObservation(pKF, bestIdx);
                    pKF->AddMapPoint(pMP, bestIdx);
                }
                nMatches++;
            }
        }
        cout << nMatches << ": Replace MPs Num In SearchReplaceKFAndMPsByProjectInGlobalMap" << endl;
        return nMatches;
    }

/**
 * @brief 通过Sim3变换，搜索两个关键帧中更多的匹配点对
 * （之前使用SearchByBoW进行特征点匹配时会有漏匹配）
 * @param[in] pKF1              当前帧
 * @param[in] pKF2              闭环候选帧
 * @param[in] vpMatches12       i表示匹配的pKF1 特征点索引，vpMatches12[i]表示匹配的pKF2中地图点，null表示没有匹配
 * @param[in] s12               pKF2 到 pKF1 的Sim 变换中的尺度
 * @param[in] R12               pKF2 到 pKF1 的Sim 变换中的旋转矩阵
 * @param[in] t12               pKF2 到 pKF1 的Sim 变换中的平移向量
 * @param[in] nThProjRad                搜索窗口的倍数
 * @return int                  新增的匹配点对数目
 */
    int ORBmatcher::SearchMatchKFAndKFByProject(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12,
                                                const float &s12, const cv::Mat &R12, const cv::Mat &t12,
                                                const float nThProjRad) {

        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();
        cv::Mat sR21 = (1.0 / s12) * R12.t();
        cv::Mat t21 = -sR21 * t12;
        Sophus::SE3f T12 = Converter::toSophus(R12, t12);
        Sophus::SE3f T21 = Converter::toSophus(R12.t(), t21);


        const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
        const int N1 = vpMapPoints1.size();
        const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
        const int N2 = vpMapPoints2.size();
        vector<bool> vbAlreadyMatched1(N1, false);
        vector<bool> vbAlreadyMatched2(N2, false);
        for (int i = 0; i < N1; i++) {
            MapPoint *pMP = vpMatches12[i];
            if (pMP) {
                vbAlreadyMatched1[i] = true;
                int idx2 = pMP->GetIndexInKeyFrame(pKF2);
                if (idx2 >= 0 && idx2 < N2)
                    vbAlreadyMatched2[idx2] = true;
            }
        }
        int bestDist, bestIdx;
        vector<int> vnMatch1(N1, -1);
        vector<int> vnMatch2(N2, -1);
        for (int i1 = 0; i1 < N1; i1++) {
            MapPoint *pMP = vpMapPoints1[i1];
            if (!pMP)
                continue;
            if (pMP->isBad())
                continue;
            if (vbAlreadyMatched1[i1])
                continue;
            if (!GetBestKPIdxInKFToMP(pKF2, T21, pMP, nThProjRad, bestDist, bestIdx, false, (1.0 / s12))) {
                continue;
            }
            if (bestDist <= TH_HIGH) {
                vnMatch1[i1] = bestIdx;
            }
        }

        // Transform from KF2 to KF1 and search
        // Step 4：通过Sim变换，寻找 pKF2 中特征点和 pKF1 中的新的匹配
        // 具体步骤同上
        for (int i2 = 0; i2 < N2; i2++) {
            MapPoint *pMP = vpMapPoints2[i2];
            if (!pMP)
                continue;
            if (pMP->isBad())
                continue;
            if (vbAlreadyMatched2[i2])
                continue;
            if (!GetBestKPIdxInKFToMP(pKF1, T12, pMP, nThProjRad, bestDist, bestIdx, false, s12)) {
                continue;
            }
            if (bestDist <= TH_HIGH) {
                vnMatch2[i2] = bestIdx;
            }
        }

        // Check agreement
        // Step 5： 一致性检查,只有在两次互相匹配中都出现才能够认为是可靠的匹配
        int nFound = 0;
        for (int i1 = 0; i1 < N1; i1++) {
            int idx2 = vnMatch1[i1];
            if (idx2 >= 0) {
                int idx1 = vnMatch2[idx2];
                if (idx1 == i1) {
                    // 更新匹配的地图点
                    vpMatches12[i1] = vpMapPoints2[idx2];
                    nFound++;
                }
            }
        }
        return nFound;
    }

/**
 * @brief 将上一帧跟踪的地图点投影到当前帧，并且搜索匹配点。用于跟踪前一帧
 * 步骤
 * Step 1 建立旋转直方图，用于检测旋转一致性
 * Step 2 计算当前帧和前一帧的平移向量
 * Step 3 对于前一帧的每一个地图点，通过相机投影模型，得到投影到当前帧的像素坐标
 * Step 4 根据相机的前后前进方向来判断搜索尺度范围
 * Step 5 遍历候选匹配点，寻找距离最小的最佳匹配点
 * Step 6 计算匹配点旋转角度差所在的直方图
 * Step 7 进行旋转一致检测，剔除不一致的匹配
 * @param[in] CurrentFrame          当前帧
 * @param[in] LastFrame             上一帧
 * @param[in] nThProjRad                    搜索范围阈值，默认单目为7，双目15
 * @param[in] bMono                 是否为单目
 * @return int                      成功匹配的数量
 */
    int ORBmatcher::SearchFrameAndFrameByProject(Frame &CurrentFrame, const Frame &LastFrame, const float nThProjRad,
                                                 const bool bMono) {
        int nmatches = 0;

        // Rotation Histogram (to check rotation consistency)
        // Step 1 建立旋转直方图，用于检测旋转一致性
        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);

        //! 原作者代码是 const float factor = 1.0f/HISTO_LENGTH; 是错误的，更改为下面代码
        const float factor = HISTO_LENGTH / 360.0f;

        // Step 2 计算当前帧和前一帧的平移向量
        //当前帧的相机位姿
        const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);

        //当前相机坐标系到世界坐标系的平移向量
        const cv::Mat twc = -Rcw.t() * tcw;

        //上一帧的相机位姿
        const cv::Mat Rlw = LastFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tlw = LastFrame.mTcw.rowRange(0, 3).col(3); // tlw(l)

        // vector from LastFrame to CurrentFrame expressed in LastFrame
        // 当前帧相对于上一帧相机的平移向量
        const cv::Mat tlc = Rlw * twc + tlw;

        // 判断前进还是后退
        const bool bForward = tlc.at<float>(2) > CurrentFrame.mb && !bMono;     // 非单目情况，如果Z大于基线，则表示相机明显前进
        const bool bBackward = -tlc.at<float>(2) > CurrentFrame.mb && !bMono;   // 非单目情况，如果-Z小于基线，则表示相机明显后退

        //  Step 3 对于前一帧的每一个地图点，通过相机投影模型，得到投影到当前帧的像素坐标
        for (int i = 0; i < LastFrame.N; i++) {
            MapPoint *pMP = LastFrame.mvpMapPoints[i];

            if (pMP) {
                if (!LastFrame.mvbOutlier[i]) {
                    // 对上一帧有效的MapPoints投影到当前帧坐标系
                    cv::Mat x3Dw = pMP->GetWorldPos();
                    cv::Mat x3Dc = Rcw * x3Dw + tcw;

                    const float xc = x3Dc.at<float>(0);
                    const float yc = x3Dc.at<float>(1);
                    const float invzc = 1.0 / x3Dc.at<float>(2);
                    if (invzc < 0)
                        continue;

                    float u = CurrentFrame.fx * xc * invzc + CurrentFrame.cx;
                    float v = CurrentFrame.fy * yc * invzc + CurrentFrame.cy;
                    if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX)
                        continue;
                    if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
                        continue;

                    // 上一帧中地图点对应二维特征点所在的金字塔层级
                    int nLastOctave = LastFrame.mvKeys[i].octave;

                    // Search in a window. Size depends on scale
                    float radius = nThProjRad * CurrentFrame.mvScaleFactors[nLastOctave]; // 尺度越大，搜索范围越大

                    // 记录候选匹配点的id
                    vector<size_t> vIndices2;
                    if (bForward) // 前进,则上一帧兴趣点在所在的尺度nLastOctave<=nCurOctave
                        vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave);
                    else if (bBackward) // 后退,则上一帧兴趣点在所在的尺度0<=nCurOctave<=nLastOctave
                        vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, 0, nLastOctave);
                    else // 在[nLastOctave-1, nLastOctave+1]中搜索
                        vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave - 1, nLastOctave + 1);
                    if (vIndices2.empty())
                        continue;

                    const cv::Mat dMP = pMP->GetDescriptor();
                    int bestDist = 256;
                    int bestIdx2 = -1;
                    // Step 5 遍历候选匹配点，寻找距离最小的最佳匹配点
                    for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end();
                         vit != vend; vit++) {
                        const size_t i2 = *vit;
                        // 如果该特征点已经有对应的MapPoint了,则退出该次循环
                        if (CurrentFrame.mvpMapPoints[i2])
                            if (CurrentFrame.mvpMapPoints[i2]->Observations() > 0)
                                continue;

                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);
                        const int dist = DescriptorDistance(dMP, d);
                        if (dist < bestDist) {
                            bestDist = dist;
                            bestIdx2 = i2;
                        }
                    }

                    // 最佳匹配距离要小于设定阈值
                    if (bestDist <= TH_HIGH) {
                        CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
                        nmatches++;

                        // Step 6 计算匹配点旋转角度差所在的直方图
                        if (mbCheckOrientation) {
                            float rot = LastFrame.mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle;
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
                    }
                }
            }
        }

        //Apply rotation consistency
        //  Step 7 进行旋转一致检测，剔除不一致的匹配
        if (mbCheckOrientation) {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

            for (int i = 0; i < HISTO_LENGTH; i++) {
                // 对于数量不是前3个的点对，剔除
                if (i != ind1 && i != ind2 && i != ind3) {
                    for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint *>(NULL);
                        nmatches--;
                    }
                }
            }
        }

        return nmatches;
    }

/**
 * @brief 通过投影的方式将关键帧中未匹配的地图点投影到当前帧中,进行匹配，并通过旋转直方图进行筛选
 *
 * @param[in] CurrentFrame          当前帧
 * @param[in] pKF                   参考关键帧
 * @param[in] sAlreadyFound         已经找到的地图点集合，不会用于PNP
 * @param[in] nThProjRad                    匹配时搜索范围，会乘以金字塔尺度
 * @param[in] ORBdist               匹配的ORB描述子距离应该小于这个阈值
 * @return int                      成功匹配的数量
 */
    int ORBmatcher::SearchFrameAndKFByProject(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint *> &sAlreadyFound,
                                              const float nThProjRad, const int ORBdist) {
        int nmatches = 0;

        const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);
        const cv::Mat Ow = -Rcw.t() * tcw;

        // Rotation Histogram (to check rotation consistency)
        // Step 1 建立旋转直方图，用于检测旋转一致性
        vector<int> rotHist[HISTO_LENGTH];
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = HISTO_LENGTH / 360.0f;

        const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        // Step 2 遍历关键帧中的每个地图点，通过相机投影模型，得到投影到当前帧的像素坐标
        for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
            MapPoint *pMP = vpMPs[i];
            if (!pMP) {
                continue;
            }
            if (pMP->isBad()) {
                continue;
            }
            if (sAlreadyFound.count(pMP)) {
                continue;
            }

            //Project
            cv::Mat x3Dw = pMP->GetWorldPos();
            cv::Mat x3Dc = Rcw * x3Dw + tcw;
            const float xc = x3Dc.at<float>(0);
            const float yc = x3Dc.at<float>(1);
            const float invzc = 1.0 / x3Dc.at<float>(2);
            const float u = CurrentFrame.fx * xc * invzc + CurrentFrame.cx;
            const float v = CurrentFrame.fy * yc * invzc + CurrentFrame.cy;
            if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX)
                continue;
            if (v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
                continue;

            // Compute predicted scale level
            cv::Mat PO = x3Dw - Ow;
            float dist3D = cv::norm(PO);
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            // Depth must be inside the scale pyramid of the image
            if (dist3D < minDistance || dist3D > maxDistance)
                continue;
            //预测尺度
            int nPredictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);
            // Search in a window
            // 搜索半径和尺度相关
            const float radius = nThProjRad * CurrentFrame.mvScaleFactors[nPredictedLevel];
            //  Step 3 搜索候选匹配点
            const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel - 1,
                                                                            nPredictedLevel + 1);
            if (vIndices2.empty())
                continue;
            const cv::Mat dMP = pMP->GetDescriptor();
            int bestDist = 256;
            int bestIdx2 = -1;
            // Step 4 遍历候选匹配点，寻找距离最小的最佳匹配点
            for (vector<size_t>::const_iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++) {
                const size_t i2 = *vit;
                if (CurrentFrame.mvpMapPoints[i2])
                    continue;
                const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);
                const int dist = DescriptorDistance(dMP, d);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx2 = i2;
                }
            }

            if (bestDist <= ORBdist) {
                CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
                nmatches++;
                // Step 5 计算匹配点旋转角度差所在的直方图
                if (mbCheckOrientation) {
                    float rot = pKF->mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle;
                    if (rot < 0.0)
                        rot += 360.0f;
                    int bin = round(rot * factor);
                    if (bin == HISTO_LENGTH)
                        bin = 0;
                    assert(bin >= 0 && bin < HISTO_LENGTH);
                    rotHist[bin].push_back(bestIdx2);
                }
            }
        }

        //  Step 6 进行旋转一致检测，剔除不一致的匹配
        if (mbCheckOrientation) {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;
            ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);
            for (int i = 0; i < HISTO_LENGTH; i++) {
                if (i != ind1 && i != ind2 && i != ind3) {
                    for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]] = NULL;
                        nmatches--;
                    }
                }
            }
        }
        return nmatches;
    }

/**
 * @brief 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
 *
 * @param[in] histo         匹配特征点对旋转方向差直方图
 * @param[in] L             直方图尺寸
 * @param[in & out] ind1          bin值第一大对应的索引
 * @param[in & out] ind2          bin值第二大对应的索引
 * @param[in & out] ind3          bin值第三大对应的索引
 */
    void ORBmatcher::ComputeThreeMaxima(vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3) {
        int max1 = 0;
        int max2 = 0;
        int max3 = 0;

        for (int i = 0; i < L; i++) {
            const int s = histo[i].size();
            if (s > max1) {
                max3 = max2;
                max2 = max1;
                max1 = s;
                ind3 = ind2;
                ind2 = ind1;
                ind1 = i;
            } else if (s > max2) {
                max3 = max2;
                max2 = s;
                ind3 = ind2;
                ind2 = i;
            } else if (s > max3) {
                max3 = s;
                ind3 = i;
            }
        }

        // 如果差距太大了,说明次优的非常不好,这里就索性放弃了,都置为-1
        if (max2 < 0.1f * (float) max1) {
            ind2 = -1;
            ind3 = -1;
        } else if (max3 < 0.1f * (float) max1) {
            ind3 = -1;
        }
    }


// Bit set count operation from
// Hamming distance：两个二进制串之间的汉明距离，指的是其不同位数的个数
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;

        // 8*32=256bit

        for (int i = 0; i < 8; i++, pa++, pb++) {
            unsigned int v = *pa ^ *pb;        // 相等为0,不等为1
            // 下面的操作就是计算其中bit为1的个数了,这个操作看上面的链接就好
            // 其实我觉得也还阔以直接使用8bit的查找表,然后做32次寻址操作就完成了;不过缺点是没有利用好CPU的字长
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

} //namespace ORB_SLAM
