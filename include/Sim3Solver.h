/**
 * @file Sim3Solver.h
 * @author guoqing (1337841346@qq.com)
 * @brief sim3 求解
 * @version 0.1
 * @date 2019-05-07
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


#ifndef SIM3SOLVER_H
#define SIM3SOLVER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "KeyFrame.h"



namespace ORB_SLAM2
{

/** @brief Sim3 求解器 */
class Sim3Solver
{
public:

    /**
     * @brief Sim 3 Solver 构造函数
     * @param[in] pKF1              当前关键帧
     * @param[in] pKF2              候选的闭环关键帧
     * @param[in] vpMatched12       通过词袋模型加速匹配所得到的,两帧特征点的匹配关系所得到的地图点,本质上是来自于候选闭环关键帧的地图点
     * @param[in] bFixScale         当前传感器类型的输入需不需要计算尺度。单目的时候需要，双目和RGBD的时候就不需要了
     */
    Sim3Solver();




    /**
     * @brief Ransac求解mvX3Dc1和mvX3Dc2之间Sim3，函数返回mvX3Dc2到mvX3Dc1的Sim3变换
     *
     * @param[in] nIterations           设置的最大迭代次数
     * @param[in] bNoMore               为true表示穷尽迭代还没有找到好的结果，说明求解失败
     * @param[in] vbInliers             标记是否是内点
     * @param[in] nInliers              内点数目
     * @return cv::Mat                  计算得到的Sim3矩阵
     */
    cv::Mat KFsiterate(std::vector<cv::Mat> &vP1s, std::vector<cv::Mat> &vP2s, int nIterations, vector<bool> &vbInliers, int &nInliers);



protected:

    /**
     * @brief 给出三个点,计算它们的质心以及去质心之后的坐标
     * 
     * @param[in] P     输入的3D点
     * @param[in] Pr    去质心后的点
     * @param[in] C     质心
     */
    void ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C);

    /**
     * @brief 根据两组匹配的3D点,计算P2到P1的Sim3变换
     * @param[in] P1    匹配的3D点(三个,每个的坐标都是列向量形式,三个点组成了3x3的矩阵)(当前关键帧)
     * @param[in] P2    匹配的3D点(闭环关键帧)
     */
    void ComputeSim3(cv::Mat &P1, cv::Mat &P2);




protected:


    // Current Estimation
    cv::Mat mR12i;                              // 存储某次RANSAC过程中得到的旋转
    cv::Mat mt12i;                              // 存储某次RANSAC过程中得到的平移
    float ms12i;                                // 存储某次RANSAC过程中得到的缩放系数
    cv::Mat mT12i;                              // 存储某次RANSAC过程中得到的变换矩阵
    cv::Mat mT21i;                              // 上面的逆
    std::vector<bool> mvbInliersi;              // 内点标记,下标和N,mvpMapPoints1等一致,用于记录某次迭代过程中的内点情况
    int mnInliersi;                             // 在某次迭代的过程中经过投影误差进行的inlier检测得到的内点数目
    double mdInliersTotErr;
    // Current Ransac State
    std::vector<bool> mvbBestInliers;           // 累计的,多次RANSAC中最好的最多的内点个数时的内点标记
    int mnBestInliers;                          // 最好的一次迭代中,得到的内点个数
    double mdBestInliersAvgErr;
    cv::Mat mBestT12;                           // 存储最好的一次迭代中得到的变换矩阵


    // Scale is fixed to 1 in the stereo/RGBD case
    bool mbFixScale;                            // 当前传感器输入的情况下,是否需要计算尺度

    // Indices for random selection
    std::vector<size_t> mvAllIndices;           // RANSAC中随机选择的时候,存储可以选择的点的id(去除那些存在问题的匹配点后重新排序)


};

} //namespace ORB_SLAM

#endif // SIM3SOLVER_H
