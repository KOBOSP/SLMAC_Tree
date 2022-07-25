/**
 * @file Sim3Solver.cc
 * @author guoqing (1337841346@qq.com)
 * @brief sim3 求解器
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


#include "Sim3Solver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>

#include "KeyFrame.h"
#include "ORBmatcher.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

namespace ORB_SLAM2
{

 /**
 * @brief Sim 3 Solver 构造函数
 * @param[in] pKF1              当前关键帧
 * @param[in] pKF2              候选的闭环关键帧
 * @param[in] vpMatched12       通过词袋模型加速匹配所得到的,两帧特征点的匹配关系所得到的地图点,本质上是来自于候选闭环关键帧的地图点
 * @param[in] bFixScale         当前传感器类型的输入需不需要计算尺度。单目的时候需要，双目和RGBD的时候就不需要了
 */
 Sim3Solver::Sim3Solver(){
 }



/**
 * @brief 根据两组many匹配的3D点,计算P2到P1的Sim3变换
 * @brief Ransac求解cv::Mat &P1和cv::Mat &P2之间Sim3，函数返回P2到P1的Sim3变换
 *
 * @param[in] P1    匹配的3D点(三个,每个的坐标都是列向量形式,三个点组成了3x3的矩阵)(当前关键帧)Gps
 * @param[in] P2    匹配的3D点(闭环关键帧)Vo
 * @param[in] nIterations           设置的最大迭代次数
 * @param[in] bNoMore               为true表示穷尽迭代还没有找到好的结果，说明求解失败
 * @param[in] vbInliers             标记是否是内点
 * @param[in] nInliers              内点数目
 * @return cv::Mat                  计算得到的Sim3矩阵
 */
cv::Mat Sim3Solver::KFsiterate(std::vector<cv::Mat> &vP1s, std::vector<cv::Mat> &vP2s, int nIterations, vector<bool> &vbInliers, int &nInliers){
    int nPMostNum=vP1s.size();
    // Step 1 如果匹配点比要求的最少内点数还少，不满足Sim3 求解条件，返回空
    if(nPMostNum<15){
        return cv::Mat();
    }
    mbFixScale=false;
    mnBestInliers=-1;
    mdBestInliersAvgErr=9999;
    // 可以使用的点对的索引,为了避免重复使用
    vector<size_t> vAvailableIndices;
    mvAllIndices.clear();
    mvAllIndices.reserve(nPMostNum);
    mvbInliersi.clear();
    mvbInliersi.reserve(nPMostNum);
    vbInliers = vector<bool>(nPMostNum,false);    // 的确和最初传递给这个解算器的地图点向量是保持一致
    for(int idx=0;idx<nPMostNum;idx++){
        mvAllIndices.emplace_back(idx);
        mvbInliersi.emplace_back(false);
    }

    // 随机选择的来自于这两个帧的三对匹配点
    cv::Mat P3Dc1i(3,3,CV_32F);
    cv::Mat P3Dc2i(3,3,CV_32F);

    int nCurrentIterations = 0;
    // Step 2 随机选择三个点，用于求解后面的Sim3
    while(nCurrentIterations<nIterations){
        nCurrentIterations++;// 这个函数中迭代的次数
        // 记录所有有效（可以采样）的候选三维点索引
        vAvailableIndices = mvAllIndices;
        // Get min set of points
        // Step 2.1 随机取三组点，取完后从候选索引中删掉
        for(short i = 0; i < 3; ++i){
            // DBoW3中的随机数生成函数
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];
            // P3Dc1i和P3Dc2i中点的排列顺序：
            // x1 x2 x3 ...
            // y1 y2 y3 ...
            // z1 z2 z3 ...
            vP2s[idx].copyTo(P3Dc2i.col(i));
            vP1s[idx].copyTo(P3Dc1i.col(i));
            // 从"可用索引列表"中删除这个点的索引
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
        cv::Mat ray11=P3Dc1i.col(1)-P3Dc1i.col(0);
        cv::Mat ray12=P3Dc1i.col(2)-P3Dc1i.col(0);
        cv::Mat ray21=P3Dc2i.col(1)-P3Dc2i.col(0);
        cv::Mat ray22=P3Dc2i.col(2)-P3Dc2i.col(0);
        const float cosParallaxRays1 = ray11.dot(ray12)/(cv::norm(ray11)*cv::norm(ray12));
        if(abs(cosParallaxRays1)<0.1392||abs(cosParallaxRays1)>0.9903){//cos8= 0.9903 cos10deg=0.9848 cos15=<0.9659 cos 82=0.1392
            continue;
        }
        // Step 2.2 根据随机取的两组匹配的3D点，计算P3Dc2i 到 P3Dc1i 的Sim3变换. modify:mT12i mt12i mR12i ms12i
        ComputeSim3(P3Dc2i, P3Dc1i);//in Function: 1Vo<-2Gps
        // Step 2.3 对计算的Sim3变换，通过投影误差进行inlier检测. modify:mnInliersi and mvbInliersi
        mnInliersi=0;
        mdInliersTotErr=0;
        cv::Mat sR21 = mT21i.rowRange(0,3).colRange(0,3);
        cv::Mat t21 = mT21i.rowRange(0,3).col(3);
        // 对每个3D地图点进行投影操作
        for(size_t i=0, iend=nPMostNum; i<iend; i++) {
            // 首先将对方关键帧的地图点坐标转换到这个关键帧的相机坐标系下
            cv::Mat P3D2in1 = sR21 * vP2s[i] + t21;//1->2: Vo->Gps
            double error=cv::norm(P3D2in1-vP1s[i]);
            if(error<1.0){//small than 1 meter
                mdInliersTotErr+=error;
                mnInliersi++;
                mvbInliersi[i]=true;
            }
            else{
                mvbInliersi[i]= false;
            }
        }
        // Step 2.4 记录并更新最多的内点数目及对应的参数
        if(mnInliersi>mnBestInliers&&mdBestInliersAvgErr>mdInliersTotErr/mnInliersi){
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mdBestInliersAvgErr=mdInliersTotErr/mnInliersi;
            mBestT12 = mT12i.clone();
        } // 更新最多的内点数目
    } // 迭代循环
    if(mnBestInliers>nPMostNum*2.0/3.0&&mnBestInliers>50){
        // 返回值,告知得到的内点数目
        nInliers = mnBestInliers;
        for(int i=0; i<nPMostNum; i++){
            if(mvbBestInliers[i]){
                vbInliers[i] = true;
            }
        }
        return mBestT12;
    }
    else{
        // Step 3 如果已经达到了最大迭代次数了还没得到满足条件的Sim3，说明失败了，放弃，返回
        return cv::Mat();   // no more的时候返回的是一个空矩阵
    }
}

/**
 * @brief 给出三个点,计算它们的质心以及去质心之后的坐标
 * 
 * @param[in] P     输入的3D点
 * @param[in] Pr    去质心后的点
 * @param[in] C     质心
 */
void Sim3Solver::ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
{
    // 矩阵P每一行求和，结果存在C。这两句也可以使用CV_REDUCE_AVG选项来实现
    cv::reduce(P,C,1,CV_REDUCE_SUM);
    C = C/P.cols;// 求平均

    for(int i=0; i<P.cols; i++)
    {
        Pr.col(i)=P.col(i)-C;//减去质心
    }
}


/**
 * @brief 根据两组匹配的3D点,计算P2到P1的Sim3变换
 * @param[in] P1    匹配的3D点(三个,每个的坐标都是列向量形式,三个点组成了3x3的矩阵)(当前关键帧)
 * @param[in] P2    匹配的3D点(闭环关键帧)
 */
void Sim3Solver::ComputeSim3(cv::Mat &P1, cv::Mat &P2)
{
    // Sim3计算过程参考论文:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: 定义3D点质心及去质心后的点
    // O1和O2分别为P1和P2矩阵中3D点的质心
    // Pr1和Pr2为减去质心后的3D点
    cv::Mat Pr1(P1.size(),P1.type()); // Relative coordinates to centroid (set 1)
    cv::Mat Pr2(P2.size(),P2.type()); // Relative coordinates to centroid (set 2)
    cv::Mat O1(3,1,Pr1.type()); // Centroid of P1
    cv::Mat O2(3,1,Pr2.type()); // Centroid of P2

    ComputeCentroid(P1,Pr1,O1);
    ComputeCentroid(P2,Pr2,O2);
    // Step 2: 计算论文中三维点数目n>3的 M 矩阵。这里只使用了3个点
    // Pr2 对应论文中 r_l,i'，Pr1 对应论文中 r_r,i',计算的是P2到P1的Sim3，论文中是left 到 right的Sim3
    cv::Mat M = Pr2*Pr1.t();

    // Step 3: 计算论文中的 mnKeyPointNum 矩阵

    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    cv::Mat N(4,4,P1.type());

    N11 = M.at<float>(0,0)+M.at<float>(1,1)+M.at<float>(2,2);   // Sxx+Syy+Szz
    N12 = M.at<float>(1,2)-M.at<float>(2,1);                    // Syz-Szy
    N13 = M.at<float>(2,0)-M.at<float>(0,2);                    // Szx-Sxz
    N14 = M.at<float>(0,1)-M.at<float>(1,0);                    // ...
    N22 = M.at<float>(0,0)-M.at<float>(1,1)-M.at<float>(2,2);
    N23 = M.at<float>(0,1)+M.at<float>(1,0);
    N24 = M.at<float>(2,0)+M.at<float>(0,2);
    N33 = -M.at<float>(0,0)+M.at<float>(1,1)-M.at<float>(2,2);
    N34 = M.at<float>(1,2)+M.at<float>(2,1);
    N44 = -M.at<float>(0,0)-M.at<float>(1,1)+M.at<float>(2,2);

    N = (cv::Mat_<float>(4,4) << N11, N12, N13, N14,
                                 N12, N22, N23, N24,
                                 N13, N23, N33, N34,
                                 N14, N24, N34, N44);


    // Step 4: 特征值分解求最大特征值对应的特征向量，就是我们要求的旋转四元数

    cv::Mat eval, evec;  // val vec
    // 特征值默认是从大到小排列，所以evec[0] 是最大值
    cv::eigen(N,eval,evec); 

    // mnKeyPointNum 矩阵最大特征值（第一个特征值）对应特征向量就是要求的四元数（q0 q1 q2 q3），其中q0 是实部
    // 将(q1 q2 q3)放入vec（四元数的虚部）
    cv::Mat vec(1,3,evec.type());
    (evec.row(0).colRange(1,4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)


    // Rotation angle. sin is the norm of the imaginary part, cos is the real part
    // 四元数虚部模长 norm(vec)=sin(theta/2), 四元数实部 evec.at<float>(0,0)=q0=cos(theta/2)
    // 这一步的ang实际是theta/2，theta 是旋转向量中旋转角度
    // ? 这里也可以用 arccos(q0)=angle/2 得到旋转角吧
    double ang=atan2(norm(vec),evec.at<float>(0,0));

    // vec/norm(vec)归一化得到归一化后的旋转向量,然后乘上角度得到包含了旋转轴和旋转角信息的旋转向量vec
    vec = 2*ang*vec/norm(vec); //Angle-axis x. quaternion angle is the half

    mR12i.create(3,3,P1.type());
    // 旋转向量（轴角）转换为旋转矩阵
    cv::Rodrigues(vec,mR12i); // computes the rotation matrix from angle-axis

    // Step 5: Rotate set 2
    // 利用刚计算出来的旋转将三维点旋转到同一个坐标系，P3对应论文里的 r_l,i', Pr1 对应论文里的r_r,i'
    cv::Mat P3 = mR12i*Pr2;

    // Step 6: 计算尺度因子 Scale
    if(!mbFixScale){
        // 论文中有2个求尺度方法。一个是p632右中的位置，考虑了尺度的对称性
        // 代码里实际使用的是另一种方法，这个公式对应着论文中p632左中位置的那个
        // Pr1 对应论文里的r_r,i',P3对应论文里的 r_l,i',(经过坐标系转换的Pr2), n=3, 剩下的就和论文中都一样了
        double nom = Pr1.dot(P3);
        // 准备计算分母
        cv::Mat aux_P3(P3.size(),P3.type());
        aux_P3=P3;
        // 先得到平方
        cv::pow(P3,2,aux_P3);
        double den = 0;

        // 然后再累加
        for(int i=0; i<aux_P3.rows; i++){
            for(int j=0; j<aux_P3.cols; j++){
                den+=aux_P3.at<float>(i,j);
            }
        }
        ms12i = nom/den;
    }
    else
        ms12i = 1.0f;

    // Step 7: 计算平移Translation
    mt12i.create(1,3,P1.type());
    // 论文中平移公式
    mt12i = O1 - ms12i*mR12i*O2;

    // Step 8: 计算双向变换矩阵，目的是在后面的检查的过程中能够进行双向的投影操作

    // Step 8.1 用尺度，旋转，平移构建变换矩阵 T12
    mT12i = cv::Mat::eye(4,4,P1.type());

    cv::Mat sR = ms12i*mR12i;

    //         |sR t|
    // mT12i = | 0 1|
    sR.copyTo(mT12i.rowRange(0,3).colRange(0,3));
    mt12i.copyTo(mT12i.rowRange(0,3).col(3));

    // Step 8.2 T21

    mT21i = cv::Mat::eye(4,4,P1.type());
    cv::Mat sRinv = (1.0/ms12i)*mR12i.t();
    sRinv.copyTo(mT21i.rowRange(0,3).colRange(0,3));
    cv::Mat tinv = -sRinv*mt12i;
    tinv.copyTo(mT21i.rowRange(0,3).col(3));
}

} //namespace ORB_SLAM
