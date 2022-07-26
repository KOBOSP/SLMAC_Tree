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


/**
 * @file  Converter.h
 * @author guoqing (1337841346@qq.com)
 * @brief 提供了一系列的常见转换。\n
 * orb中以cv::Mat为基本存储结构，到g2o和Eigen需要一个转换。
 * 这些转换都很简单，整个文件可以单独从orbslam里抽出来而不影响其他功能.
 * @version 0.1
 * @date 2019-01-03
 */

#ifndef CONVERTER_H
#define CONVERTER_H

#include<opencv2/core/core.hpp>

#include<Eigen/Dense>
#include"g2o/types/types_six_dof_expmap.h"
#include"g2o/types/types_seven_dof_expmap.h"

#include "sophus/geometry.hpp"
#include "sophus/sim3.hpp"
/**
 * @brief ORB-SLAM2 自定义的命名空间。 
 * @details 该命名空间中包含了所有的ORB-SLAM2的组件。
 * 
 */
namespace ORB_SLAM2 {

/**
 * @brief 实现了 ORB-SLAM2中的一些常用的转换。 
 * @details 注意这是一个完全的静态类，没有成员变量，所有的成员函数均为静态的。
 */
    class Converter {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

        static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);

        static g2o::SE3Quat toSE3Quat(const Sophus::SE3f &T);

        static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);

        // TODO templetize these functions
        static cv::Mat toCvMat(const g2o::SE3Quat &SE3);

        static cv::Mat toCvMat(const g2o::Sim3 &Sim3);

        static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4> &m);

        static cv::Mat toCvMat(const Eigen::Matrix<float, 4, 4> &m);

        static cv::Mat toCvMat(const Eigen::Matrix<float, 3, 4> &m);

        static cv::Mat toCvMat(const Eigen::Matrix3d &m);

        static cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1> &m);

        static cv::Mat toCvMat(const Eigen::Matrix<float, 3, 1> &m);

        static cv::Mat toCvMat(const Eigen::Matrix<float, 3, 3> &m);

        static cv::Mat toCvMat(const Eigen::MatrixXf &m);

        static cv::Mat toCvMat(const Eigen::MatrixXd &m);

        static void tosRt(const Eigen::Matrix<float, 4, 4> &Sim3T, float &s, Eigen::Matrix<float, 3, 3> &R,
                          Eigen::Matrix<float, 3, 1> &t);

        static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t);

        static cv::Mat tocvSkewMatrix(const cv::Mat &v);

        static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Mat &cvVector);

        static Eigen::Matrix<float, 3, 1> toVector3f(const cv::Mat &cvVector);

        static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Point3f &cvPoint);

        static Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat &cvMat3);

        static Eigen::Matrix<double, 4, 4> toMatrix4d(const cv::Mat &cvMat4);

        static Eigen::Matrix<float, 3, 3> toMatrix3f(const cv::Mat &cvMat3);

        static Eigen::Matrix<float, 4, 4> toMatrix4f(const cv::Mat &cvMat4);

        static std::vector<float> toQuaternion(const cv::Mat &M);

        static bool isRotationMatrix(const cv::Mat &R);

        static std::vector<float> toEuler(const cv::Mat &R);

        //TODO: Sophus migration, to be deleted in the future
        static Sophus::SE3<float> toSophus(const cv::Mat &T);

        static Sophus::SE3<float> toSophus(const cv::Mat &cvR, const cv::Mat &cvt);

        static Sophus::Sim3f toSophus(const g2o::Sim3 &S);

        static Sophus::Sim3f toSophus(const cv::Mat &S, bool bIsSim3);
    };

}// namespace ORB_SLAM

#endif // CONVERTER_H
