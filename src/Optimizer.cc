/**
 * @file Optimizer.cc
 * @author guoqing (1337841346@qq.com)
 * @brief 优化器，所有的优化函数的实现
 * @version 0.1
 * @date 2019-05-22
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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include<mutex>

namespace ORB_SLAM2
{

/**
 * @brief 全局BA： pMap中所有的MapPoints和关键帧做bundle adjustment优化
 * 这个全局BA优化在本程序中有两个地方使用：
 * 1、单目初始化：CreateInitialMapMonocular函数
 * 2、闭环优化：RunGlobalBundleAdjustment函数
 * @param[in] pMap                  地图点
 * @param[in] nIterations           迭代次数
 * @param[in] pbStopFlag            外部控制BA结束标志
 * @param[in] nLoopKF               形成了闭环的当前关键帧的id
 * @param[in] bRobust               是否使用鲁棒核函数
 */
void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    // 获取地图中的所有关键帧
    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    // 获取地图中的所有地图点
    vector<MapPoint*> vpMP = pMap->GetAllMapPoints(true);
    // 调用GBA
    OptimizeAllKFsAndMPs(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
}

/**
 * @brief bundle adjustment 优化过程
 * 1. Vertex: g2o::VertexSE3Expmap()，即当前帧的Tcw
 *            g2o::VertexSBAPointXYZ()，MapPoint的mWorldPos
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + Vertex：待优化MapPoint的mWorldPos
 *         + measurement：MapPoint在当前帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 * 
 * @param[in] vpKFs                 参与BA的所有关键帧
 * @param[in] vpMP                  参与BA的所有地图点
 * @param[in] nIterations           优化迭代次数
 * @param[in] pbStopFlag            外部控制BA结束标志
 * @param[in] nLoopKF               形成了闭环的当前关键帧的id
 * @param[in] bRobust               是否使用核函数
 */
void Optimizer::OptimizeAllKFsAndMPs(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                     int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    // 不参与优化的地图点
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    // Step 1 初始化g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    // 使用LM算法优化
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // 如果这个时候外部请求终止，那就结束
    // 注意这句执行之后，外部再请求结束BA，就结束不了了
    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    // 记录添加到优化器中的顶点的最大关键帧id
    long unsigned int maxKFid = 0;

    // Step 2 向优化器添加顶点

    // Set KeyFrame vertices
    // Step 2.1 ：向优化器添加关键帧位姿顶点
    // 遍历当前地图中的所有关键帧
    for(size_t i=0; i<vpKFs.size(); i++){
        KeyFrame* pKF = vpKFs[i];
        // 跳过无效关键帧
        if(pKF->isBad())
            continue;
        // 对于每一个能用的关键帧构造SE3顶点,其实就是当前关键帧的位姿
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        // 顶点的id就是关键帧在所有关键帧中的id
        vSE3->setId(pKF->mnId);
        // 只有第0帧关键帧不优化（参考基准）
        vSE3->setFixed(pKF->mnId == 0);
        // 向优化器中添加顶点，并且更新maxKFid
        optimizer.addVertex(vSE3);
        if(pKF->mnId > maxKFid)
            maxKFid=pKF->mnId;
    }

    // 卡方分布 95% 以上可信度的时候的阈值
    const float thHuber2D = sqrt(5.99);     // 自由度为2
    const float thHuber3D = sqrt(7.815);    // 自由度为3

    // Set MapPoint vertices
    // Step 2.2：向优化器添加地图点作为顶点
    // 遍历地图中的所有地图点
    for(size_t i=0; i<vpMP.size(); i++){
        MapPoint* pMP = vpMP[i];
        // 跳过无效地图点
//        if(pMP->GetbBad() || pMP->mnObjectID > 0){
//            continue;
//        }
        if(pMP->GetbBad()){
            continue;
        }

        // 创建顶点
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        // 注意由于地图点的位置是使用cv::Mat数据类型表示的,这里需要转换成为Eigen::Vector3d类型
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        // 前面记录maxKFid 是在这里使用的
        const int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        // 注意g2o在做BA的优化时必须将其所有地图点全部schur掉，否则会出错。
        // 原因是使用了g2o::LinearSolver<BalBlockSolver::PoseMatrixType>这个类型来指定linearsolver,
        // 其中模板参数当中的位姿矩阵类型在程序中为相机姿态参数的维度，于是BA当中schur消元后解得线性方程组必须是只含有相机姿态变量。
        // Ceres库则没有这样的限制
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        // 取出地图点和关键帧之间观测的关系
        const map<KeyFrame*,size_t> observations = pMP->GetObservationsKFAndMPIdx();

        // 边计数
        int nEdges = 0;
        //SET EDGES
        // Step 3：向优化器添加投影边（是在遍历地图点、添加地图点的顶点的时候顺便添加的）
        // 遍历观察到当前地图点的所有关键帧
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++){
            KeyFrame* pKF = mit->first;
            // 跳过不合法的关键帧
            if(pKF->isBad() || pKF->mnId > maxKFid)
                continue;
            nEdges++;
            // 取出该地图点对应该关键帧的2D特征点
            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];
            // 以下是单目相机模式：
            // 构造观测
            Eigen::Matrix<double,2,1> obs;
            obs << kpUn.pt.x, kpUn.pt.y;
            // 创建边
            g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
            // 边连接的第0号顶点对应的是第id个地图点
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
            // 边连接的第1号顶点对应的是第id个关键帧
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
            e->setMeasurement(obs);
            // 信息矩阵，也是协方差，表明了这个约束的观测在各个维度（x,y）上的可信程度，在我们这里对于具体的一个点，两个坐标的可信程度都是相同的，
            // 其可信程度受到特征点在图像金字塔中的图层有关，图层越高，可信度越差
            // 为了避免出现信息矩阵中元素为负数的情况，这里使用的是sigma^(-2)
            const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
            // 使用鲁棒核函数
            if(bRobust){
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                // 这里的重投影误差，自由度为2，所以这里设置为卡方分布中自由度为2的阈值，如果重投影的误差大约大于1个像素的时候，就认为不太靠谱的点了，
                // 核函数是为了避免其误差的平方项出现数值上过大的增长
                rk->setDelta(thHuber2D);
            }

            // 设置相机内参
            e->fx = pKF->fx;
            e->fy = pKF->fy;
            e->cx = pKF->cx;
            e->cy = pKF->cy;
            // 添加边
            optimizer.addEdge(e);
        } // 向优化器添加投影边,也就是遍历所有观测到当前地图点的关键帧

        // 如果因为一些特殊原因,实际上并没有任何关键帧观测到当前的这个地图点,那么就删除掉这个顶点,并且这个地图点也就不参与优化
        if(nEdges==0){
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else{
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    // Step 4：开始优化
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data
    // Step 5：得到优化的结果

    // Step 5.1 遍历所有的关键帧
    for(size_t i=0; i<vpKFs.size(); i++){
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        // 获取到优化后的位姿
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if(nLoopKF==0){
            // 原则上来讲不会出现"当前闭环关键帧是第0帧"的情况,如果这种情况出现,只能够说明是在创建初始地图点的时候调用的这个全局BA函数.
            // 这个时候,地图中就只有两个关键帧,其中优化后的位姿数据可以直接写入到帧的成员变量中
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else{
            // 正常的操作,先把优化后的位姿写入到帧的一个专门的成员变量mTcwGBA中备用
            pKF->mTcwGBA.create(4,4,CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
        }
    }

    // Step 5.2 Points
    // 遍历所有地图点,去除其中没有参与优化过程的地图点
    for(size_t i=0; i<vpMP.size(); i++){
        if(vbNotIncludedMP[i])
            continue;
        MapPoint* pMP = vpMP[i];
        // 跳过无效地图点
//        if(pMP->GetbBad() || pMP->mnObjectID > 0){
//            continue;
//        }
        if(pMP->GetbBad()){
            continue;
        }
        // 获取优化之后的地图点的位置
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        // 和上面对关键帧的操作一样
        if(nLoopKF==0){
            // 如果这个GBA是在创建初始地图的时候调用的话,那么地图点的位姿也可以直接写入
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else{
            // 反之,如果是正常的闭环过程调用,就先临时保存一下
            pMP->mPosGBA.create(3,1,CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
        }// 判断是因为什么原因调用的GBA
    } // 遍历所有地图点,保存优化之后地图点的位姿
}

/**
 * @brief Pose Only Optimization
 * 
 * 3D-2D 最小化重投影误差 e = (u,v) - project(Tcw*Pw) \n
 * 只优化Frame的Tcw，不优化MapPoints的坐标
 * 
 * 1. Vertex: g2o::VertexSE3Expmap()，即当前帧的Tcw
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZOnlyPose()，BaseUnaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + measurement：MapPoint在当前帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeStereoSE3ProjectXYZOnlyPose()，BaseUnaryEdge
 *         + Vertex：待优化当前帧的Tcw
 *         + measurement：MapPoint在当前帧中的二维位置(ul,v,ur)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *
 * @param   pFrame Frame
 * @return  inliers数量
 */
int Optimizer::OptimizeFramePose(Frame *pFrame, double ErrorAddFactor)
{
    // 该优化函数主要用于Tracking线程中：运动跟踪、参考帧跟踪、地图跟踪、重定位

    // Step 1：构造g2o优化器, BlockSolver_6_3表示：位姿 _PoseDim 为6维，路标点 _LandmarkDim 是3维
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // 输入的帧中,有效的,参与优化过程的2D-3D点对
    int nInitialCorrespondences=0;

    // Set Frame vertex
    // Step 2：添加顶点：待优化当前帧的Tcw
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
     // 设置id
    vSE3->setId(0);    
    // 要优化的变量，所以不能固定
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->mnKeyPointNum;

    // for Monocular
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);


    // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
    const double deltaMono = sqrt(5.991);

    // Step 3：添加一元边
    {
    // 锁定地图点。由于需要使用地图点来构造顶点和边,因此不希望在构造的过程中部分地图点被改写造成不一致甚至是段错误
//    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    // 遍历当前地图中的所有地图点
    for(int i=0; i<N; i++){
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        // 如果这个地图点还存在没有被剔除掉
        if(pMP){
            // 跳过无效地图点
            if(pMP->GetbBad()){
                continue;
            }
            // Monocular observation
            nInitialCorrespondences++;
            pFrame->mvbOutlier[i] = false;

            // 对这个地图点的观测
            Eigen::Matrix<double,2,1> obs;
            const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
            obs << kpUn.pt.x, kpUn.pt.y;
            // 新建单目的边，一元边，误差为观测特征点坐标减去投影点的坐标
            g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
            // 设置边的顶点
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(obs);
            // 这个点的可信程度和特征点所在的图层有关
            const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
            // 在这里使用了鲁棒核函数
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono);    // 前面提到过的卡方阈值

            // 设置相机内参
            e->fx = pFrame->fx;
            e->fy = pFrame->fy;
            e->cx = pFrame->cx;
            e->cy = pFrame->cy;
            // 地图点的空间位置,作为迭代的初始值
            cv::Mat Xw = pMP->GetWorldPos();
            e->Xw[0] = Xw.at<float>(0);
            e->Xw[1] = Xw.at<float>(1);
            e->Xw[2] = Xw.at<float>(2);

            optimizer.addEdge(e);
            vpEdgesMono.emplace_back(e);
            vnIndexEdgeMono.emplace_back(i);
        }
    }
    } // 离开临界区

    // 如果没有足够的匹配点,那么就只好放弃了
    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    // Step 4：开始优化，总共优化四次，每次优化迭代10次,每次优化后，将观测分为outlier和inlier，outlier不参与下次优化
    // 由于每次优化后是对所有的观测进行outlier和inlier判别，因此之前被判别为outlier有可能变成inlier，反之亦然
    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
//    const float chi2Mono[4]={5.991,5.991,5.991,5.991};          // 单目
    const double chi2Mono[4]={5.991*(1+ErrorAddFactor),5.991*(1+ErrorAddFactor*2/3),5.991*(1+ErrorAddFactor*1/2),5.991*(1+ErrorAddFactor*1/3)};
    const int its[4]={10,10,10,10};// 四次迭代，每次迭代的次数

    // bad 的地图点个数
    int nBad=0;
    // 一共进行四次优化
    for(size_t it=0; it<4; it++){
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        // 其实就是初始化优化器,这里的参数0就算是不填写,默认也是0,也就是只对level为0的边进行优化
        optimizer.initializeOptimization(0);
        // 开始优化，优化10次
        optimizer.optimize(its[it]);

        nBad=0;
        // 优化结束,开始遍历参与优化的每一条误差边(单目)
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++){
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];
            const size_t idx = vnIndexEdgeMono[i];
            // 如果这条误差边是来自于outlier
            if(pFrame->mvbOutlier[idx]){
                e->computeError(); 
            }
            // 就是error*\Omega*error,表征了这个点的误差大小(考虑置信度以后)
            const float chi2 = e->chi2();
            if(chi2>chi2Mono[it]){
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);                 // 设置为outlier , level 1 对应为外点,上面的过程中我们设置其为不优化
                nBad++;
            }
            else{
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);                 // 设置为inlier, level 0 对应为内点,上面的过程中我们就是要优化这些关系
            }

            if(it==2){
                e->setRobustKernel(0); // 除了前两次优化需要RobustKernel以外, 其余的优化都不需要 -- 因为重投影的误差已经有明显的下降了
            }
        } // 对单目误差边的处理
        if(optimizer.edges().size()<10)
            break;
    } // 一共要进行四次优化

    // Recover optimized pose and return number of inliers
    // Step 5 得到优化后的当前帧的位姿
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetTcwPose(pose);

    // 并且返回内点数目
    return nInitialCorrespondences-nBad;
}

/**
 * @brief Local Bundle Adjustment
 *
 * 1. Vertex:
 *     - g2o::VertexSE3Expmap()，LocalKeyFrames，即当前关键帧的位姿、与当前关键帧相连的关键帧的位姿
 *     - g2o::VertexSE3Expmap()，FixedCameras，即能观测到LocalMapPoints的关键帧（并且不属于LocalKeyFrames）的位姿，在优化中这些关键帧的位姿不变
 *     - g2o::VertexSBAPointXYZ()，LocalMapPoints，即LocalKeyFrames能观测到的所有MapPoints的位置
 * 2. Edge:
 *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *     - g2o::EdgeStereoSE3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Tcw，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(ul,v,ur)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *         
 * @param pKF        KeyFrame
 * @param pbStopFlag 是否停止优化的标志
 * @param pMap       在优化后，更新状态时需要用到Map的互斥量mMutexMapUpdate
 * @note 由局部建图线程调用,对局部地图进行优化的函数
 */
void Optimizer::OptimizeLocalMapPoint(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{
    // 该优化函数用于LocalMapping线程的局部BA优化

    // Local KeyFrames: First Breadth Search from Current Keyframe
    // 局部关键帧
    list<KeyFrame*> lLocalKeyFrames;

    // Step 1 将当前关键帧及其共视关键帧加入局部关键帧
    lLocalKeyFrames.emplace_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    // 找到关键帧连接的共视关键帧（一级相连），加入局部关键帧中
    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    for(int i=0, iend=vNeighKFs.size(); i<iend; i++){
        KeyFrame* pKFi = vNeighKFs[i];
        if(!pKFi->isBad()){
            // 把参与局部BA的每一个关键帧的 mnBALocalForKF设置为当前关键帧的mnId，防止重复添加
            pKFi->mnBALocalForKF = pKF->mnId;
            // 保证该关键帧有效才能加入
            lLocalKeyFrames.emplace_back(pKFi);
        }
    }

    // Local MapPoints seen in Local KeyFrames
    // Step 2 遍历局部关键帧中(一级相连)关键帧，将它们观测的地图点加入到局部地图点
    list<MapPoint*> lLocalMapPoints;
    // 遍历局部关键帧中的每一个关键帧
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++){
        // 取出该关键帧对应的地图点
        vector<MapPoint*> vpMPs = (*lit)->GetAllMapPointVectorInKF(false);
        // 遍历这个关键帧观测到的每一个地图点，加入到局部地图点
        for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++){
            MapPoint* pMP = *vit;
            if(pMP){
                if(!pMP->GetbBad())   //保证地图点有效
                    // 把参与局部BA的每一个地图点的 mnBALocalForKF设置为当前关键帧的mnId
                    // mnBALocalForKF 是为了防止重复添加
                    if(pMP->mnBALocalForKF!=pKF->mnId){
                        lLocalMapPoints.emplace_back(pMP);
                        pMP->mnBALocalForKF=pKF->mnId;
                    }
            }   // 判断这个地图点是否靠谱
        } // 遍历这个关键帧观测到的每一个地图点
    } // 遍历 lLocalKeyFrames 中的每一个关键帧

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    // Step 3 得到能被局部地图点观测到，但不属于局部关键帧的关键帧(二级相连)，这些二级相连关键帧在局部BA优化时不优化
    list<KeyFrame*> lFixedCameras;
    // 遍历局部地图中的每个地图点
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++){
        // 观测到该地图点的KF和该地图点在KF中的索引
        map<KeyFrame*,size_t> observations = (*lit)->GetObservationsKFAndMPIdx();
        // 遍历所有观测到该地图点的关键帧
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++){
            KeyFrame* pKFi = mit->first;
            // pKFi->mnBALocalForKF!=pKF->mnId 表示不属于局部关键帧，
            // pKFi->mnBAFixedForKF!=pKF->mnId 表示还未标记为fixed（固定的）关键帧
            if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId && !pKFi->isBad()){
                // 将局部地图点能观测到的、但是不属于局部BA范围的关键帧的mnBAFixedForKF标记为pKF（触发局部BA的当前关键帧）的mnId
                pKFi->mnBAFixedForKF=pKF->mnId;
                lFixedCameras.emplace_back(pKFi);
            }
        }
    }

    // Setup optimizer
    // Step 4 构造g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // 外界设置的停止优化标志
    // 可能在 Tracking::NeedNewKeyFrame() 里置位
    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    // 记录参与局部BA的最大关键帧mnId
    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    // Step 5 添加待优化的位姿顶点：局部关键帧的位姿
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++){
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        // 设置初始优化位姿
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        // 如果是初始关键帧，要锁住位姿不优化
        vSE3->setFixed(pKFi->mnId == 0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId > maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    // Step  6 添加不优化的位姿顶点：固定关键帧的位姿，注意这里调用了vSE3->setFixed(true)
    // 不优化为啥也要添加？回答：为了增加约束信息
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++){
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
         // 所有的这些顶点的位姿都不优化，只是为了增加约束项
        vSE3->setFixed(true);  
        optimizer.addVertex(vSE3);
        if(pKFi->mnId > maxKFid){
            maxKFid=pKFi->mnId;
        }
    }

    // Set MapPoint vertices
    // Step  7 添加待优化的局部地图点顶点
    // 边的最大数目 = 位姿数目 * 地图点数目
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);
    vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);
    vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
    const float thHuberMono = sqrt(5.991);


    // 遍历所有的局部地图点
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        // 添加顶点：MapPoint
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        // 前面记录maxKFid的作用在这里体现
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        // 因为使用了LinearSolverType，所以需要将所有的三维点边缘化掉
        vPoint->setMarginalized(true);  
        optimizer.addVertex(vPoint);

        // 观测到该地图点的KF和该地图点在KF中的索引
        const map<KeyFrame*,size_t> observations = pMP->GetObservationsKFAndMPIdx();

        // Set edges
        // Step 8 在添加完了一个地图点之后, 对每一对关联的地图点和关键帧构建边
        // 遍历所有观测到当前地图点的关键帧
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++){
            KeyFrame* pKFi = mit->first;
            if(pKFi->isBad()||(pKFi->mnBAFixedForKF!=pKF->mnId && pKFi->mnBALocalForKF != pKF->mnId)){
                continue;
            }
            const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];
            // 根据单目/双目两种不同的输入构造不同的误差边
            // Monocular observation
            // 单目模式下
            Eigen::Matrix<double,2,1> obs;
            obs << kpUn.pt.x, kpUn.pt.y;

            g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
            // 边的第一个顶点是地图点
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
            // 边的第一个顶点是观测到该地图点的关键帧
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
            e->setMeasurement(obs);
            // 权重为特征点所在图像金字塔的层数的倒数
            const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

            // 使用鲁棒核函数抑制外点
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(thHuberMono);

            e->fx = pKFi->fx;
            e->fy = pKFi->fy;
            e->cx = pKFi->cx;
            e->cy = pKFi->cy;
            // 将边添加到优化器，记录边、边连接的关键帧、边连接的地图点信息
            optimizer.addEdge(e);
            vpEdgesMono.emplace_back(e);
            vpEdgeKFMono.emplace_back(pKFi);
            vpMapPointEdgeMono.emplace_back(pMP);
        } // 遍历所有观测到当前地图点的关键帧
    } // 遍历所有的局部地图中的地图点

    // 开始BA前再次确认是否有外部请求停止优化，因为这个变量是引用传递，会随外部变化
    // 可能在 Tracking::NeedNewKeyFrame(), mpLocalMapper->InsertKeyFrameInQueue 里置位
    if(pbStopFlag)
        if(*pbStopFlag)
            return;

    // Step 9 分成两个阶段开始优化。
    // 第一阶段优化
    optimizer.initializeOptimization();
    // 迭代5次
    optimizer.optimize(5);  

    bool bDoMore= true;

    // 检查是否外部请求停止
    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    // 如果有外部请求停止,那么就不在进行第二阶段的优化
    if(bDoMore){
        // Check inlier observations
        // Step 10 检测outlier，并设置下次不优化
        // 遍历所有的单目误差边
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++){
            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];
            // 跳过无效地图点
            if(pMP->GetbBad()){
                continue;
            }
            // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
            // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
            // 如果 当前边误差超出阈值，或者边链接的地图点深度值为负，说明这个边有问题，不优化了。
            if(e->chi2()>5.991 || !e->isDepthPositive()){
                // 不优化
                e->setLevel(1);
            }
            // 第二阶段优化的时候就属于精求解了,所以就不使用核函数
            e->setRobustKernel(0);
        }

        // Optimize again without the outliers
        // Step 11：排除误差较大的outlier后再次优化 -- 第二阶段优化
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size());

    // Check inlier observations
    // Step 12：在优化后重新计算误差，剔除连接误差比较大的关键帧和地图点
    // 对于单目误差边
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++){
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];
        // 跳过无效地图点
        if(pMP->GetbBad()){
            continue;
        }
        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
        // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值5.991
        // 如果 当前边误差超出阈值，或者边链接的地图点深度值为负，说明这个边有问题，要删掉了
        if(e->chi2()>5.991 || !e->isDepthPositive()){
            // outlier
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.emplace_back(make_pair(pKFi,pMP));
        }
    }


    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // 删除点
    // 连接偏差比较大，在关键帧中剔除对该地图点的观测
    // 连接偏差比较大，在地图点中剔除对该关键帧的观测
    if(!vToErase.empty()){
        for(size_t i=0;i<vToErase.size();i++){
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;

            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data
    // Step 13：优化后更新关键帧位姿以及地图点的位置、平均观测方向等属性

    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}



} //namespace ORB_SLAM
