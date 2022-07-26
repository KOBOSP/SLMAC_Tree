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


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<opencv2/core/core.hpp>
#include<System.h>
#include<unistd.h>

using namespace std;

/**
 * @brief 获取图像文件的信息
 * @param[in]  strImagePath     图像文件存放路径
 * @param[in]  strPathTimes     时间戳文件的存放路径
 * @param[out] vstrImages       图像文件名数组
 * @param[out] vTimeStamps      时间戳数组
 */
void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

//./Examples/Monocular/mono_euroc ../ORB3Vocabulary/ORBvoc.txt Examples/Monocular/EuRoC.yaml /media/kobosp/POCKET2/EuRoc/MH_05_difficult/mav0/cam0/data Examples/Monocular/EuRoC_TimeStamps/MH05.txt
int main(int argc, char **argv) {
    // step 0 检查输入参数个数是否足够
    if (argc != 5) {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_image_folder path_to_times_file"
             << endl;
        return 1;
    }

    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]),             // path_to_image_folder
               string(argv[4]),             // path_to_times_file
               vstrImageFilenames,          // 读取到的图像名称数组
               vTimestamps);                // 时间戳数组

    int nImages = vstrImageFilenames.size();

    if (nImages <= 0) {
        cerr << "ERROR: Failed to load images" << endl;
        return 1;
    }

    ORB_SLAM2::System SLAM(
            argv[1],
            argv[2],
            ORB_SLAM2::System::MONOCULAR,
            true);


    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    cv::Mat im;
    for (int ni = 0; ni < nImages; ni++) {
        im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (im.empty()) {
            cerr << endl << "Failed to load image at: "
                 << vstrImageFilenames[ni] << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        SLAM.TrackMonocular(im, tframe);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        vTimesTrack[ni] = ttrack;
    }

    // step 5 如果所有的图像都预测完了,那么终止当前的SLAM系统
    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    // step 6 计算平均耗时
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++) {
        totaltime += vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    // Save camera trajectory
    // step 7 保存TUM格式的相机轨迹
    // 估计是单目时有尺度漂移, 而LGA GBA都只能优化关键帧使尺度漂移最小, 普通帧所产生的轨迹漂移这里无能为力, 我猜作者这样就只
    // 保存了关键帧的位姿,从而避免普通帧带有尺度漂移的位姿对最终误差计算的影响
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    SLAM.SaveTrajectoryTUM("AllFrameTrajectory.txt");
    return 0;
}

// 从文件中加载图像序列中每一张图像的文件路径和时间戳
void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps) {
    // 打开文件
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    // 遍历文件
    while (!fTimes.eof()) {
        string s;
        getline(fTimes, s);
        // 只有在当前行不为空的时候执行
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            // 生成当前行所指出的RGB图像的文件名称
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            // 记录该图像的时间戳
            vTimeStamps.push_back(t / 1e9);

        }
    }
}
