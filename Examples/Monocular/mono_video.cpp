#include "System.h"
#include <string>
#include <chrono>
#include <iostream>


using namespace std;


const double PI = 3.141592657;
const int EARTH_RADIUS = 6378137;//meter
static inline double DegreeToRad(double degree){
    return  PI * degree / 180.0;
}
static inline double RadToDegree(double rad){
    return  rad / PI * 180.0;
}

//1-1024+1025-1342
///home/kobosp/SLAM_YOLO/ORB_SLAM2_v6_GPS_fuse/Vocabulary/ORBvoc.txt /home/kobosp/SLAM_YOLO/ORB_SLAM2_v6_GPS_fuse/Examples/Monocular/mono_video.yaml /media/kobosp/downloads/Downloads/ScienceIsolandForest98/
///home/kobosp/SLAM_YOLO/ORB_SLAM2_v6_GPS_fuse/Vocabulary/ORBvoc.txt /home/kobosp/SLAM_YOLO/ORB_SLAM2_v6_GPS_fuse/Examples/Monocular/mono_video.yaml /media/kobosp/downloads/Downloads/20220212DongHeSuccessImage/
///home/kobosp/SLAM_YOLO/ORB_SLAM2_v6_GPS_fuse/Vocabulary/ORBvoc.txt /home/kobosp/SLAM_YOLO/ORB_SLAM2_v6_GPS_fuse/Examples/Monocular/mono_video.yaml /media/kobosp/downloads/20220224科学岛树林30m照片0.5ms/
///home/kobosp/SLAM_YOLO/ORB_SLAM2_v6_GPS_fuse/Vocabulary/ORBvoc.txt /home/kobosp/SLAM_YOLO/ORB_SLAM2_v6_GPS_fuse/Examples/Monocular/mono_video.yaml /media/kobosp/POCKET2/SelfMakeTreeDataset/0.5ms2s/
int main(int argc, char **argv) {
    cv::FileStorage fSettings(argv[2], cv::FileStorage::READ);
    int frame_width=fSettings["Camera.width"];
    int frame_height=fSettings["Camera.height"];
    int nFrameNum=fSettings["Image.FrameNum"];
    int bSaveKeyFramesGps = fSettings["Map.SaveKeyFrameGps"];
    int bSaveObjectsGps = fSettings["Map.bSaveObjectsGps"];
    auto TimeSystemInit = chrono::steady_clock::now();
    cv::Mat TgpsFrame, TgpsFirst;
    TgpsFirst = cv::Mat(3, 1, CV_32F);
    TgpsFrame = cv::Mat(3, 1, CV_32F);

    ORB_SLAM2::System orb_slam2(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR, true);
    std::string sDetectImageLoLaAtPath = argv[3];

    long unsigned int nFrameID = 1, nTargetID, nMaxTarInFrame=60;
    int tmpTargetX, tmpTargetY, tmpTargetL, tmpTargetH, tmpConfid;
    vector<cv::KeyPoint> vTarsInFrame;
    vTarsInFrame.resize(nMaxTarInFrame);
    vector<vector<vector<int> > > vvTarsAllFrame=vector<vector<vector<int>>>(nFrameNum+1,vector<vector<int>>());
    FILE * fprTarget= fopen ((sDetectImageLoLaAtPath + "compound.txt").c_str(), "r");
    while(fscanf(fprTarget, "%d %d %d %d %d %d %d %d %d %d\n",&nFrameID, &nTargetID, &tmpTargetX, &tmpTargetY, &tmpTargetL, &tmpTargetH, &tmpConfid, &tmpConfid, &tmpConfid, &tmpConfid)!=-1){
        int dNowTarRad=sqrt(pow(tmpTargetL,2)+pow(tmpTargetH,2))/3.0;//cute the bargin
        vector<int>tmp;
        tmp.emplace_back(nTargetID);
        tmp.emplace_back(tmpTargetX+tmpTargetL/2);
        tmp.emplace_back(tmpTargetY+tmpTargetH/2);
        tmp.emplace_back(dNowTarRad);
        vvTarsAllFrame[nFrameID].emplace_back(tmp);
    }
    fclose(fprTarget);

    nFrameID=1;
    cv::Mat FrameOrigin, FrameResized;
    double dLon, dLat, dAlt;
    FILE * fprgps= fopen ((sDetectImageLoLaAtPath + "LoLaAt.txt").c_str(), "r");
    while(fscanf(fprgps, "%lf %lf %lf\n", &dLon, &dLat, &dAlt) != -1) {
        if(nFrameID==nFrameNum){
            if(bSaveKeyFramesGps||bSaveObjectsGps){
                orb_slam2.SaveKeyFrameAndMapPointInGps(sDetectImageLoLaAtPath + "MapPointAndKeyFrame.txt",bSaveKeyFramesGps,bSaveObjectsGps);
            }
            break;
        }
        if(nFrameID==1){
            TgpsFirst.at<float>(0)= dLon;
            TgpsFirst.at<float>(1)= dLat;
            TgpsFirst.at<float>(2)= dAlt;
        }
        TgpsFrame.at<float>(0)=DegreeToRad(TgpsFirst.at<float>(0)-dLon)*EARTH_RADIUS*cos(DegreeToRad(dLat));
        TgpsFrame.at<float>(1)=DegreeToRad(TgpsFirst.at<float>(1)-dLat)*EARTH_RADIUS;
        TgpsFrame.at<float>(2)=TgpsFirst.at<float>(2)-dAlt;

        for(int i=0;i<vvTarsAllFrame[nFrameID].size();i++){
            vTarsInFrame.emplace_back(cv::KeyPoint(cv::Point2f(vvTarsAllFrame[nFrameID][i][1], vvTarsAllFrame[nFrameID][i][2]),
                                                   vvTarsAllFrame[nFrameID][i][3],
                                                   tmpConfid,
                                                   0,
                                                   0,
                                                   int(vvTarsAllFrame[nFrameID][i][0])));
        }

        FrameOrigin = cv::imread((sDetectImageLoLaAtPath + "images/" + to_string(nFrameID) + ".jpg").c_str());
        cv::resize(FrameOrigin, FrameResized, cv::Size(frame_width, frame_height));
        auto TimeStart = chrono::steady_clock::now();
        orb_slam2.TrackMonocular(FrameResized, chrono::duration_cast<chrono::milliseconds>(TimeStart - TimeSystemInit).count() / 1000.0, nFrameID++, vTarsInFrame, TgpsFrame);
        vTarsInFrame.clear();
    }
    fclose(fprgps);

//    auto TimeSystemEnd = std::chrono::steady_clock::now();
//    std::chrono::duration<double> SysSpent = TimeSystemEnd-TimeSystemInit;
//    std::cout <<"System end in "<< SysSpent.count() << " seconds \n";
    cv::waitKey(0);
    orb_slam2.Shutdown();
    return 0;
}