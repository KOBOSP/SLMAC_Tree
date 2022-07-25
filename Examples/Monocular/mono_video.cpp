

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
    auto TimeSystemInit = chrono::system_clock::now();
    cv::Mat TgpsFrame, TgpsFirst;
    TgpsFirst = cv::Mat(3, 1, CV_32F);
    TgpsFrame = cv::Mat(3, 1, CV_32F);

    ORB_SLAM2::System orb_slam2(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR, true);
    std::string sDetectImageLoLaAtPath = argv[3];
    long unsigned int nFrameID = 1, nTargetID = 1, nClassID, nAngDiv=8;
    double dLon, dLat, dAlt;
    FILE * fprgps= fopen ((sDetectImageLoLaAtPath + "LoLaAt.txt").c_str(), "r");
    double tmpTargetX, tmpTargetY, tmpTargetL, tmpTargetH, tmpConfid;
    vector<cv::KeyPoint> vTarsInFrame;
    vTarsInFrame.resize(50);
    cv::Mat FrameOrigin, FrameResized;
    vector<vector<double> > vvdLastFrameTargetPos;
    vvdLastFrameTargetPos.reserve(70);
    vector<vector<double> > vvdNowFrameTargetPos;//cenX,cenY,Rad,Id
    vvdNowFrameTargetPos.reserve(70);
    while(fscanf(fprgps, "%lf %lf %lf\n", &dLon, &dLat, &dAlt) != -1) {
        if(nFrameID==1288){
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

        FILE * fprTarget= fopen ((sDetectImageLoLaAtPath + "targets/" + to_string(nFrameID) + ".txt").c_str(), "r");
        while(fscanf(fprTarget, "%d %lf %lf %lf %lf %lf\n", &nClassID, &tmpTargetX, &tmpTargetY, &tmpTargetL, &tmpTargetH, &tmpConfid)!=-1){
            double dNowTarRad=sqrt(pow(tmpTargetL*frame_width,2)+pow(tmpTargetH*frame_height,2))/2.0;
            vector<double>tmp;
            tmp.emplace_back(tmpTargetX*frame_width);
            tmp.emplace_back(tmpTargetY*frame_height);
            tmp.emplace_back(dNowTarRad);
            tmp.emplace_back(-1);
            vvdNowFrameTargetPos.emplace_back(tmp);
        }
        vector<vector<int> > vnMatchId(vvdNowFrameTargetPos.size(),vector<int>(nAngDiv,-1));
        vector<vector<double> > vnMatchDis(vvdNowFrameTargetPos.size(),vector<double>(nAngDiv,999));
        vector<int> vnMatchAng(nAngDiv,0);
        for(int i=0;i<vvdNowFrameTargetPos.size();i++) {
            double DiffX,DiffY,DiffDis,Angle;
            int tmpAng;
            for(int j=0;j<vvdLastFrameTargetPos.size();j++){
                DiffX=vvdNowFrameTargetPos[i][0]-vvdLastFrameTargetPos[j][0];
                DiffY=vvdNowFrameTargetPos[i][1]-vvdLastFrameTargetPos[j][1];
                DiffDis=sqrt(pow(DiffX,2) + pow(DiffY,2));
                Angle=RadToDegree(atan(DiffY/DiffX))+((DiffX<0)?270:90);
                tmpAng=Angle/(360/nAngDiv);
                if(DiffDis<vvdNowFrameTargetPos[i][2]*2){
                    vnMatchAng[tmpAng]++;
                    if(DiffDis<vnMatchDis[i][tmpAng]) {
                        vnMatchId[i][tmpAng] = j;
                        vnMatchDis[i][tmpAng] = DiffDis;
                    }
                }
            }
        }
        int nFirst=-1,nSecond=-1,nCandidId;
        for(int i=0;i<nAngDiv;i++){
            if(vnMatchAng[i]>nFirst){
                nSecond=nFirst;
                nFirst=vnMatchAng[i];
                nCandidId=i;
            }
            else if(vnMatchAng[i]>nSecond){
                nSecond=vnMatchAng[i];
            }
        }
        if(nFirst/2<nSecond){
            cout<<"nFirst/2<nSecond "<<nFirst/2<<" "<<nSecond<<endl;
        }
        for(int i=0;i<vvdNowFrameTargetPos.size();i++) {
            if(vnMatchId[i][nCandidId]==-1){
                vvdNowFrameTargetPos[i][3]=nTargetID++;
            }
            else{
                vvdNowFrameTargetPos[i][3]=vnMatchId[i][nCandidId];
            }
        }
        for(int i=0;i<vvdNowFrameTargetPos.size();i++){
            cout<<"vvdNowFrameTargetPos[i][0] "<<vvdNowFrameTargetPos[i][0]<<" "<<nCandidId<<" "<<vvdNowFrameTargetPos[i][1]<<" "<<vvdNowFrameTargetPos[i][2]<<" "<<vvdNowFrameTargetPos[i][3]<<endl;
            vTarsInFrame.emplace_back(cv::KeyPoint(cv::Point2f(vvdNowFrameTargetPos[i][0], vvdNowFrameTargetPos[i][1]),
                                                   vvdNowFrameTargetPos[i][2],
                                                   tmpConfid,
                                                   0,
                                                   0,
                                                   int(vvdNowFrameTargetPos[i][3])));
        }
        cout<<"vvdLastFrameTargetPos.size()<<\" \"<<vvdNowFrameTargetPos.size() "<<vvdLastFrameTargetPos.size()<<" "<<vvdNowFrameTargetPos.size()<<endl;
        vvdLastFrameTargetPos.assign(vvdNowFrameTargetPos.begin(),vvdNowFrameTargetPos.end());
        vvdNowFrameTargetPos.clear();
        FrameOrigin = cv::imread((sDetectImageLoLaAtPath + "images/" + to_string(nFrameID) + ".jpg").c_str());
        cv::resize(FrameOrigin, FrameResized, cv::Size(frame_width, frame_height));
        auto TimeStart = chrono::system_clock::now();
        orb_slam2.TrackMonocular(FrameResized, chrono::duration_cast<chrono::milliseconds>(TimeStart- TimeSystemInit).count() / 1000.0, nFrameID++, vTarsInFrame, TgpsFrame);
        vTarsInFrame.clear();
        fclose(fprTarget);
    }
    fclose(fprgps);
    cv::waitKey(0);
    orb_slam2.Shutdown();
    return 0;
}

