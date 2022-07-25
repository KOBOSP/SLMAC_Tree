#include "System.h"
#include <string>
#include <chrono>
#include <iostream>
#include <sys/stat.h>


using namespace std;


/**
 * 判断是否是一个文件
 */
static bool IsFile(std::string filename) {
    struct stat buffer;
    return (stat (filename.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

/**
 * 判断是否是一个文件夹,
 * */
static bool IsDir(std::string filefodler) {
    struct stat buffer;
    return (stat (filefodler.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}


const double PI = 3.141592657;
const int EARTH_RADIUS = 6378137;

static inline double DegreeToRad(double degree){
    return  PI * degree / 180.0;
}
static inline double HaverSin(double x){
    double v = sin(x / 2.0);
    return v * v;
}
//计算距离(单位 : m). input: LonLatAlt(degree,degree,m) output distance(m)
static double GetLLADistance(double lon1, double lat1, double alt1, double lon2, double lat2, double alt2){
    double radlon1 = DegreeToRad(lon1);
    double radlat1 = DegreeToRad(lat1);
    double radlon2 = DegreeToRad(lon2);
    double radlat2 = DegreeToRad(lat2);

    double a = fabs(radlat1 - radlat2);
    double b = fabs(radlon1 - radlon2);

    double h = HaverSin(b) + cos(lat1) * cos(lat2) * HaverSin(a);
    double distance = 2 * EARTH_RADIUS * asin(sqrt(h));
    return sqrt(distance*distance+(alt1-alt2)*(alt1-alt2));
}

static double GetXYZDistance(double X1, double Y1, double Z1, double X2, double Y2, double Z2){
    return sqrt((X1-X2)*(X1-X2)+(Y1-Y2)*(Y1-Y2)+(Z1-Z2)*(Z1-Z2));
}


//Examples/Monocular/mono_video ./Vocabulary/ORBvoc.txt ./Examples/Monocular/mono_video.yaml ../long_time_tree.mp4 ../long_time_tree.txt
int main(int argc, char **argv) {
    cv::FileStorage fSettings(argv[2], cv::FileStorage::READ);
    int frame_width=fSettings["Camera.width"];
    int frame_height=fSettings["Camera.height"];
    int IsSeq=fSettings["Image.IsSeq"];
    double FrameInterval = fSettings["Camera.fps"];
    FrameInterval= 1 / FrameInterval;
    cv::Mat Trtk;
    auto TimeSystemInit = chrono::system_clock::now();

    ORB_SLAM2::System orb_slam2(argv[1], argv[2], argv[4], ORB_SLAM2::System::MONOCULAR, true);
    if(IsSeq==1){
        std::string sSeqPath = "/media/kobosp/POCKET3/dataset/slam数据集/RosarioDataset/sequence06/zed/*.png";
        std::string sGpsPath = "/media/kobosp/POCKET3/dataset/slam数据集/RosarioDataset/sequence06/gps.log";
        std::string sImageGpsPath = "/media/kobosp/POCKET3/dataset/slam数据集/RosarioDataset/sequence06/ImageGps.log";
        std::vector<cv::String> vsImageFile;
        cv::glob(sSeqPath, vsImageFile);
        cv::Mat frame;
        FILE * fpr= fopen (sGpsPath.c_str(), "r");
        FILE * fpw= fopen (sImageGpsPath.c_str(), "a+");
        int ntemp, nSVs;
        double dtemp1, dtemp2, dGpsTimeStamp=0, dImageTimeStamp, dLat, dLon, dHorDil, dHSL, dHgeo;
        char stemp[150], sMode[10];

        for(vector<cv::String>::iterator iter = vsImageFile.begin(), iend = vsImageFile.end();iter!=iend;iter++) {
            string sImageTS = (*iter).substr((*iter).length() - 21, 16);
            sscanf(sImageTS.c_str(), "%lf", &dImageTimeStamp);
            cv::namedWindow("Display Sequence");
            cv::Mat DisSeq=imread((*iter));
            imshow("Display Sequence",DisSeq);
            while (dGpsTimeStamp < dImageTimeStamp){
                if(fscanf(fpr, "%lf GPS-RTK: %s ", &dtemp1, sMode)==-1){
                    return 0;
                }
                if (sMode[4] != 'G') {
                    fscanf(fpr, "%[^\n]\n", stemp);
//                    printf("1-||%lf, ||%lf, ||%s， ||%s\n", dImageTimeStamp, dGpsTimeStamp, sMode, stemp);
                }
                else{
                    dGpsTimeStamp = dtemp1;
                    printf("%lf\r",dGpsTimeStamp);
                    fscanf(fpr, "%lf %lf S %lf W %d %d %lf %lf M %lf M %lf %s\n",&dtemp2,&dLat,&dLon,&ntemp,&nSVs,&dHorDil,&dHSL,&dHgeo,&dtemp2,stemp);
//                    printf("2-||%lf, ||%lf, ||%lf, ||%lf, ||%s\n", dImageTimeStamp, dGpsTimeStamp, dLat, dLon);
                }
            }
            fprintf(fpw, "%016.06lf %016.06lf %011.07lf %012.07lf %06.03lf %02d\n", dImageTimeStamp, dGpsTimeStamp, dLon, dLat, dHgeo, nSVs);
        }
        return 0;
        //            frame = cv::imread(*iter);
    }
    else if(IsSeq==2) {
        ///home/kobosp/SLAM_YOLO/ORB_SLAM2_v5.2_Optical_Flow/Vocabulary/ORBvoc.txt /home/kobosp/SLAM_YOLO/ORB_SLAM2_v5.2_Optical_Flow/Examples/Monocular/mono_video.yaml /media/kobosp/POCKET5/ScienceIsolandForest98/ /home/kobosp/SLAM_YOLO/empty.txt
        std::string sImageGpsPath = argv[3];
//        std::string sImageGpsPath = "/media/kobosp/POCKET5/ScienceIsolandForest/";
        int FrameID=1;
        char stemp[10];
        double dLat, dLon, dAlt;
        FILE * fpr= fopen ((sImageGpsPath+"AllFigLLA.txt").c_str(), "r");
        printf("%s\n",(sImageGpsPath+"Images/"+stemp).c_str());
        while(fscanf(fpr, "%s %lf %lf %lf\n", stemp, &dLon, &dLat, &dAlt)!=-1) {
            printf("%s\n",(sImageGpsPath+"Images/"+stemp).c_str());
            cv::Mat frame = cv::imread((sImageGpsPath+"Images/"+stemp).c_str());
            cv::Mat FrameResized;
            cv::resize(frame, FrameResized, cv::Size(frame_width, frame_height));
            Trtk = cv::Mat(3,1,CV_32F);
            Trtk.at<float>(0)=(dLon/360.0)*2.0*PI*EARTH_RADIUS*cos(DegreeToRad(dLat));
            Trtk.at<float>(1)=(dLat/360.0)*2.0*PI*EARTH_RADIUS;
            Trtk.at<float>(2)=dAlt;
            orb_slam2.TrackMonocular(FrameResized, FrameInterval * FrameID, FrameID++, Trtk);
        }
    }
    else if(IsSeq==0){
        ///home/kobosp/SLAM_YOLO/ORB_SLAM2_v5.2_Optical_Flow/Vocabulary/ORBvoc.txt /home/kobosp/SLAM_YOLO/ORB_SLAM2_v5.2_Optical_Flow/Examples/Monocular/mono_video.yaml /home/kobosp/SLAM_YOLO/5m20s740416p10fps.mp4 /home/kobosp/SLAM_YOLO/5m20s740416p10fps.txt
        cv::VideoCapture cap(argv[3]);    // change to 1 if you want to use USB camera.
        unsigned long FrameID=0;
        while (1) {
            cv::Mat frame;
            cap >> frame;
            ++FrameID;
            if (frame.data == nullptr)
                break;
            cv::Mat FrameResized;
            cv::resize(frame, FrameResized, cv::Size(frame_width, frame_height));
            auto TimeStart = chrono::system_clock::now();
            orb_slam2.TrackMonocular(FrameResized, chrono::duration_cast<chrono::milliseconds>(TimeStart- TimeSystemInit).count() / 1000.0, FrameID, Trtk);
            auto TimeEnd = chrono::system_clock::now();
            double TimePast = chrono::duration_cast<chrono::milliseconds>(TimeEnd - TimeStart).count() / 1000.0;
            int FramePast = TimePast / FrameInterval;
            while(FramePast > 1){
                cap >> frame;
                FrameID++;
                FramePast--;
            }
        }
    }

    cv::waitKey(0);
    orb_slam2.Shutdown();
    return 0;
}

