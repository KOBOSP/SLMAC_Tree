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




//Examples/Monocular/mono_video ./Vocabulary/ORBvoc.txt ./Examples/Monocular/mono_video.yaml ../long_time_tree.mp4 ../long_time_tree.txt
int main(int argc, char **argv) {
    cv::FileStorage fSettings(argv[2], cv::FileStorage::READ);
    int frame_width=fSettings["Camera.width"];
    int frame_height=fSettings["Camera.height"];
    int IsSeq=fSettings["Image.IsSeq"];
    double FrameInterval = fSettings["Camera.fps"];
    FrameInterval= 1 / FrameInterval;
    cv::Mat Trtk = cv::Mat::eye(4,4,CV_32F);
    auto TimeSystemInit = chrono::system_clock::now();

    ORB_SLAM2::System orb_slam2(argv[1], argv[2], argv[4], ORB_SLAM2::System::MONOCULAR, true);

    std::string sSeqPath = "/media/kobosp/POCKET3/dataset/slam数据集/RosarioDataset/sequence06/zed/*.png";
    std::string sGpsPath = "/media/kobosp/POCKET3/dataset/slam数据集/RosarioDataset/sequence06/gps.log";
    std::string sImageGpsPath = "/media/kobosp/POCKET3/dataset/slam数据集/RosarioDataset/sequence06/ImageGps.log";

    if(IsSeq==1){
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
        int nSVs, FrameID=1;
        double dGpsTimeStamp, dImageTimeStamp, dLat, dLon, dHgeo;
        char stemp[20];
        FILE * fpr= fopen (sImageGpsPath.c_str(), "r");
        while(fscanf(fpr, "%lf %lf %lf %lf %lf %d\n", &dImageTimeStamp, &dGpsTimeStamp, &dLon, &dLat, &dHgeo, &nSVs)!=-1) {
            sprintf(stemp, "%016.06lf", dImageTimeStamp);
            string sImagePath = sSeqPath.substr(0, sSeqPath.length() - 5) + "left_" + stemp + ".png";
            printf("%s\r",sImagePath.c_str());
            cv::Mat frame = cv::imread(sImagePath);
            cv::Mat FrameResized;
            cv::resize(frame, FrameResized, cv::Size(frame_width, frame_height));

            double* Trtkptr = Trtk.ptr<double>(0);
            Trtkptr[0]=dLon;
            Trtkptr = Trtk.ptr<double>(1);
            Trtkptr[0]=dLat;
            Trtkptr = Trtk.ptr<double>(2);
            Trtkptr[0]=dHgeo;
            // 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到相机坐标系的变换矩阵
//            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
//            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            orb_slam2.TrackMonocular(FrameResized, FrameInterval * FrameID, FrameID++, Trtk);

        }
    }
    else if(IsSeq==0){
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

