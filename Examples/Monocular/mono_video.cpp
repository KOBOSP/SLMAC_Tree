#include "System.h"
#include <string>
#include <chrono>
#include <iostream>


using namespace std;

//Examples/Monocular/mono_video ./Vocabulary/ORBvoc.txt ./Examples/Monocular/mono_video.yaml ../long_time_tree.mp4 ../long_time_tree.txt
int main(int argc, char **argv) {
    cv::FileStorage fSettings(argv[2], cv::FileStorage::READ);
    int frame_width=fSettings["Camera.width"];
    int frame_height=fSettings["Camera.height"];
    double FrameInterval = fSettings["Camera.fps"];

    ORB_SLAM2::System orb_slam2(argv[1], argv[2], argv[4], ORB_SLAM2::System::MONOCULAR, true);
    cv::VideoCapture cap(argv[3]);    // change to 1 if you want to use USB camera.

    FrameInterval= 1 / FrameInterval;
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
        orb_slam2.TrackMonocular(FrameResized, FrameInterval * FrameID, FrameID);
        auto TimeEnd = chrono::system_clock::now();
        double TimePast = chrono::duration_cast<chrono::milliseconds>(TimeEnd - TimeStart).count() / 1000.0;
        int FramePast = TimePast / FrameInterval;
        while(FramePast > 0){
            cap >> frame;
            FrameID++;
            FramePast--;
        }
    }
    cv::waitKey(0);
    orb_slam2.Shutdown();
    return 0;
}

