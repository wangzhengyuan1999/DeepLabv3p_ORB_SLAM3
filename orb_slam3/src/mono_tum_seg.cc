#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include <System.h>
#include "segment_interface/SegmentImage.h"

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mono_tum_seg");
    ros::NodeHandle nh;
    ros::ServiceClient client = nh.serviceClient<segment_interface::SegmentImage>("/segment_interface");
    ros::service::waitForService("/segment_interface");
    segment_interface::SegmentImage srv;

    if (argc != 4)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3]) + "/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);
    int nImages = vstrImageFilenames.size();
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);
    cout << endl << "-------" << endl << "Start processing sequence ..." << endl << "Images in the sequence: " << nImages << endl << endl;
    double t_resize = 0.f, t_track = 0.f;

    cv::Mat im;
    for (int ni = 0; ni < nImages; ni++)
    {
        im = cv::imread(string(argv[3]) + "/" + vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (im.empty())
        {
            cerr << endl << "Failed to load image at: " << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

        if (imageScale != 1.f)
        {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
        }

        vector<cv::Mat> masks;
        srv.request.image = *cv_bridge::CvImage(std_msgs::Header(), "bgr8", im).toImageMsg();
        if (!client.call(srv)) return 1;
        for (sensor_msgs::Image mask : srv.response.segmentImage)
            masks.push_back(cv_bridge::toCvCopy(mask, "mono8")->image);
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        SLAM.TrackMonocular(masks, im, tframe);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;
        double T = 0;
        if (ni < nImages - 1) T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0) T = tframe - vTimestamps[ni - 1];

        if (ttrack < T) usleep((T - ttrack) * 1e6);
    }

    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++) totaltime += vTimesTrack[ni];
    cout << "-------" << endl << endl << "median tracking time: " << vTimesTrack[nImages / 2] << endl << "mean tracking time: " << totaltime / nImages << endl;
    
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    SLAM.Shutdown();

    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    string s0;
    getline(f, s0);
    getline(f, s0);
    getline(f, s0);

    while (!f.eof())
    {
        string s;
        getline(f, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}
