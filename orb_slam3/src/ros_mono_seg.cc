#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include <System.h>
#include "segment_interface/SegmentImage.h"

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System *pSLAM, ros::ServiceClient *pclient) : mpSLAM(pSLAM), mpClient(pclient) {}

    void GrabImage(const sensor_msgs::ImageConstPtr &msg);

    ORB_SLAM3::System *mpSLAM;
    ros::ServiceClient *mpClient;
    segment_interface::SegmentImage srv;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ros_mono_seg");
    ros::NodeHandle nh;

    if (argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM3 Mono path_to_vocabulary path_to_settings" << endl;
        ros::shutdown();
        return 1;
    }

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);

    ros::ServiceClient client = nh.serviceClient<segment_interface::SegmentImage>("/segment_interface");
    ros::service::waitForService("/segment_interface");

    ImageGrabber igb(&SLAM, &client);

    ros::Subscriber sub = nh.subscribe("/camera/color/image_raw", 1, &ImageGrabber::GrabImage, &igb);

    ros::spin();

    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    SLAM.Shutdown();

    return 0;
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr &msg)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    vector<cv::Mat> masks;
    srv.request.image = *cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_ptr->image).toImageMsg();
    if (!mpClient->call(srv)) exit -1;
    for (sensor_msgs::Image mask : srv.response.segmentImage)
    {
        masks.push_back(cv_bridge::toCvCopy(mask, "mono8")->image);
    }
    mpSLAM->TrackMonocular(masks, cv_ptr->image, cv_ptr->header.stamp.toSec());
}
