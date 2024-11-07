#include <vector>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include "segment_interface/SegmentImage.h"


int main(int argc, char *argv[])
{
    ros::init(argc, argv, "test_node_cpp");
    ros::NodeHandle nh;
    ros::ServiceClient client = nh.serviceClient<segment_interface::SegmentImage>("/segment_interface");
    ros::service::waitForService("/segment_interface");
    segment_interface::SegmentImage req;
    cv::Mat image = cv::imread("~/DeepLabv3p_ORB_SLAM3/src/segment_interface/img/image2_1341846313.922055.png");
    req.request.image = *cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    client.call(req);
    for (sensor_msgs::Image mask : req.response.segmentImage)
    {
        cv::imshow("mask", cv_bridge::toCvCopy(mask, "mono8")->image);
        cv::waitKey();
    }
    

    return 0;
}
