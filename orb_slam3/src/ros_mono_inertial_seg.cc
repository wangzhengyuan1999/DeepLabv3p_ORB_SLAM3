#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Imu.h>

#include <opencv2/core/core.hpp>

#include <System.h>
#include <ImuTypes.h>
#include "segment_interface/SegmentImage.h"

using namespace std;

class ImuGrabber
{
public:
    ImuGrabber(){};
    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System *pSLAM, ImuGrabber *pImuGb, const bool bClahe, ros::ServiceClient *pclient) : mpSLAM(pSLAM), mpImuGb(pImuGb), mbClahe(bClahe), mpClient(pclient) {}

    void GrabImage(const sensor_msgs::ImageConstPtr &msg);
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);
    void SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> img0Buf;
    std::mutex mBufMutex;

    ORB_SLAM3::System *mpSLAM;
    ImuGrabber *mpImuGb;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));

    ros::ServiceClient *mpClient;
    segment_interface::SegmentImage srv;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ros_mono_inertial_seg");
    ros::NodeHandle nh;

    bool bEqual = false;

    if (argc < 3 || argc > 4)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM3 Mono_Inertial path_to_vocabulary path_to_settings [do_equalize]" << endl;
        ros::shutdown();
        return 1;
    }

    if (argc == 4)
        if ("true" == std::string(argv[3]))
            bEqual = true;

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_MONOCULAR, true);

    ros::ServiceClient client = nh.serviceClient<segment_interface::SegmentImage>("/segment_interface");
    ros::service::waitForService("/segment_interface");

    ImuGrabber imugb;
    ImageGrabber igb(&SLAM, &imugb, bEqual, &client);

    ros::Subscriber sub_imu = nh.subscribe("/camera/imu", 1000, &ImuGrabber::GrabImu, &imugb);
    ros::Subscriber sub_img0 = nh.subscribe("/camera/color/image_raw", 100, &ImageGrabber::GrabImage, &igb);

    std::thread sync_thread(&ImageGrabber::SyncWithImu, &igb);

    ros::spin();

    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    SLAM.Shutdown();

    return 0;
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr &img_msg)
{
    std::unique_lock<std::mutex> lock(mBufMutex);
    if (!img0Buf.empty())
    {
        img0Buf.pop();
    }
    img0Buf.push(img_msg);
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    cv_ptr = cv_bridge::toCvShare(img_msg);
    return cv_ptr->image.clone();
}

void ImageGrabber::SyncWithImu()
{
    while (1)
    {
        cv::Mat im;
        double tIm = 0;
        if (!img0Buf.empty() && !mpImuGb->imuBuf.empty())
        {
            tIm = img0Buf.front()->header.stamp.toSec();
            if (tIm > mpImuGb->imuBuf.back()->header.stamp.toSec())
                continue;
            {
                this->mBufMutex.lock();
                im = GetImage(img0Buf.front());
                img0Buf.pop();
                this->mBufMutex.unlock();
            }

            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            mpImuGb->mBufMutex.lock();
            if (!mpImuGb->imuBuf.empty())
            {
                vImuMeas.clear();
                while (!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec() <= tIm)
                {
                    double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
                    cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x, mpImuGb->imuBuf.front()->linear_acceleration.y, mpImuGb->imuBuf.front()->linear_acceleration.z);
                    cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z);
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                    mpImuGb->imuBuf.pop();
                }
            }
            mpImuGb->mBufMutex.unlock();
            if (mbClahe)
                mClahe->apply(im, im);


            vector<cv::Mat> masks;
            srv.request.image = *cv_bridge::CvImage(std_msgs::Header(), (CV_8UC3 == im.type() ? "bgr8" : "mono8"), im).toImageMsg();
            if (!mpClient->call(srv)) exit -1;
            for (sensor_msgs::Image mask : srv.response.segmentImage)
            {
                masks.push_back(cv_bridge::toCvCopy(mask)->image);
            }

            mpSLAM->TrackMonocular(masks, im, tIm, vImuMeas);
        }

        std::chrono::milliseconds tSleep(1);
        std::this_thread::sleep_for(tSleep);
    }
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    std::unique_lock<std::mutex> lock(mBufMutex);
    imuBuf.push(imu_msg);
}
