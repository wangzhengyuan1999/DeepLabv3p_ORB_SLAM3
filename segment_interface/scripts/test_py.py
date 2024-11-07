import cv2
import rospy
from cv_bridge import CvBridge
from segment_interface.srv import SegmentImage


def test_client():
    rospy.init_node('test_node_py')    
    rospy.wait_for_service('/segment_interface')
    bridge = CvBridge()
    image = cv2.imread('~/DeepLabv3p_ORB_SLAM3/src/segment_interface/img/image1_1341846313.592026.png')
    image_message = bridge.cv2_to_imgmsg(image, "bgr8")
    client = rospy.ServiceProxy('/segment_interface', SegmentImage)
    resp = client(image_message)
    for mask in resp.segmentImage:
        cv2.imshow('mask', bridge.imgmsg_to_cv2(mask, "mono8"))
        cv2.waitKey()

	
if __name__ == '__main__':
    test_client()
