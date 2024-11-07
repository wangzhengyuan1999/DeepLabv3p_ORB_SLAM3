import os
import sys
import numpy as np
import rospy
from cv_bridge import CvBridge
from segment_interface.srv import SegmentImage, SegmentImageResponse
import cv2
import torch
from ultralytics import YOLO
from torchvision.transforms import transforms
sys.path.append(os.path.split(__file__)[0])
from models import deeplabv3plus
from datasets.data_loader_voc import *


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


def deeplabv3plus_predict(image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = deeplabv3plus.deeplabv3plussc_resnest50()
    model = model.to(device)
    input = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input = normalize(to_tensor(input)).unsqueeze(0)
    input = input.to(device)
    model.eval()
    output = model(image).argmax(dim=1)
    return output


model_name = 'yolo11x-seg.pt'
model_path = os.path.join(os.path.split(__file__)[0], 'ultralytics_model', model_name)
model = YOLO(model_path)


def yolo_predict(image):
    results = model.predict(image, retina_masks=True, classes=[0])
    result = results[0]
    if result.masks:
        res = result.masks.data.cpu().numpy()
        return res * 255


def call_back(req):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(req.image, "bgr8")
    masks = yolo_predict(cv_image)
    if masks is None:
        masks = np.zeros((1, cv_image.shape[0], cv_image.shape[1]))
    return SegmentImageResponse([bridge.cv2_to_imgmsg(mask.astype(np.uint8), "mono8") for mask in masks])


def segment_server():
    rospy.init_node('segment_node')
    rospy.Service('/segment_interface', SegmentImage, call_back)
    rospy.spin()


if __name__ == '__main__':
    segment_server()
