import numpy as np


class SegmentationMetric(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        # L\P   P   N
        #   P   TP  FN
        #   N   FP  TN

    def add_batch(self, label, predict):
        assert predict.shape == label.shape
        self.confusion_matrix += self.get_confusion_matrix(label, predict)

    def get_confusion_matrix(self, label, predict):
        mask = (label >= 0) & (label < self.num_classes)
        return np.bincount(self.num_classes * label[mask] + predict[mask], minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
 
    def pixel_accuracy(self):
        # 准确率 (Accuracy) PA = (TP + TN) / (TP + TN + FP + TN)
        return np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
 
    def class_pixel_accuracy(self):
        # 精准率 (Precision) cPA = TP / (TP + FP)
        return np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0) # ！网上这里的 axis 都为 1，不对
 
    def mean_pixel_accuracy(self):
        return np.nanmean(self.class_pixel_accuracy())
    
    def intersection_over_union(self):
        # IoU = TP / (TP + FP + FN)
        return np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix))  
 
    def mean_intersection_over_union(self):
        return np.nanmean(self.intersection_over_union())
    
    def frequency_weighted_intersection_over_union(self):
        # FWIoU = [(TP + FN) / (TP + FP + TN + FN)] * [TP / (TP + FP + FN)]
        freq = self.confusion_matrix.sum(axis=1) / self.confusion_matrix.sum()
        iou = self.intersection_over_union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def get_results(self):
        return {
            'PA': self.pixel_accuracy(),
            'cPA': self.class_pixel_accuracy(),
            'mPA': self.mean_pixel_accuracy(),
            'IoU': self.intersection_over_union(),
            'mIoU': self.mean_intersection_over_union(),
            'FWIoU': self.frequency_weighted_intersection_over_union()
        }


if __name__ ==  '__main__':
    # metric = SegmentationMetric(3)
    # pred = np.array([[1, 0, 2], [2, 2, 1], [0, 1, 0]])
    # lbl = np.array([[0, 1, 0], [1, 2, 0], [0, 1, 2]])
    # metric.add_batch(lbl, pred)
    # print(metric.confusion_matrix)
    # print(metric.get_results())

    # from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
    # print(confusion_matrix(lbl.reshape(9), pred.reshape(9)))
    # print(accuracy_score(lbl.reshape(9), pred.reshape(9)))
    # print(precision_score(lbl.reshape(9), pred.reshape(9), average=None))
    metric = SegmentationMetric(21)
    import torch
    metric.add_batch(torch.zeros((1000, 513, 513), dtype=torch.long).numpy(), torch.zeros((1000, 513, 513), dtype=torch.long).numpy())
