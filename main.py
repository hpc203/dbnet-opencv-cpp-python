import cv2
import argparse
import numpy as np
import pyclipper
from shapely.geometry import Polygon

class DetectorDecoder:
    def __init__(self, thresh=0.3, box_thresh=0.5, max_candidates=200, unclip_ratio=2.0, min_box_size=3):
        self.min_size = min_box_size
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, pred, height, width):
        segmentation = self.binarize(pred)
        boxes, scores = self.boxes_from_bitmap(pred, segmentation, width, height)
        return boxes, scores

    def binarize(self, pred):
        return pred > self.thresh

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(bitmap.shape) == 2
        # bitmap = _bitmap.cpu().numpy()  # The first channel
        # pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        label_img = (bitmap * 255).astype(np.uint8)
        contours, _ = cv2.findContours(label_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)
        # label_points = list()
        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue
            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
            # points_tmp = contour
            # points_tmp[:, 0] = np.clip(np.round(points_tmp[:, 0] / width * dest_width), 0, dest_width)
            # points_tmp[:, 1] = np.clip(np.round(points_tmp[:, 1] / height * dest_height), 0, dest_height)
            # label_points.append(points_tmp)
        return boxes, scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        # 计算轮廓所包含的面积
        th_ss = cv2.contourArea(contour)
        # 计算轮廓的周长
        # th_ss = cv2.arcLength(contour, True)

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        # return box, min(bounding_box[1])
        return box, th_ss

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

class dbnet:
    def __init__(self, binaryThreshold=0.3, polygonThreshold=0.5, unclipRatio=2.0, maxCandidates=200):
        self.model = cv2.dnn.readNet('DB_TD500_resnet50.onnx')
        self.decode_handel = DetectorDecoder(thresh=binaryThreshold, box_thresh=polygonThreshold, max_candidates=maxCandidates, unclip_ratio=unclipRatio)
    def detect(self, srcimg):
        h, w = srcimg.shape[:2]
        blob = cv2.dnn.blobFromImage(srcimg, scalefactor=1 / 255.0, size=(736, 736),
                                     mean=(122.67891434, 116.66876762, 104.00698793))
        self.model.setInput(blob)
        preb = self.model.forward()
        box_list, score_list = self.decode_handel(preb, h, w)
        if len(box_list) > 0:
            idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
            box_list, score_list = box_list[idx], score_list[idx]
        else:
            box_list, score_list = [], []
        for point in box_list:
            point = point.astype(int)
            cv2.polylines(srcimg, [point + 1], True, (0, 0, 255), thickness=1)
            for i in range(4):
                cv2.circle(srcimg, tuple(point[i, :]), 3, (0, 255, 0), thickness=-1)
        return srcimg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RetinaPL')
    parser.add_argument('--imgpath', default='imgs/3.jpg', type=str, help='image path')
    parser.add_argument('--binaryThreshold', default=0.3, type=float, help='binary Threshold')
    parser.add_argument('--polygonThreshold', default=0.5, type=float, help='polygon Threshold')
    parser.add_argument('--unclipRatio', default=2.0, type=float, help='unclip Ratio')
    parser.add_argument('--maxCandidates', default=200, type=int, help='max Candidates')
    args = parser.parse_args()

    net = dbnet(binaryThreshold=args.binaryThreshold, polygonThreshold=args.polygonThreshold, unclipRatio=args.unclipRatio, maxCandidates=args.maxCandidates)
    srcimg = cv2.imread(args.imgpath)
    srcimg = net.detect(srcimg)
    #cv2.imwrite('result.jpg', srcimg)
    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    cv2.imshow('detect', srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
