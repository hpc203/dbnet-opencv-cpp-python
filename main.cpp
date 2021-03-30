#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class DBNet
{
    public:
        DBNet(const float binaryThreshold = 0.3, const float polygonThreshold = 0.5, const float unclipRatio = 2.0, const int maxCandidates = 200)
        {
            this->net = readNet("DB_TD500_resnet50.onnx");
            this->binaryThreshold = binaryThreshold;
            this->polygonThreshold = polygonThreshold;
            this->unclipRatio = unclipRatio;
            this->maxCandidates = maxCandidates;
        }
        void detect(Mat& srcimg);
    private:
        float binaryThreshold;
        float polygonThreshold;
        float unclipRatio;
        int maxCandidates;
        Net net;
        float contourScore(const Mat& binary, const vector<Point>& contour);
        void unclip(const vector<Point2f>& inPoly, vector<Point2f> &outPoly);
};

void DBNet::detect(Mat& srcimg)
{
    int h = srcimg.rows;
    int w = srcimg.cols;
    Mat blob = blobFromImage(srcimg, 1/255.0, Size(736, 736), Scalar(122.67891434, 116.66876762, 104.00698793));
    this->net.setInput(blob);
    vector<Mat> outs;
    this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
    CV_Assert(outs.size() == 1);
    Mat binary = outs[0];

    // Threshold
    Mat bitmap;
    threshold(binary, bitmap, binaryThreshold, 255, THRESH_BINARY);
    // Scale ratio
    float scaleHeight = (float)(h) / (float)(binary.size[0]);
    float scaleWidth = (float)(w) / (float)(binary.size[1]);
    // Find contours
    vector< vector<Point> > contours;
    bitmap.convertTo(bitmap, CV_8UC1);
    findContours(bitmap, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // Candidate number limitation
    size_t numCandidate = min(contours.size(), (size_t)(maxCandidates > 0 ? maxCandidates : INT_MAX));
    vector<float> confidences;
    vector< vector<Point2f> > results;
    for (size_t i = 0; i < numCandidate; i++)
    {
        vector<Point>& contour = contours[i];

        // Calculate text contour score
        if (contourScore(binary, contour) < polygonThreshold)
            continue;

        // Rescale
        vector<Point> contourScaled; contourScaled.reserve(contour.size());
        for (size_t j = 0; j < contour.size(); j++)
        {
            contourScaled.push_back(Point(int(contour[j].x * scaleWidth),
                                          int(contour[j].y * scaleHeight)));
        }

        // Unclip
        RotatedRect box = minAreaRect(contourScaled);

        // minArea() rect is not normalized, it may return rectangles with angle=-90 or height < width
        const float angle_threshold = 60;  // do not expect vertical text, TODO detection algo property
        bool swap_size = false;
        if (box.size.width < box.size.height)  // horizontal-wide text area is expected
            swap_size = true;
        else if (fabs(box.angle) >= angle_threshold)  // don't work with vertical rectangles
            swap_size = true;
        if (swap_size)
        {
            swap(box.size.width, box.size.height);
            if (box.angle < 0)
                box.angle += 90;
            else if (box.angle > 0)
                box.angle -= 90;
        }

        Point2f vertex[4];
        box.points(vertex);  // order: bl, tl, tr, br
        vector<Point2f> approx;
        for (int j = 0; j < 4; j++)
            approx.emplace_back(vertex[j]);
        vector<Point2f> polygon;
        unclip(approx, polygon);
        results.push_back(polygon);
    }
    confidences = vector<float>(contours.size(), 1.0f);
    polylines(srcimg, results, true, Scalar(0, 255, 0), 2);
}

// According to https://github.com/MhLiao/DB/blob/master/structure/representers/seg_detector_representer.py (2020-10)
float DBNet::contourScore(const Mat& binary, const vector<Point>& contour)
{
    Rect rect = boundingRect(contour);
    int xmin = max(rect.x, 0);
    int xmax = min(rect.x + rect.width, binary.cols - 1);
    int ymin = max(rect.y, 0);
    int ymax = min(rect.y + rect.height, binary.rows - 1);

    Mat binROI = binary(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));

    Mat mask = Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
    vector<Point> roiContour;
    for (size_t i = 0; i < contour.size(); i++) {
        Point pt = Point(contour[i].x - xmin, contour[i].y - ymin);
        roiContour.push_back(pt);
    }
    vector<vector<Point>> roiContours = {roiContour};
    fillPoly(mask, roiContours, Scalar(1));
    float score = mean(binROI, mask).val[0];
    return score;
}

// According to https://github.com/MhLiao/DB/blob/master/structure/representers/seg_detector_representer.py (2020-10)
void DBNet::unclip(const vector<Point2f>& inPoly, vector<Point2f> &outPoly)
{
    float area = contourArea(inPoly);
    float length = arcLength(inPoly, true);
    float distance = area * unclipRatio / length;

    size_t numPoints = inPoly.size();
    vector<vector<Point2f>> newLines;
    for (size_t i = 0; i < numPoints; i++)
    {
        vector<Point2f> newLine;
        Point pt1 = inPoly[i];
        Point pt2 = inPoly[(i - 1) % numPoints];
        Point vec = pt1 - pt2;
        float unclipDis = (float)(distance / norm(vec));
        Point2f rotateVec = Point2f(vec.y * unclipDis, -vec.x * unclipDis);
        newLine.push_back(Point2f(pt1.x + rotateVec.x, pt1.y + rotateVec.y));
        newLine.push_back(Point2f(pt2.x + rotateVec.x, pt2.y + rotateVec.y));
        newLines.push_back(newLine);
    }

    size_t numLines = newLines.size();
    for (size_t i = 0; i < numLines; i++)
    {
        Point2f a = newLines[i][0];
        Point2f b = newLines[i][1];
        Point2f c = newLines[(i + 1) % numLines][0];
        Point2f d = newLines[(i + 1) % numLines][1];
        Point2f pt;
        Point2f v1 = b - a;
        Point2f v2 = d - c;
        float cosAngle = (v1.x * v2.x + v1.y * v2.y) / (norm(v1) * norm(v2));

        if( fabs(cosAngle) > 0.7 )
        {
            pt.x = (b.x + c.x) * 0.5;
            pt.y = (b.y + c.y) * 0.5;
        }
        else
        {
            float denom = a.x * (float)(d.y - c.y) + b.x * (float)(c.y - d.y) +
                          d.x * (float)(b.y - a.y) + c.x * (float)(a.y - b.y);
            float num = a.x * (float)(d.y - c.y) + c.x * (float)(a.y - d.y) + d.x * (float)(c.y - a.y);
            float s = num / denom;

            pt.x = a.x + s*(b.x - a.x);
            pt.y = a.y + s*(b.y - a.y);
        }
        outPoly.push_back(pt);
    }
}

int main()
{
    DBNet mynet(0.3, 0.5, 2.0, 200);
    string imgpath = "/home/wangbo/Desktop/data/yolo/license-plate-detect-recoginition/License-Plate-Detector-master/imgs/3.jpg";
    Mat srcimg = imread(imgpath);
    mynet.detect(srcimg);

    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);
    imshow(kWinName, srcimg);
    waitKey(0);
    destroyAllWindows();
}