#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <map>

using namespace cv;
using namespace std;

Mat quantizeColors(const Mat& src, int levelsPerChannel) {
    // 创建输入图像 src 的副本 quantized，以便在不改变原始图像的情况下进行颜色量化
    Mat quantized = src.clone();
    float scale = 256 / levelsPerChannel; // 计算量化比例
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // 访问图像 quantized 中位置 (i, j) 的像素
            Vec3b& pixel = quantized.at<Vec3b>(i, j);
            // 确保颜色值被量化到最近的整数倍 scale
            pixel[0] = floor(pixel[0] / scale) * scale;
            pixel[1] = floor(pixel[1] / scale) * scale;
            pixel[2] = floor(pixel[2] / scale) * scale;
        }
    }
    return quantized;
}

map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)> calculateHistogram(const Mat& img) {
    // 用于 map 的排序，确保颜色值在映射中是有序的
    auto comp = [](const Vec3b& a, const Vec3b& b) {
        return (a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1]) || (a[0] == b[0] && a[1] == b[1] && a[2] < b[2]);
        };
    // 存储颜色和它们出现次数的映射
    map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)> histogram(comp);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3b color = img.at<Vec3b>(i, j); // 访问图像 img 中位置 (i, j) 的像素
            histogram[color]++; // 更新直方图
        }
    }
    return histogram;
}

// Smoothing saliency values
map<Vec3b, float, bool(*)(const Vec3b&, const Vec3b&)> smoothSaliency(const map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)>& histogram) {
    map<Vec3b, float, bool(*)(const Vec3b&, const Vec3b&)> smoothedSaliency(histogram.key_comp());
    const float similarityThreshold = 200.0; // 颜色相似性阈值
    for (auto& outer : histogram) { // 遍历直方图的每个颜色
        float totalWeight = 0.0;
        float weightedSaliency = 0.0;
        for (auto& inner : histogram) {
            // 对于每个 inner 颜色，计算与 outer 颜色之间的距离 colorDistance
            float colorDistance = norm(outer.first - inner.first);
            // 如果颜色相似，将 inner 颜色的显著性值乘以其权重，加到 weightedSaliency 上，并累加到 totalWeight
            if (colorDistance < similarityThreshold) {
                float weight = similarityThreshold - colorDistance; // Weight inversely proportional to distance
                weightedSaliency += weight * inner.second; // 次数
                totalWeight += weight;
            }
        }
        if (totalWeight > 0) // 存在相似颜色的权重，可以计算加权平均显著性
            smoothedSaliency[outer.first] = weightedSaliency / totalWeight;
        else
            smoothedSaliency[outer.first] = outer.second;
    }
    return smoothedSaliency;
}

// Compute saliency map using the histogram
Mat computeSaliencyMap(const Mat& img, map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)> histogram) {
    Mat saliencyMap = Mat::zeros(img.size(), CV_32F); // 零矩阵，用于存储显著性值
    for (int i = 0; i < img.rows; i++) { // 循环遍历图像的每个像素
        for (int j = 0; j < img.cols; j++) {
            Vec3b color = img.at<Vec3b>(i, j);
            float saliency = 0.0f;
            for (auto& h : histogram) {
                // 计算当前像素颜色与直方图中颜色的距离
                float colorDistance = norm(color - h.first);
                // 距离越近，显著性值越高
                saliency += h.second * exp(-colorDistance);
            }
            saliencyMap.at<float>(i, j) = saliency;
        }
    }
    normalize(saliencyMap, saliencyMap, 0, 1, NORM_MINMAX);
    saliencyMap = 1.0 - saliencyMap; // 显著性高的区域为亮色
    return saliencyMap;
}

Mat computeSaliencyMap2(const Mat& img, map<Vec3b, float, bool(*)(const Vec3b&, const Vec3b&)> smoothedSaliency) {
    Mat saliencyMap = Mat::zeros(img.size(), CV_32F);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3b color = img.at<Vec3b>(i, j);
            saliencyMap.at<float>(i, j) = smoothedSaliency[color];
        }
    }
    normalize(saliencyMap, saliencyMap, 0, 1, NORM_MINMAX);
    saliencyMap = 1.0 - saliencyMap;
    return saliencyMap;
}

void applyHistogramEqualization(Mat& src) {
    if (src.channels() == 1) {
        // Apply histogram equalization directly on single channel
        equalizeHist(src, src);
    }
    else {
        // Convert to YCrCb color space if src is a color image
        Mat ycrcb;
        cvtColor(src, ycrcb, COLOR_BGR2YCrCb);
        // Split the channels
        vector<Mat> channels;
        split(ycrcb, channels);
        // Equalize the Y channel
        equalizeHist(channels[0], channels[0]);
        // Merge the channels and convert back to BGR color space
        merge(channels, ycrcb);
        cvtColor(ycrcb, src, COLOR_YCrCb2BGR);
    }
}

void displayHistogram(const map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)>& histogram) {
    int histSize = 256; // 直方图大小
    vector<int> r_hist(histSize, 0);
    vector<int> g_hist(histSize, 0);
    vector<int> b_hist(histSize, 0);

    for (const auto& pair : histogram) {
        const Vec3b& color = pair.first;
        int count = pair.second;
        r_hist[color[2]] += count; // 注意：OpenCV中颜色顺序为BGR
        g_hist[color[1]] += count;
        b_hist[color[0]] += count;
    }

    // 创建直方图画布
    int histWidth = 512, histHeight = 400;
    int binWidth = cvRound((double)histWidth / histSize);

    Mat histImage(histHeight, histWidth, CV_8UC3, Scalar(255, 255, 255));

    // 归一化直方图
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    // 绘制直方图
    for (int i = 1; i < histSize; i++) {
        line(histImage, Point(binWidth * (i - 1), histHeight - r_hist[i - 1]),
            Point(binWidth * i, histHeight - r_hist[i]),
            Scalar(0, 0, 255), 2, 8, 0);
        line(histImage, Point(binWidth * (i - 1), histHeight - g_hist[i - 1]),
            Point(binWidth * i, histHeight - g_hist[i]),
            Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(binWidth * (i - 1), histHeight - b_hist[i - 1]),
            Point(binWidth * i, histHeight - b_hist[i]),
            Scalar(255, 0, 0), 2, 8, 0);
    }

    imshow("Histogram", histImage);
    waitKey(0);
}

