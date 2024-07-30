#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <map>

using namespace cv;
using namespace std;

Mat quantizeColors(const Mat& src, int levelsPerChannel) {
    // ��������ͼ�� src �ĸ��� quantized���Ա��ڲ��ı�ԭʼͼ�������½�����ɫ����
    Mat quantized = src.clone();
    float scale = 256 / levelsPerChannel; // ������������
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // ����ͼ�� quantized ��λ�� (i, j) ������
            Vec3b& pixel = quantized.at<Vec3b>(i, j);
            // ȷ����ɫֵ������������������� scale
            pixel[0] = floor(pixel[0] / scale) * scale;
            pixel[1] = floor(pixel[1] / scale) * scale;
            pixel[2] = floor(pixel[2] / scale) * scale;
        }
    }
    return quantized;
}

map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)> calculateHistogram(const Mat& img) {
    // ���� map ������ȷ����ɫֵ��ӳ�����������
    auto comp = [](const Vec3b& a, const Vec3b& b) {
        return (a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1]) || (a[0] == b[0] && a[1] == b[1] && a[2] < b[2]);
        };
    // �洢��ɫ�����ǳ��ִ�����ӳ��
    map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)> histogram(comp);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3b color = img.at<Vec3b>(i, j); // ����ͼ�� img ��λ�� (i, j) ������
            histogram[color]++; // ����ֱ��ͼ
        }
    }
    return histogram;
}

// Smoothing saliency values
map<Vec3b, float, bool(*)(const Vec3b&, const Vec3b&)> smoothSaliency(const map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)>& histogram) {
    map<Vec3b, float, bool(*)(const Vec3b&, const Vec3b&)> smoothedSaliency(histogram.key_comp());
    const float similarityThreshold = 200.0; // ��ɫ��������ֵ
    for (auto& outer : histogram) { // ����ֱ��ͼ��ÿ����ɫ
        float totalWeight = 0.0;
        float weightedSaliency = 0.0;
        for (auto& inner : histogram) {
            // ����ÿ�� inner ��ɫ�������� outer ��ɫ֮��ľ��� colorDistance
            float colorDistance = norm(outer.first - inner.first);
            // �����ɫ���ƣ��� inner ��ɫ��������ֵ������Ȩ�أ��ӵ� weightedSaliency �ϣ����ۼӵ� totalWeight
            if (colorDistance < similarityThreshold) {
                float weight = similarityThreshold - colorDistance; // Weight inversely proportional to distance
                weightedSaliency += weight * inner.second; // ����
                totalWeight += weight;
            }
        }
        if (totalWeight > 0) // ����������ɫ��Ȩ�أ����Լ����Ȩƽ��������
            smoothedSaliency[outer.first] = weightedSaliency / totalWeight;
        else
            smoothedSaliency[outer.first] = outer.second;
    }
    return smoothedSaliency;
}

// Compute saliency map using the histogram
Mat computeSaliencyMap(const Mat& img, map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)> histogram) {
    Mat saliencyMap = Mat::zeros(img.size(), CV_32F); // ��������ڴ洢������ֵ
    for (int i = 0; i < img.rows; i++) { // ѭ������ͼ���ÿ������
        for (int j = 0; j < img.cols; j++) {
            Vec3b color = img.at<Vec3b>(i, j);
            float saliency = 0.0f;
            for (auto& h : histogram) {
                // ���㵱ǰ������ɫ��ֱ��ͼ����ɫ�ľ���
                float colorDistance = norm(color - h.first);
                // ����Խ����������ֵԽ��
                saliency += h.second * exp(-colorDistance);
            }
            saliencyMap.at<float>(i, j) = saliency;
        }
    }
    normalize(saliencyMap, saliencyMap, 0, 1, NORM_MINMAX);
    saliencyMap = 1.0 - saliencyMap; // �����Ըߵ�����Ϊ��ɫ
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
    int histSize = 256; // ֱ��ͼ��С
    vector<int> r_hist(histSize, 0);
    vector<int> g_hist(histSize, 0);
    vector<int> b_hist(histSize, 0);

    for (const auto& pair : histogram) {
        const Vec3b& color = pair.first;
        int count = pair.second;
        r_hist[color[2]] += count; // ע�⣺OpenCV����ɫ˳��ΪBGR
        g_hist[color[1]] += count;
        b_hist[color[0]] += count;
    }

    // ����ֱ��ͼ����
    int histWidth = 512, histHeight = 400;
    int binWidth = cvRound((double)histWidth / histSize);

    Mat histImage(histHeight, histWidth, CV_8UC3, Scalar(255, 255, 255));

    // ��һ��ֱ��ͼ
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    // ����ֱ��ͼ
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

