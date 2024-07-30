#ifndef __MYGRABCUT_H__
#define __MYGRABCUT_H__

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "graph.h"
#include "opencv2/imgproc/detail/gcgraph.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <limits>

using namespace std;
using namespace cv;
using namespace detail;

class ColorHistogram {
private:
    map<Vec3b, float, bool(*)(const Vec3b&, const Vec3b&)> data;
    int totalPixels;
    static const int levelsPerChannel = 20;  // ��������
    map<Vec3b, float, bool(*)(const Vec3b&, const Vec3b&)> fgProbabilities; // ǰ������
    map<Vec3b, float, bool(*)(const Vec3b&, const Vec3b&)> bgProbabilities; // ��������

    // �ȽϺ�����ȷ����ɫ��map�а�˳��洢
    static bool colorComp(const Vec3b& a, const Vec3b& b) {
        return (a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1]) || (a[0] == b[0] && a[1] == b[1] && a[2] < b[2]);
    }

public:
    ColorHistogram() : totalPixels(0), data(colorComp), fgProbabilities(colorComp), bgProbabilities(colorComp) {}

    // ��ɫ��������
    static Mat quantizeColors(const Mat& src) {
        Mat quantized = src.clone();
        // 12��12��12=1728 �ֲ�ͬ����ɫ
        float scale = 256.0f / levelsPerChannel; // 21.33
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                Vec3b& pixel = quantized.at<Vec3b>(i, j);
                // pixel=150, floor(pixel[0]/scale)=7, 7��21.33=149.31, 150 ����������� 149
                pixel[0] = floor(pixel[0] / scale) * scale; 
                pixel[1] = floor(pixel[1] / scale) * scale;
                pixel[2] = floor(pixel[2] / scale) * scale;
            }
        }
        return quantized;
    }

    // ����ֱ��ͼ
    map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)> calculateHistogram(const Mat& img) {
        Mat quantized = quantizeColors(img);
        map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)> histogram(colorComp);
        for (int i = 0; i < quantized.rows; i++) {
            for (int j = 0; j < quantized.cols; j++) {
                Vec3b color = quantized.at<Vec3b>(i, j); // ����ͼ�� img ��λ�� (i, j) ������
                histogram[color]++; // ����ֱ��ͼ
            }
        }
        return histogram;
    }

    // ƽ�������Լ���
    void smoothSaliency(const map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)>& histogram) {
        const float similarityThreshold = 200.0f;
        map<Vec3b, float, bool(*)(const Vec3b&, const Vec3b&)> smoothedData(histogram.key_comp());

        for (auto& outer : histogram) {
            float totalWeight = 0.0f;
            float weightedSaliency = 0.0f;
            for (auto& inner : histogram) {
                /* 
                ���������ɫ�ľ���С�������ֵ�����Ǿͱ���Ϊ�����Ƶ�
                ���������ɫ�Ĳ���ϴ�������ɫ�ʿռ��еľ���ͻ��Զ���Ӷ���������֮���Ȩ�ظ�С����Ϊ��
                ����ζ���ڽ���������ƽ������ʱ���˴˾����Զ����ɫ�������ụ��Ӱ��
                ��ˣ������ɫ�������ʹ����Щ��ɫ���Ӿ��ϸ����������ֿ�����������ͼ�������ͻ����ͬ������������ 
                */
                float colorDistance = norm(outer.first - inner.first);
                if (colorDistance < similarityThreshold) {
                    float weight = similarityThreshold - colorDistance;
                    //float weight = exp(-colorDistance / 10.0);
                    weightedSaliency += weight * inner.second;
                    totalWeight += weight;
                }
            }
            if (totalWeight > 0)
                smoothedData[outer.first] = weightedSaliency / totalWeight;
            else
                smoothedData[outer.first] = outer.second;
        }
        data = smoothedData;
    }

    // ��ȡ��ɫ�ĸ���
    float getProbability(const Vec3b& color) const {
        auto it = data.find(color);
        if (it != data.end()) {
            return it->second / static_cast<float>(totalPixels);
        }
        return 1e-5f;
    }

    // ��ʼ�������ǰ���ͱ�������
    void assignAndUpdateProbabilities(const Mat& img, const Mat& mask) {
        map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)> colorCounts(colorComp); // ʹ���Զ���ȽϺ���
        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                Vec3b color = img.at<Vec3b>(y, x);
                uchar maskValue = mask.at<uchar>(y, x);
                if (maskValue == GC_PR_FGD || maskValue == GC_FGD) {
                    fgProbabilities[color]++;
                }
                else if (maskValue == GC_BGD || maskValue == GC_PR_BGD) {
                    bgProbabilities[color]++;
                }
                colorCounts[color]++;
            }
        }
        // Update probabilities based on counts
        for (auto& entry : colorCounts) {
            Vec3b color = entry.first;
            if (colorCounts[color] > 0) { // Avoid division by zero
                if (fgProbabilities.find(color) != fgProbabilities.end()) {
                    fgProbabilities[color] /= colorCounts[color];
                }
                if (bgProbabilities.find(color) != bgProbabilities.end()) {
                    bgProbabilities[color] /= colorCounts[color];
                }
            }
        }
    }
};

static double calcBetaBasedHis(const Mat& img) {
    double beta = 0;
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            Vec3b color = img.at<Vec3b>(y, x);
            if (x > 0) { // left
                Vec3b diff = color - img.at<Vec3b>(y, x - 1);
                beta += norm(diff);
            }
            if (y > 0) { // up
                Vec3b diff = color - img.at<Vec3b>(y - 1, x);
                beta += norm(diff);
            }
            if (y > 0 && x > 0) { // upleft
                Vec3b diff = color - img.at<Vec3b>(y - 1, x - 1);
                beta += norm(diff);
            }
            if (y > 0 && x < img.cols - 1) { // upright
                Vec3b diff = color - img.at<Vec3b>(y - 1, x + 1);
                beta += norm(diff);
            }
        }
    }
    if (beta <= std::numeric_limits<double>::epsilon())
        beta = 0;
    else
        beta = 1.f / (2 * beta / (4 * img.cols * img.rows - 3 * img.cols - 3 * img.rows + 2));

    return beta;
}

static void constructGCGraphBasedHis(const Mat& img, const Mat& mask, const ColorHistogram& bgdHist, const ColorHistogram& fgdHist, double lambda,
    const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
    GCGraph<double>& graph) {
    int vtxCount = img.cols * img.rows,
        edgeCount = 2 * (4 * img.cols * img.rows - 3 * (img.cols + img.rows) + 2);
    graph.create(vtxCount, edgeCount);
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            // add node
            int vtxIdx = graph.addVtx();
            Vec3b color = img.at<Vec3b>(p);
            // set t-weights
            double fromSource, toSink;
            if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD) {
                double bgdProb = bgdHist.getProbability(color);
                double fgdProb = fgdHist.getProbability(color);
                fromSource = -log(bgdProb);
                toSink = -log(fgdProb);
            }
            else if (mask.at<uchar>(p) == GC_BGD) {
                fromSource = 0;
                toSink = lambda;
            }
            else {
                fromSource = lambda;
                toSink = 0;
            }
            graph.addTermWeights(vtxIdx, fromSource, toSink);

            // set n-weights
            if (p.x > 0) {
                double w = leftW.at<double>(p) * exp(-norm(color - img.at<Vec3b>(p.y, p.x - 1)) / lambda);
                graph.addEdges(vtxIdx, vtxIdx - 1, w, w);
            }
            if (p.x > 0 && p.y > 0) {
                double w = upleftW.at<double>(p) * exp(-norm(color - img.at<Vec3b>(p.y - 1, p.x - 1)) / lambda);
                graph.addEdges(vtxIdx, vtxIdx - img.cols - 1, w, w);
            }
            if (p.y > 0) {
                double w = upW.at<double>(p) * exp(-norm(color - img.at<Vec3b>(p.y - 1, p.x)) / lambda);
                graph.addEdges(vtxIdx, vtxIdx - img.cols, w, w);
            }
            if (p.x < img.cols - 1 && p.y > 0) {
                double w = uprightW.at<double>(p) * exp(-norm(color - img.at<Vec3b>(p.y - 1, p.x + 1)) / lambda);
                graph.addEdges(vtxIdx, vtxIdx - img.cols + 1, w, w);
            }
        }
    }
}

void estimateSegmentationBasedHis(GCGraph<double>& graph, Mat& mask) {
    graph.maxFlow();
    Point p;
    for (p.y = 0; p.y < mask.rows; p.y++) {
        for (p.x = 0; p.x < mask.cols; p.x++) {
            if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD) {
                if (graph.inSourceSegment(p.y * mask.cols + p.x /*vertex index*/))
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}

static void initMaskWithRectBasedHis(Mat& mask, Size imgSize, Rect rect)
{
    mask.create(imgSize, CV_8UC1);
    mask.setTo(GC_BGD);

    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, imgSize.width - rect.x);
    rect.height = std::min(rect.height, imgSize.height - rect.y);

    (mask(rect)).setTo(Scalar(GC_PR_FGD));
}

static void calcNWeightsBasedHis(const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma) {
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create(img.rows, img.cols, CV_64FC1);
    upleftW.create(img.rows, img.cols, CV_64FC1);
    upW.create(img.rows, img.cols, CV_64FC1);
    uprightW.create(img.rows, img.cols, CV_64FC1);
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            Vec3d color = img.at<Vec3b>(y, x);
            if (x - 1 >= 0) {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
                leftW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
            }
            else
                leftW.at<double>(y, x) = 0;
            if (x - 1 >= 0 && y - 1 >= 0) {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
                upleftW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
            }
            else
                upleftW.at<double>(y, x) = 0;
            if (y - 1 >= 0) {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
                upW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
            }
            else
                upW.at<double>(y, x) = 0;
            if (x + 1 < img.cols && y - 1 >= 0) {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
                uprightW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
            }
            else
                uprightW.at<double>(y, x) = 0;
        }
    }
}

void segmentImage(const Mat& img, const Mat& mask, int iter, const string& baseFilename);

void myGrabCutBasedHis(InputArray _img, InputOutputArray _mask, Rect rect, InputOutputArray _bgdModel, InputOutputArray _fgdModel, int iterCount, int mode) {
    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef();

    ColorHistogram bgdHist, fgdHist;

    if (mode == GC_INIT_WITH_RECT)
        initMaskWithRectBasedHis(mask, img.size(), rect);

    auto bgdHistogram = bgdHist.calculateHistogram(img);
    auto fgdHistogram = fgdHist.calculateHistogram(img);
    bgdHist.smoothSaliency(bgdHistogram);
    fgdHist.smoothSaliency(fgdHistogram);

    const double gamma = 50;
    const double lambda = 9 * gamma;
    const double beta = calcBetaBasedHis(img);

    Mat leftW, upleftW, upW, uprightW;
    calcNWeightsBasedHis(img, leftW, upleftW, upW, uprightW, beta, gamma);

    string baseFilename = "D:output/myGrabCutBasedHis";  // ��������ļ��Ļ���·�����ļ���
    

    for (int i = 0; i < iterCount; i++) {
        bgdHist.assignAndUpdateProbabilities(img, mask);
        fgdHist.assignAndUpdateProbabilities(img, mask);

        GCGraph<double> graph;
        constructGCGraphBasedHis(img, mask, bgdHist, fgdHist, lambda, leftW, upleftW, upW, uprightW, graph);
        estimateSegmentationBasedHis(graph, mask);
        segmentImage(img, mask, i + 1, baseFilename);
    }
}

void segmentImage(const Mat& img, const Mat& mask, int iter, const string& baseFilename) {
    string filename = baseFilename + "_iteration_" + to_string(iter) + ".png";
    Mat output;
    compare(mask, GC_PR_FGD, output, CMP_EQ);
    output = output * 255;  // ת��Ϊ��ֵͼ��
    imwrite(filename, output);  // д���ļ�
    cout << "Saved mask image for iteration " << iter << " to " << filename << endl;
}

namespace {
    class GMM {
    public:
        static const int componentsCount = 5;

        GMM(Mat& _model); // ���캯��
        double operator()(const Vec3d color) const; // ���������ɫ�������и�˹�ֲ������ϸ���
        double operator()(int ci, const Vec3d color) const;
        int whichComponent(const Vec3d color) const; // ȷ��������ɫ����������ĸ���˹�ֲ�

        void initLearning(); // ��ʼ����˹�ֲ���ѧϰ����
        void addSample(int ci, const Vec3d color); // ���˹�ֲ����������
        void endLearning(); // ��ɸ�˹�ֲ���ѧϰ�������ֵ��Э�����ͳ������

    private:
        void calcInverseCovAndDeterm(int ci, double singularFix);
        Mat model;
        double* coefs;
        double* mean;
        double* cov;

        double inverseCovs[componentsCount][3][3];
        double covDeterms[componentsCount];

        double sums[componentsCount][3];
        double prods[componentsCount][3][3];
        int sampleCounts[componentsCount];
        int totalSampleCount;
    };
    // ���캯��
    GMM::GMM(Mat& _model) { // ������ǰ������һ����Ӧ��GMM
        // һ������RGB����ͨ��ֵ����3����ֵ��3*3��Э�������һ��Ȩֵ
        const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
        if (_model.empty()) {
            // һ��GMM����componentsCount����˹ģ�ͣ�һ����˹ģ����modelSize��ģ�Ͳ���
            _model.create(1, modelSize * componentsCount, CV_64FC1);
            _model.setTo(Scalar(0));
        }
        else if ((_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize * componentsCount))
            CV_Error(cv::Error::StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount");

        model = _model;

        coefs = model.ptr<double>(0);
        mean = coefs + componentsCount;
        cov = mean + 3 * componentsCount;

        for (int ci = 0; ci < componentsCount; ci++)
            if (coefs[ci] > 0)
                // ����GMM�е�ci����˹ģ�͵�Э�������Inverse������ʽDeterminant
                // Ϊ�˺������ÿ���������ڸø�˹ģ�͵ĸ��ʣ�Ҳ������������� 
                calcInverseCovAndDeterm(ci, 0.0);
        totalSampleCount = 0;
    }

    // ����һ�������������GMM��ϸ�˹ģ�͵ĸ���
    double GMM::operator()(const Vec3d color) const {
        double res = 0;
        for (int ci = 0; ci < componentsCount; ci++)
            // �����������������componentsCount����˹ģ�͵ĸ������Ӧ��Ȩֵ��������
            res += coefs[ci] * (*this)(ci, color);
        return res;
    }

    // ����һ���������ڵ�ci����˹ģ�͵ĸ��ʡ�
    double GMM::operator()(int ci, const Vec3d color) const {
        double res = 0;
        if (coefs[ci] > 0) {
            // ȷ��Э������������ʽ����һ����С���������Ա������0�Ĵ���
            CV_Assert(covDeterms[ci] > std::numeric_limits<double>::epsilon());
            Vec3d diff = color; // ��ʾ��ɫ�������ֵ����֮��Ĳ���
            double* m = mean + 3 * ci;
            diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
            // �����ֵ������Э������������˻��Ķ�����
            double mult = diff[0] * (diff[0] * inverseCovs[ci][0][0] + diff[1] * inverseCovs[ci][1][0] + diff[2] * inverseCovs[ci][2][0])
                + diff[1] * (diff[0] * inverseCovs[ci][0][1] + diff[1] * inverseCovs[ci][1][1] + diff[2] * inverseCovs[ci][2][1])
                + diff[2] * (diff[0] * inverseCovs[ci][0][2] + diff[1] * inverseCovs[ci][1][2] + diff[2] * inverseCovs[ci][2][2]);
            // ���ݸ�˹�ֲ���PDF��ʽ��������ܶ�
            res = 1.0f / sqrt(covDeterms[ci]) * exp(-0.5f * mult);
        }
        return res;
    }

    // whichComponent����������ȷ��������ɫ����������ĸ���˹�ֲ�
    int GMM::whichComponent(const Vec3d color) const {
        int k = 0;
        double max = 0;
        for (int ci = 0; ci < componentsCount; ci++) {
            double p = (*this)(ci, color);
            if (p > max) {
                k = ci;
                max = p;
            }
        }
        return k;
    }

    // initLearning��������ʼ����˹�ֲ���ѧϰ���̣���Ҫ�Ƕ�Ҫ��͵ı�������
    void GMM::initLearning() {
        for (int ci = 0; ci < componentsCount; ci++) {
            sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
            prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
            prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
            prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
            sampleCounts[ci] = 0;
        }
        totalSampleCount = 0;
    }

    // addSample���������˹�ֲ����������
    // Ϊǰ�����߱���GMM�ĵ�ci����˹ģ�͵����ؼ�������������
    void GMM::addSample(int ci, const Vec3d color) {
        // �������color������غ����ؼ����������ص�RGB����ͨ���ĺ�sums�����������ֵ��
        // prods����������Э���
        // ���ؼ������ظ������ܵ����ظ������������������˹ģ�͵�Ȩֵ��
        sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
        prods[ci][0][0] += color[0] * color[0]; prods[ci][0][1] += color[0] * color[1]; prods[ci][0][2] += color[0] * color[2];
        prods[ci][1][0] += color[1] * color[0]; prods[ci][1][1] += color[1] * color[1]; prods[ci][1][2] += color[1] * color[2];
        prods[ci][2][0] += color[2] * color[0]; prods[ci][2][1] += color[2] * color[1]; prods[ci][2][2] += color[2] * color[2];
        sampleCounts[ci]++;
        totalSampleCount++;
    }

    // endLearning��������ɸ�˹�ֲ���ѧϰ�������ֵ��Э�����ͳ������
    void GMM::endLearning() {
        for (int ci = 0; ci < componentsCount; ci++) {
            int n = sampleCounts[ci];  // ��ci����˹ģ�͵��������ظ���
            if (n == 0)
                coefs[ci] = 0;
            else {
                CV_Assert(totalSampleCount > 0);
                double inv_n = 1.0 / n;
                // �����ci����˹ģ�͵�Ȩֵϵ��
                coefs[ci] = (double)n / totalSampleCount;
                // �����ci����˹ģ�͵ľ�ֵ
                double* m = mean + 3 * ci;
                m[0] = sums[ci][0] * inv_n; m[1] = sums[ci][1] * inv_n; m[2] = sums[ci][2] * inv_n;
                // �����ci����˹ģ�͵�Э����
                double* c = cov + 9 * ci;
                c[0] = prods[ci][0][0] * inv_n - m[0] * m[0]; c[1] = prods[ci][0][1] * inv_n - m[0] * m[1]; c[2] = prods[ci][0][2] * inv_n - m[0] * m[2];
                c[3] = prods[ci][1][0] * inv_n - m[1] * m[0]; c[4] = prods[ci][1][1] * inv_n - m[1] * m[1]; c[5] = prods[ci][1][2] * inv_n - m[1] * m[2];
                c[6] = prods[ci][2][0] * inv_n - m[2] * m[0]; c[7] = prods[ci][2][1] * inv_n - m[2] * m[1]; c[8] = prods[ci][2][2] * inv_n - m[2] * m[2];

                calcInverseCovAndDeterm(ci, 0.01);
            }
        }
    }

    // ����Э���������������ʽ�����ڸ����ܶȺ����ļ���
    void GMM::calcInverseCovAndDeterm(int ci, const double singularFix) {
        if (coefs[ci] > 0) {
            double* c = cov + 9 * ci;
            // �����ci����˹ģ�͵�Э���������ʽ
            double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
            if (dtrm <= 1e-6 && singularFix > 0) {
                // �������ʽС�ڵ���0�����Խ���Ԫ�أ����Ӱ��������������Ϊ�˻������ȣ�Э�������
                // ������������󣬵�����ļ�����Ҫ���������
                c[0] += singularFix;
                c[4] += singularFix;
                c[8] += singularFix;
                dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
            }
            covDeterms[ci] = dtrm;
            // ��֤dtrm>0��������ʽ�ļ�����ȷ
            CV_Assert(dtrm > std::numeric_limits<double>::epsilon());
            double inv_dtrm = 1.0 / dtrm;
            // ���׷��������
            inverseCovs[ci][0][0] = (c[4] * c[8] - c[5] * c[7]) * inv_dtrm;
            inverseCovs[ci][1][0] = -(c[3] * c[8] - c[5] * c[6]) * inv_dtrm;
            inverseCovs[ci][2][0] = (c[3] * c[7] - c[4] * c[6]) * inv_dtrm;
            inverseCovs[ci][0][1] = -(c[1] * c[8] - c[2] * c[7]) * inv_dtrm;
            inverseCovs[ci][1][1] = (c[0] * c[8] - c[2] * c[6]) * inv_dtrm;
            inverseCovs[ci][2][1] = -(c[0] * c[7] - c[1] * c[6]) * inv_dtrm;
            inverseCovs[ci][0][2] = (c[1] * c[5] - c[2] * c[4]) * inv_dtrm;
            inverseCovs[ci][1][2] = -(c[0] * c[5] - c[2] * c[3]) * inv_dtrm;
            inverseCovs[ci][2][2] = (c[0] * c[4] - c[1] * c[3]) * inv_dtrm;
        }
    }

} // namespace

/*
  ����GrabCut�㷨�е�beta�������ò�������ͼ������ɫֵ�Ĳ���
  beta���������߻��ߵͶԱȶ�ʱ�������������صĲ���Ӱ���
  �����ڵͶԱȶ�ʱ�������������صĲ����ܾͻ�Ƚ�С����ʱ����Ҫ����һ���ϴ��beta���Ŵ�������
  �ڸ߶Աȶ�ʱ������Ҫ��С����ͱȽϴ�Ĳ��
  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
static double calcBeta(const Mat& img) {
    double beta = 0;
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            // �����ĸ��������������صĲ��Ҳ����ŷʽ�������˵���׷���
            // �����������ض�����󣬾��൱�ڼ������������ز��ˣ� 
            Vec3d color = img.at<Vec3b>(y, x);
            if (x > 0) { // left ������ͼ��߽��ʱ�򻹼��㣬����Խ��
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
                beta += diff.dot(diff);
            }
            if (y > 0 && x > 0) { // upleft
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
                beta += diff.dot(diff);
            }
            if (y > 0) { // up
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
                beta += diff.dot(diff);
            }
            if (y > 0 && x < img.cols - 1) { // upright
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
                beta += diff.dot(diff);
            }
        }
    }
    if (beta <= std::numeric_limits<double>::epsilon())
        beta = 0;
    else
        beta = 1.f / (2 * beta / (4 * img.cols * img.rows - 3 * img.cols - 3 * img.rows + 2));

    return beta;
}

/*
  ����ͼ�з��ն˽ڵ��Ȩ�أ���ЩȨ�ػ���ͼ��������֮�����ɫ����
  ����ͼÿ���Ƕ˵㶥��ıߵ�Ȩֵ���൱�ڼ���Gibbs������ƽ����
 */
static void calcNWeights(const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma) {
    // �൱�ڹ�ʽ��4���е�gamma * dis(i,j)^(-1)
    // i��j�Ǵ�ֱ����ˮƽ��ϵʱ��dis(i,j)=1�����ǶԽǹ�ϵʱ��dis(i,j)=sqrt(2.0f)
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    // ÿ������ıߵ�Ȩֵͨ��һ����ͼ��С��ȵ�Mat������
    leftW.create(img.rows, img.cols, CV_64FC1);
    upleftW.create(img.rows, img.cols, CV_64FC1);
    upW.create(img.rows, img.cols, CV_64FC1);
    uprightW.create(img.rows, img.cols, CV_64FC1);
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            Vec3d color = img.at<Vec3b>(y, x);
            if (x - 1 >= 0) { // left 
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
                leftW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
            }
            else
                leftW.at<double>(y, x) = 0;
            if (x - 1 >= 0 && y - 1 >= 0) { // upleft
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
                upleftW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
            }
            else
                upleftW.at<double>(y, x) = 0;
            if (y - 1 >= 0) { // up
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
                upW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
            }
            else
                upW.at<double>(y, x) = 0;
            if (x + 1 < img.cols && y - 1 >= 0) { // upright
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
                uprightW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
            }
            else
                uprightW.at<double>(y, x) = 0;
        }
    }
}

// ʹ�þ��γ�ʼ�����룬�������ڵ����ر��Ϊǰ����ѡ
static void initMaskWithRect(Mat& mask, Size imgSize, Rect rect) {
    mask.create(imgSize, CV_8UC1);
    mask.setTo(GC_BGD);

    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, imgSize.width - rect.x);
    rect.height = std::min(rect.height, imgSize.height - rect.y);

    (mask(rect)).setTo(Scalar(GC_PR_FGD));
}

// ʹ��kmeans�㷨��ʼ��������ǰ���ĸ�˹���ģ��
static void initGMMs(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM) {
    const int kMeansItCount = 10; // ��������
    const int kMeansType = KMEANS_PP_CENTERS;
    // ��¼������ǰ����������������ÿ�����ض�ӦGMM���ĸ���˹ģ�ͣ������е�kn
    Mat bgdLabels, fgdLabels;
    std::vector<Vec3f> bgdSamples, fgdSamples;
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            // mask�б��ΪGC_BGD��GC_PR_BGD�����ض���Ϊ��������������
            if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD)
                bgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
        }
    }
    CV_Assert(!bgdSamples.empty() && !fgdSamples.empty());
    {
        // kmeans�в���_bgdSamplesΪ��ÿ��һ������
        // kmeans�����ΪbgdLabels�����汣�����������������ÿһ��������Ӧ�����ǩ
        Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
        int num_clusters = GMM::componentsCount;
        num_clusters = std::min(num_clusters, (int)bgdSamples.size());
        kmeans(_bgdSamples, num_clusters, bgdLabels,
            TermCriteria(TermCriteria::MAX_ITER, kMeansItCount, 0.0), 0, kMeansType);
    }
    {
        Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
        int num_clusters = GMM::componentsCount;
        num_clusters = std::min(num_clusters, (int)fgdSamples.size());
        kmeans(_fgdSamples, num_clusters, fgdLabels,
            TermCriteria(TermCriteria::MAX_ITER, kMeansItCount, 0.0), 0, kMeansType);
    }
    // ��������Ĳ����ÿ�����������ĸ�˹ģ�;�ȷ�����ˣ���ô�Ϳ��Թ���GMM��ÿ����˹ģ�͵Ĳ�����
    bgdGMM.initLearning();
    for (int i = 0; i < (int)bgdSamples.size(); i++)
        bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for (int i = 0; i < (int)fgdSamples.size(); i++)
        fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
    fgdGMM.endLearning();
}
// Ϊÿ�����ط����˹���ģ�͵����
// ������С���㷨step 1��Ϊÿ�����ط���GMM�������ĸ�˹ģ�ͣ�kn������Mat compIdxs��
static void assignGMMsComponents(const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs) {
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            Vec3d color = img.at<Vec3b>(p);
            // ͨ��mask���жϸ��������ڱ������ػ���ǰ������
            // ���ж�������ǰ�����߱���GMM�е��ĸ���˹����
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
                bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
        }
    }
}

// �������ص��������ѧϰ��˹���ģ�͵Ĳ���
// �����У�������С���㷨step 2����ÿ����˹ģ�͵�������������ѧϰÿ����˹ģ�͵Ĳ���
static void learnGMMs(const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM) {
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    Point p;
    for (int ci = 0; ci < GMM::componentsCount; ci++) {
        for (p.y = 0; p.y < img.rows; p.y++) {
            for (p.x = 0; p.x < img.cols; p.x++) {
                if (compIdxs.at<int>(p) == ci) {
                    if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD)
                        bgdGMM.addSample(ci, img.at<Vec3b>(p));
                    else
                        fgdGMM.addSample(ci, img.at<Vec3b>(p));
                }
            }
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

// ͨ������õ����������ͼ��ͼ�Ķ���Ϊ���ص㣬ͼ�ı��������ֹ���
// һ����ǣ�ÿ����������t������������Դ��s������ǰ�������ӵı�
// ����ߵ�Ȩֵͨ��Gibbs������ĵ�һ������������ʾ
// ��һ����ǣ�ÿ�������������򶥵����ӵı�
// ����ߵ�Ȩֵͨ��Gibbs������ĵڶ�������������ʾ
static void constructGCGraph(const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda,
    const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
    GCGraph<double>& graph) {
    int vtxCount = img.cols * img.rows, // ��������ÿһ��������һ������
        // ��������Ҫ����ͼ�߽�ıߵ�ȱʧ
        edgeCount = 2 * (4 * img.cols * img.rows - 3 * (img.cols + img.rows) + 2);
    graph.create(vtxCount, edgeCount); // ͨ���������ͱ�������ͼ
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            // add node
            int vtxIdx = graph.addVtx(); // �������������ͼ�е�����
            Vec3b color = img.at<Vec3b>(p);
            // set t-weights
            // ����ÿ����������t������������Դ��s������ǰ�������ӵ�Ȩֵ��
            // Ҳ������Gibbs������ÿһ�����ص���Ϊ�������ػ���ǰ�����أ��ĵ�һ�������� 
            double fromSource, toSink;
            if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD) {
                fromSource = -log(bgdGMM(color));
                toSink = -log(fgdGMM(color));
            } else if (mask.at<uchar>(p) == GC_BGD) {
                fromSource = 0;
                toSink = lambda;
            } else { // GC_FGD
                fromSource = lambda;
                toSink = 0;
            }
            // ���øö���vtxIdx�ֱ���Source���Sink�������Ȩֵ
            graph.addTermWeights(vtxIdx, fromSource, toSink);
            // set n-weights
            // �����������򶥵�֮�����ӵ�Ȩֵ��Ҳ������Gibbs�����ĵڶ��������ƽ��� 
            if (p.x > 0) {
                double w = leftW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - 1, w, w);
            }
            if (p.x > 0 && p.y > 0) {
                double w = upleftW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - img.cols - 1, w, w);
            }
            if (p.y > 0) {
                double w = upW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - img.cols, w, w);
            }
            if (p.x < img.cols - 1 && p.y>0) {
                double w = uprightW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - img.cols + 1, w, w);
            }
        }
    }
}
// �����У�������С���㷨step 3���ָ���ƣ���С�����������㷨
static void estimateSegmentation(GCGraph<double>& graph, Mat& mask) {
    graph.maxFlow(); // ͨ��������㷨ȷ��ͼ����С�Ҳ�����ͼ��ķָ�
    Point p;
    for (p.y = 0; p.y < mask.rows; p.y++) {
        for (p.x = 0; p.x < mask.cols; p.x++) {
            // ͨ��ͼ�ָ�Ľ��������mask��������ͼ��ָ���
            // ע����ǣ���Զ����������û�ָ��Ϊ��������ǰ�������� 
            if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD) {
                if (graph.inSourceSegment(p.y * mask.cols + p.x /*vertex index*/))
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}

// ����ͼ�����롢�������򡢱�����ǰ��ģ�͡�����������ģʽ��Ϊ����
void myGrabCut(InputArray _img, InputOutputArray _mask, Rect rect,
    InputOutputArray _bgdModel, InputOutputArray _fgdModel,
    int iterCount, int mode) {

    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef();
    Mat& bgdModel = _bgdModel.getMatRef();
    Mat& fgdModel = _fgdModel.getMatRef();

    GMM bgdGMM(bgdModel), fgdGMM(fgdModel);
    Mat compIdxs(img.size(), CV_32SC1);

    if (mode == GC_INIT_WITH_RECT) {
        if (mode == GC_INIT_WITH_RECT)
            initMaskWithRect(mask, img.size(), rect);
        initGMMs(img, mask, bgdGMM, fgdGMM);
    }

    const double gamma = 200;
    const double lambda = 9 * gamma;
    const double beta = calcBeta(img);

    Mat leftW, upleftW, upW, uprightW;
    calcNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma);

    string baseFilename = "D:output/myGrabCut";  // ��������ļ��Ļ���·�����ļ���

    for (int i = 0; i < iterCount; i++) {
        GCGraph<double> graph;
        assignGMMsComponents(img, mask, bgdGMM, fgdGMM, compIdxs);
        if (mode != GC_EVAL_FREEZE_MODEL)
            learnGMMs(img, mask, compIdxs, bgdGMM, fgdGMM);
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
        estimateSegmentation(graph, mask);

        // segmentImage(img, mask, i + 1, baseFilename);
    }
}

#endif