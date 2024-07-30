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
    static const int levelsPerChannel = 20;  // 量化级别
    map<Vec3b, float, bool(*)(const Vec3b&, const Vec3b&)> fgProbabilities; // 前景概率
    map<Vec3b, float, bool(*)(const Vec3b&, const Vec3b&)> bgProbabilities; // 背景概率

    // 比较函数，确保颜色在map中按顺序存储
    static bool colorComp(const Vec3b& a, const Vec3b& b) {
        return (a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1]) || (a[0] == b[0] && a[1] == b[1] && a[2] < b[2]);
    }

public:
    ColorHistogram() : totalPixels(0), data(colorComp), fgProbabilities(colorComp), bgProbabilities(colorComp) {}

    // 颜色量化函数
    static Mat quantizeColors(const Mat& src) {
        Mat quantized = src.clone();
        // 12×12×12=1728 种不同的颜色
        float scale = 256.0f / levelsPerChannel; // 21.33
        for (int i = 0; i < src.rows; i++) {
            for (int j = 0; j < src.cols; j++) {
                Vec3b& pixel = quantized.at<Vec3b>(i, j);
                // pixel=150, floor(pixel[0]/scale)=7, 7×21.33=149.31, 150 在量化后会变成 149
                pixel[0] = floor(pixel[0] / scale) * scale; 
                pixel[1] = floor(pixel[1] / scale) * scale;
                pixel[2] = floor(pixel[2] / scale) * scale;
            }
        }
        return quantized;
    }

    // 计算直方图
    map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)> calculateHistogram(const Mat& img) {
        Mat quantized = quantizeColors(img);
        map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)> histogram(colorComp);
        for (int i = 0; i < quantized.rows; i++) {
            for (int j = 0; j < quantized.cols; j++) {
                Vec3b color = quantized.at<Vec3b>(i, j); // 访问图像 img 中位置 (i, j) 的像素
                histogram[color]++; // 更新直方图
            }
        }
        return histogram;
    }

    // 平滑显著性计算
    void smoothSaliency(const map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)>& histogram) {
        const float similarityThreshold = 200.0f;
        map<Vec3b, float, bool(*)(const Vec3b&, const Vec3b&)> smoothedData(histogram.key_comp());

        for (auto& outer : histogram) {
            float totalWeight = 0.0f;
            float weightedSaliency = 0.0f;
            for (auto& inner : histogram) {
                /* 
                如果两个颜色的距离小于这个阈值，它们就被认为是相似的
                如果两种颜色的差异较大，它们在色彩空间中的距离就会更远，从而导致它们之间的权重更小或者为零
                这意味着在进行显著性平滑处理时，彼此距离较远的颜色几乎不会互相影响
                因此，大的颜色差异可以使得这些颜色在视觉上更加容易区分开，有助于在图像分析中突出不同的特征或区域 
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

    // 获取颜色的概率
    float getProbability(const Vec3b& color) const {
        auto it = data.find(color);
        if (it != data.end()) {
            return it->second / static_cast<float>(totalPixels);
        }
        return 1e-5f;
    }

    // 初始化或更新前景和背景概率
    void assignAndUpdateProbabilities(const Mat& img, const Mat& mask) {
        map<Vec3b, int, bool(*)(const Vec3b&, const Vec3b&)> colorCounts(colorComp); // 使用自定义比较函数
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

    string baseFilename = "D:output/myGrabCutBasedHis";  // 定义输出文件的基本路径和文件名
    

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
    output = output * 255;  // 转换为二值图像
    imwrite(filename, output);  // 写入文件
    cout << "Saved mask image for iteration " << iter << " to " << filename << endl;
}

namespace {
    class GMM {
    public:
        static const int componentsCount = 5;

        GMM(Mat& _model); // 构造函数
        double operator()(const Vec3d color) const; // 计算给定颜色对于所有高斯分布的联合概率
        double operator()(int ci, const Vec3d color) const;
        int whichComponent(const Vec3d color) const; // 确定给定颜色最可能属于哪个高斯分布

        void initLearning(); // 初始化高斯分布的学习过程
        void addSample(int ci, const Vec3d color); // 向高斯分布中添加样本
        void endLearning(); // 完成高斯分布的学习，计算均值、协方差等统计数据

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
    // 构造函数
    GMM::GMM(Mat& _model) { // 背景和前景各有一个对应的GMM
        // 一个像素RGB三个通道值，故3个均值，3*3个协方差，共用一个权值
        const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
        if (_model.empty()) {
            // 一个GMM共有componentsCount个高斯模型，一个高斯模型有modelSize个模型参数
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
                // 计算GMM中第ci个高斯模型的协方差的逆Inverse和行列式Determinant
                // 为了后面计算每个像素属于该高斯模型的概率（也就是数据能量项） 
                calcInverseCovAndDeterm(ci, 0.0);
        totalSampleCount = 0;
    }

    // 计算一个像素属于这个GMM混合高斯模型的概率
    double GMM::operator()(const Vec3d color) const {
        double res = 0;
        for (int ci = 0; ci < componentsCount; ci++)
            // 把这个像素像素属于componentsCount个高斯模型的概率与对应的权值相乘再相加
            res += coefs[ci] * (*this)(ci, color);
        return res;
    }

    // 计算一个像素属于第ci个高斯模型的概率。
    double GMM::operator()(int ci, const Vec3d color) const {
        double res = 0;
        if (coefs[ci] > 0) {
            // 确保协方差矩阵的行列式大于一个很小的正数，以避免除以0的错误
            CV_Assert(covDeterms[ci] > std::numeric_limits<double>::epsilon());
            Vec3d diff = color; // 表示颜色向量与均值向量之间的差异
            double* m = mean + 3 * ci;
            diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
            // 计算差值向量与协方差矩阵逆矩阵乘积的二次项
            double mult = diff[0] * (diff[0] * inverseCovs[ci][0][0] + diff[1] * inverseCovs[ci][1][0] + diff[2] * inverseCovs[ci][2][0])
                + diff[1] * (diff[0] * inverseCovs[ci][0][1] + diff[1] * inverseCovs[ci][1][1] + diff[2] * inverseCovs[ci][2][1])
                + diff[2] * (diff[0] * inverseCovs[ci][0][2] + diff[1] * inverseCovs[ci][1][2] + diff[2] * inverseCovs[ci][2][2]);
            // 根据高斯分布的PDF公式计算概率密度
            res = 1.0f / sqrt(covDeterms[ci]) * exp(-0.5f * mult);
        }
        return res;
    }

    // whichComponent函数，用于确定给定颜色最可能属于哪个高斯分布
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

    // initLearning函数，初始化高斯分布的学习过程，主要是对要求和的变量置零
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

    // addSample函数，向高斯分布中添加样本
    // 为前景或者背景GMM的第ci个高斯模型的像素集增加样本像素
    void GMM::addSample(int ci, const Vec3d color) {
        // 计算加入color这个像素后，像素集中所有像素的RGB三个通道的和sums（用来计算均值）
        // prods（用来计算协方差）
        // 像素集的像素个数和总的像素个数（用来计算这个高斯模型的权值）
        sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
        prods[ci][0][0] += color[0] * color[0]; prods[ci][0][1] += color[0] * color[1]; prods[ci][0][2] += color[0] * color[2];
        prods[ci][1][0] += color[1] * color[0]; prods[ci][1][1] += color[1] * color[1]; prods[ci][1][2] += color[1] * color[2];
        prods[ci][2][0] += color[2] * color[0]; prods[ci][2][1] += color[2] * color[1]; prods[ci][2][2] += color[2] * color[2];
        sampleCounts[ci]++;
        totalSampleCount++;
    }

    // endLearning函数，完成高斯分布的学习，计算均值、协方差等统计数据
    void GMM::endLearning() {
        for (int ci = 0; ci < componentsCount; ci++) {
            int n = sampleCounts[ci];  // 第ci个高斯模型的样本像素个数
            if (n == 0)
                coefs[ci] = 0;
            else {
                CV_Assert(totalSampleCount > 0);
                double inv_n = 1.0 / n;
                // 计算第ci个高斯模型的权值系数
                coefs[ci] = (double)n / totalSampleCount;
                // 计算第ci个高斯模型的均值
                double* m = mean + 3 * ci;
                m[0] = sums[ci][0] * inv_n; m[1] = sums[ci][1] * inv_n; m[2] = sums[ci][2] * inv_n;
                // 计算第ci个高斯模型的协方差
                double* c = cov + 9 * ci;
                c[0] = prods[ci][0][0] * inv_n - m[0] * m[0]; c[1] = prods[ci][0][1] * inv_n - m[0] * m[1]; c[2] = prods[ci][0][2] * inv_n - m[0] * m[2];
                c[3] = prods[ci][1][0] * inv_n - m[1] * m[0]; c[4] = prods[ci][1][1] * inv_n - m[1] * m[1]; c[5] = prods[ci][1][2] * inv_n - m[1] * m[2];
                c[6] = prods[ci][2][0] * inv_n - m[2] * m[0]; c[7] = prods[ci][2][1] * inv_n - m[2] * m[1]; c[8] = prods[ci][2][2] * inv_n - m[2] * m[2];

                calcInverseCovAndDeterm(ci, 0.01);
            }
        }
    }

    // 计算协方差的逆矩阵和行列式，用于概率密度函数的计算
    void GMM::calcInverseCovAndDeterm(int ci, const double singularFix) {
        if (coefs[ci] > 0) {
            double* c = cov + 9 * ci;
            // 计算第ci个高斯模型的协方差的行列式
            double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
            if (dtrm <= 1e-6 && singularFix > 0) {
                // 如果行列式小于等于0，（对角线元素）增加白噪声，避免其变为退化（降秩）协方差矩阵
                // （不存在逆矩阵，但后面的计算需要计算逆矩阵）
                c[0] += singularFix;
                c[4] += singularFix;
                c[8] += singularFix;
                dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
            }
            covDeterms[ci] = dtrm;
            // 保证dtrm>0，即行列式的计算正确
            CV_Assert(dtrm > std::numeric_limits<double>::epsilon());
            double inv_dtrm = 1.0 / dtrm;
            // 三阶方阵的求逆
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
  计算GrabCut算法中的beta参数，该参数基于图像中颜色值的差异
  beta用来调整高或者低对比度时，两个邻域像素的差别的影响的
  例如在低对比度时，两个邻域像素的差别可能就会比较小，这时候需要乘以一个较大的beta来放大这个差别
  在高对比度时，则需要缩小本身就比较大的差别。
  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
static double calcBeta(const Mat& img) {
    double beta = 0;
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            // 计算四个方向邻域两像素的差别，也就是欧式距离或者说二阶范数
            // （当所有像素都算完后，就相当于计算八邻域的像素差了） 
            Vec3d color = img.at<Vec3b>(y, x);
            if (x > 0) { // left 避免在图像边界的时候还计算，导致越界
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
  计算图中非终端节点的权重，这些权重基于图像中像素之间的颜色差异
  计算图每个非端点顶点的边的权值，相当于计算Gibbs能量的平滑项
 */
static void calcNWeights(const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma) {
    // 相当于公式（4）中的gamma * dis(i,j)^(-1)
    // i和j是垂直或者水平关系时，dis(i,j)=1，当是对角关系时，dis(i,j)=sqrt(2.0f)
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    // 每个方向的边的权值通过一个和图大小相等的Mat来保存
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

// 使用矩形初始化掩码，将矩形内的像素标记为前景候选
static void initMaskWithRect(Mat& mask, Size imgSize, Rect rect) {
    mask.create(imgSize, CV_8UC1);
    mask.setTo(GC_BGD);

    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, imgSize.width - rect.x);
    rect.height = std::min(rect.height, imgSize.height - rect.y);

    (mask(rect)).setTo(Scalar(GC_PR_FGD));
}

// 使用kmeans算法初始化背景和前景的高斯混合模型
static void initGMMs(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM) {
    const int kMeansItCount = 10; // 迭代次数
    const int kMeansType = KMEANS_PP_CENTERS;
    // 记录背景和前景的像素样本集中每个像素对应GMM的哪个高斯模型，论文中的kn
    Mat bgdLabels, fgdLabels;
    std::vector<Vec3f> bgdSamples, fgdSamples;
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            // mask中标记为GC_BGD和GC_PR_BGD的像素都作为背景的样本像素
            if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD)
                bgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
        }
    }
    CV_Assert(!bgdSamples.empty() && !fgdSamples.empty());
    {
        // kmeans中参数_bgdSamples为：每行一个样本
        // kmeans的输出为bgdLabels，里面保存的是输入样本集中每一个样本对应的类标签
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
    // 经过上面的步骤后，每个像素所属的高斯模型就确定的了，那么就可以估计GMM中每个高斯模型的参数了
    bgdGMM.initLearning();
    for (int i = 0; i < (int)bgdSamples.size(); i++)
        bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for (int i = 0; i < (int)fgdSamples.size(); i++)
        fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
    fgdGMM.endLearning();
}
// 为每个像素分配高斯混合模型的组件
// 迭代最小化算法step 1：为每个像素分配GMM中所属的高斯模型，kn保存在Mat compIdxs中
static void assignGMMsComponents(const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs) {
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            Vec3d color = img.at<Vec3b>(p);
            // 通过mask来判断该像素属于背景像素还是前景像素
            // 再判断它属于前景或者背景GMM中的哪个高斯分量
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
                bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
        }
    }
}

// 根据像素的组件分配学习高斯混合模型的参数
// 论文中：迭代最小化算法step 2：从每个高斯模型的像素样本集中学习每个高斯模型的参数
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

// 通过计算得到的能量项构建图，图的顶点为像素点，图的边由两部分构成
// 一类边是：每个顶点与汇点t（代表背景）和源点s（代表前景）连接的边
// 这类边的权值通过Gibbs能量项的第一项能量项来表示
// 另一类边是：每个顶点与其邻域顶点连接的边
// 这类边的权值通过Gibbs能量项的第二项能量项来表示
static void constructGCGraph(const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda,
    const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
    GCGraph<double>& graph) {
    int vtxCount = img.cols * img.rows, // 顶点数，每一个像素是一个顶点
        // 边数，需要考虑图边界的边的缺失
        edgeCount = 2 * (4 * img.cols * img.rows - 3 * (img.cols + img.rows) + 2);
    graph.create(vtxCount, edgeCount); // 通过顶点数和边数创建图
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            // add node
            int vtxIdx = graph.addVtx(); // 返回这个顶点在图中的索引
            Vec3b color = img.at<Vec3b>(p);
            // set t-weights
            // 计算每个顶点与汇点t（代表背景）和源点s（代表前景）连接的权值。
            // 也即计算Gibbs能量（每一个像素点作为背景像素或者前景像素）的第一个能量项 
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
            // 设置该顶点vtxIdx分别与Source点和Sink点的连接权值
            graph.addTermWeights(vtxIdx, fromSource, toSink);
            // set n-weights
            // 计算两个邻域顶点之间连接的权值，也即计算Gibbs能量的第二个能量项（平滑项） 
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
// 论文中：迭代最小化算法step 3：分割估计：最小割或者最大流算法
static void estimateSegmentation(GCGraph<double>& graph, Mat& mask) {
    graph.maxFlow(); // 通过最大流算法确定图的最小割，也即完成图像的分割
    Point p;
    for (p.y = 0; p.y < mask.rows; p.y++) {
        for (p.x = 0; p.x < mask.cols; p.x++) {
            // 通过图分割的结果来更新mask，即最后的图像分割结果
            // 注意的是，永远都不会更新用户指定为背景或者前景的像素 
            if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD) {
                if (graph.inSourceSegment(p.y * mask.cols + p.x /*vertex index*/))
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}

// 接受图像、掩码、矩形区域、背景和前景模型、迭代次数和模式作为参数
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

    string baseFilename = "D:output/myGrabCut";  // 定义输出文件的基本路径和文件名

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