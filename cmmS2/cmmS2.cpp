#include <cstdio>
#include <iostream> 
#include <iomanip>
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <Eigen/Eigen>

#include "seam_carving.h"
#include "mesh_placement.h"
#include "energy.h"
#include "fLine.h"
#include "GLout.h"
#include "tool.h"

using namespace cv;
using namespace std;



int main() {
    double st = clock();
    Mat img = imread("D:/.A_myTest/cmm/input/1_input.jpg");
    Mat line_img;
    //imshow("Input", img);
    //waitKey(0);

    seam_carving::init(img);
    img = seam_carving::get_rec_img();
    double st1 = clock();
    cout << "seam carving cost time: " << (st1 - st) / CLOCKS_PER_SEC << endl;

    vector<vector<point>> U = seam_carving::get_U(), pos;
    vector<vector<point>> mesh = Mesh_Placement(seam_carving::get_seam_carving(), img, U, pos); // 网格坐标
    double st2 = clock();
    cout << "mesh cost time: " << (st2 - st1) / CLOCKS_PER_SEC << endl;

    int* n_line = new int, * Nl = new int; // 线段数量
    double* line = fline::line_img(img, n_line, line_img); // 线段
    vector<vector<vector<Line>>> mesh_line = fline::init_line(mesh, line, *n_line, Nl, 0); // 每个网格中的线段
    fline::check_line(img, mesh_line); // 绘图

    int M = 50; // 桶的数量
    int* num = new int[M];
    double* bins;
    bins = new double[M];
    fline::init_bins(num, mesh_line, M);
    energy::Line_rotate_count(bins, num, mesh_line, mesh, pos, M); // 为后续的迭代提供基线数据

    double Nq = 20 * 20, lambdab = 1e6, lambdal = 10;
    SparseMatrix<double> ES = (1.0 / Nq) * energy::shape_energy(mesh); 
    pair<SparseMatrix<double>, MatrixXd > EB = energy::bound_energy(mesh, lambdab, img.rows, img.cols);
    double st3 = clock();
    cout << "shape and bound cost time: " << (st3 - st2) / CLOCKS_PER_SEC << endl;

    double lstcost = 1000000, cost;

    for (int iter = 0; iter < 10; ++iter) {
        cout << "iter: " << iter << endl;

        SparseMatrix<double> EL = lambdal * (1.0 / *Nl) * energy::line_energy(mesh_line, mesh, pos, bins, *Nl, img.rows, img.cols);

        VectorXd B = VectorXd::Zero(ES.rows() + EL.rows());
        SparseMatrix<double> A = merge_matrix(merge_matrix(ES, EL), EB.first);
        B = merge_matrix(B, EB.second);

        SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> solver; // QR分解求解器
        solver.compute(A); // 对矩阵A进行QR分解
        VectorXd V = solver.solve(B); // 求解线性方程组A*V=B

        // 将求解得到的向量V和向量B转换成稀疏矩阵形式
        SparseMatrix<double> Vq(21 * 21 * 2, 1), Wq(B.rows(), 1);
        for (int i = 0; i < V.rows(); ++i)
            Vq.insert(i, 0) = V(i);
        for (int i = 0; i < B.rows(); ++i)
            Wq.insert(i, 0) = B(i);

        SparseMatrix<double> energy_cost;
        energy_cost = (A * Vq) - Wq; // 当前计算的能量项值与目标值B之间的差异
        energy_cost = energy_cost.transpose() * energy_cost; // 范数平方
        cost = energy_cost.coeffRef(0, 0); // 提取值

        cout << "energy cost: " << energy_cost.coeffRef(0, 0) << endl;

        SparseMatrix<double> shape_cost;
        shape_cost = ES * Vq;
        shape_cost = shape_cost.transpose() * shape_cost;
        cout << "shape cost: " << shape_cost.coeffRef(0, 0) << endl;

        SparseMatrix<double> line_cost;
        line_cost = EL * Vq;
        line_cost = line_cost.transpose() * line_cost;
        cout << "line cost: " << line_cost.coeffRef(0, 0) << endl;

        pos = vec_to_mesh(V); // 更新网格顶点的位置

        // 当前迭代的总能量代价与上一次迭代的能量代价之差小于阈值，则认为能量已经收敛
        if (abs(cost - lstcost) < 0.01)
            break;
        else
            lstcost = cost;
        // 每次迭代中，网格的顶点位置可能会被更新以最小化整体能量
        // 因此每次迭代之后都需要重新计算网格中每条线段的角度变化
        energy::Line_rotate_count(bins, num, mesh_line, mesh, pos, M);
    }

    double st4 = clock();
    cout << "energy cost time: " << (st4 - st3) / CLOCKS_PER_SEC << endl;

    img_mesh(line_img, mesh);
    GLout(line_img, mesh, pos);

    cout << "total cost time: " << (st4 - st) / CLOCKS_PER_SEC << endl;
    GLout(img, mesh, pos);
    waitKey(0);
    return 0;
}