#include "Helper.h"

void Helper::ShowMatrix(std::unique_ptr<std::unique_ptr<double[]>[]>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%5.3f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void Helper::ShowVector(std::unique_ptr<double[]>& ptr, int N) {
    int cnt = 0;
    for (int i = 0; i < N; ++i) {
        printf("%5.2f ", ptr[i]);
        if (++cnt >= 10) {
            printf("\n");
            cnt = 0;
        }
    }
}

void Helper::SwapRows(std::unique_ptr<double[]>& row1, std::unique_ptr<double[]>& row2, int cols) {
    auto ptr = std::make_unique<double[]>(cols);
    for (int i = 0; i < cols; ++i) {
        ptr[i] = row1[i];
    }
    for (int i = 0; i < cols; ++i) {
        row1[i] = row2[i];
    }
    for (int i = 0; i < cols; ++i) {
        row2[i] = ptr[i];
    }
}

void Helper::SwapScalars(double& x1, double& x2) {
    double buff = x1;
    x1 = x2;
    x2 = buff;
}

void Helper::Shuffle(std::unique_ptr<std::unique_ptr<double[]>[]>& matrix, std::unique_ptr<double[]>& vector, int rows, int cols) {
    for (int i = 0; i < 2 * rows; ++i) {
        int n1 = rand() % rows;
        int n2 = rand() % rows;
        SwapRows(matrix[n1], matrix[n2], cols);
        SwapScalars(vector[n1], vector[n2]);
    }
}

void Helper::FindMinMax(std::unique_ptr<double[]>& xmin, std::unique_ptr<double[]>& xmax,
    double& targetMin, double& targetMax,
    std::unique_ptr<std::unique_ptr<double[]>[]>& matrix,
    std::unique_ptr<double[]>& target, int nRows, int nCols) {

    for (int i = 0; i < nCols; ++i) {
        xmin[i] = DBL_MAX;
        xmax[i] = -DBL_MIN;
    }
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            if (matrix[i][j] < xmin[j]) xmin[j] = static_cast<double>(matrix[i][j]);
            if (matrix[i][j] > xmax[j]) xmax[j] = static_cast<double>(matrix[i][j]);
        }
    }
    targetMin = DBL_MAX;
    targetMax = -DBL_MIN;
    for (int j = 0; j < nRows; ++j) {
        if (target[j] < targetMin) targetMin = target[j];
        if (target[j] > targetMax) targetMax = target[j];
    }
}
