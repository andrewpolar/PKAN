//Concept: Andrew Polar and Mike Poluektov
//Developer Andrew Polar

//License
//In case if end user finds the way of making a profit by using this code and earns
//billions of US dollars and meet developer bagging change in the street near McDonalds,
//he or she is not in obligation to buy him a sandwich.

//Symmetricity
//In case developer became rich and famous by publishing this code and meet misfortunate
//end user who went bankrupt by using this code, he is also not in obligation to buy
//end user a sandwich.

//Publications:
//https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://arxiv.org/abs/2305.08194
//https://arxiv.org/abs/2104.01714

//This is probabilistic KAN. DataHolder returns targets with probabilistic uncertainty,
//the model captures this uncertainty and returns the sample, for both true and modeled
//samples we estimate expectation and standard deviation and compare them.
//The names are 'sample' and 'monteCarlo'. 

#include <iostream>
#include <thread>
#include <vector>
#include <iterator>
#include <algorithm>
#include "DataHolder.h"
#include "KANAddendPL.h"
#include "Helper.h"
#include "Resorter.h"
#include "SlidingWindow.h"

std::unique_ptr<std::unique_ptr<KANAddendPL>[]> Training(
    std::unique_ptr<std::unique_ptr<double[]>[]>& inputs, 
    std::unique_ptr<double[]>& target,
    std::unique_ptr<double[]>& xmin,
    std::unique_ptr<double[]>& xmax,
    double zmin, double zmax,
    int nRecords, int nFeatures, int nEpochs, int nModels) {

    int innerPoints = 6;
    int outerPoints = 12;
    double muInnerPL = 0.01;
    double muOuterPL = 0.01;

    //initialization of piecewise linear model
    auto addends = std::make_unique<std::unique_ptr<KANAddendPL>[]>(nModels);
    for (int i = 0; i < nModels; ++i) {
        addends[i] = std::make_unique<KANAddendPL>(xmin, xmax, zmin/nModels, zmax/nModels, innerPoints,
            outerPoints, muInnerPL, muOuterPL, nFeatures);
    }
    auto residualError = std::make_unique<double[]>(nRecords);
    for (int epoch = 0; epoch < nEpochs; ++epoch) {
        double error2 = 0.0;
        int cnt = 0;
        for (int i = 0; i < nRecords; ++i) {
            double residual = target[i];
            for (int j = 0; j < nModels; ++j) {
                residual -= addends[j]->ComputeUsingInput(inputs[i]);
            }
            for (int j = 0; j < nModels; ++j) {
                addends[j]->UpdateUsingMemory(residual);
            }
            error2 += residual * residual;
            residualError[i] = residual;
            ++cnt;
        }
        if (0 == cnt) error2 = 0.0;
        else {
            error2 /= cnt;
            error2 = sqrt(error2);
            error2 /= (zmax - zmin);
        }
        printf("Training step %d, current relative RMSE %4.4f\n", epoch, error2);
    }
    return addends;
}

std::unique_ptr<std::unique_ptr<double[]>[]> GetIntermediateInput(std::unique_ptr<std::unique_ptr<KANAddendPL>[]>& addends,
    std::unique_ptr<std::unique_ptr<double[]>[]>& inputs,
    int nModels, int nRecords) {

    auto intermediate = std::make_unique<std::unique_ptr<double[]>[]>(nRecords);
    for (int i = 0; i < nRecords; ++i) {
        intermediate[i] = std::make_unique<double[]>(nModels);
        for (int j = 0; j < nModels; ++j) {
            intermediate[i][j] = addends[j]->GetIntermediate(inputs[i]);
        }
    }
    return intermediate;
}

double GetMean(std::unique_ptr<double[]>& x, int size) {
    double mean = 0.0;
    for (int i = 0; i < size; ++i) {
        mean += x[i];
    }
    return mean / size;
}

double GetSTD(std::unique_ptr<double[]>& x, int size, double mean) {
    double std = 0.0;
    for (int i = 0; i < size; ++i) {
        std += (x[i] - mean) * (x[i] - mean);
    }
    std /= size;
    return sqrt(std);
}

int main() {
    srand((unsigned int)time(NULL));
    auto dataHolder = std::make_unique<DataHolder>();
    bool status = dataHolder->ReadDataMFormula();
    if (false == status) {
        printf("Failed to open file");
        exit(0);
    }

    clock_t start_application = clock();
    auto helper = std::make_unique<Helper>();
    helper->Shuffle(dataHolder->inputs, dataHolder->target, dataHolder->nRecords, dataHolder->nFeatures);

    auto xmin = std::make_unique<double[]>(dataHolder->nFeatures);
    auto xmax = std::make_unique<double[]>(dataHolder->nFeatures);
    double targetMin;
    double targetMax;

    helper->FindMinMax(xmin, xmax, targetMin, targetMax, dataHolder->inputs, dataHolder->target,
        dataHolder->nRecords, dataHolder->nFeatures);

    int nModels = 11;
    int nEpochs = 10;
    auto addends = Training(dataHolder->inputs, dataHolder->target, xmin, xmax, targetMin, targetMax, 
        dataHolder->nRecords, dataHolder->nFeatures, nEpochs, nModels);

    auto intermediate = GetIntermediateInput(addends, dataHolder->inputs, nModels, dataHolder->nRecords);

    auto resorter = std::make_unique<Resorter>();
    resorter->Resort(intermediate, dataHolder->target, 1, dataHolder->nRecords, nModels, 12);
    resorter->Resort(intermediate, dataHolder->target, 2, dataHolder->nRecords, nModels, 12);
    resorter->Resort(intermediate, dataHolder->target, 4, dataHolder->nRecords, nModels, 12);
    resorter->Resort(intermediate, dataHolder->target, 8, dataHolder->nRecords, nModels, 12);
    resorter->Resort(intermediate, dataHolder->target, 16, dataHolder->nRecords, nModels, 12);
 
    auto slidingWindow = std::make_unique<SlidingWindow>();
    slidingWindow->BuildEnsemble(intermediate, dataHolder->target, dataHolder->nRecords, nModels, 400, 200, 7);
  
    clock_t end_PWL_training = clock();
    printf("Time for training %2.3f sec.\n", (double)(end_PWL_training - start_application) / CLOCKS_PER_SEC);

    //testing of probabilistic KAN
    double stdError = 0.0;
    double meanError = 0.0;
    double minMean = DBL_MAX;
    double maxMean = -DBL_MIN;
    double minSTD = DBL_MAX;
    double maxSTD = -DBL_MIN;
    int N = 100;
    int nU = slidingWindow->GetSampleSize();
    for (int i = 0; i < N; ++i) {
        auto input = dataHolder->makeRandomInput();
        auto middle = std::make_unique<double[]>(nModels);
        for (int j = 0; j < nModels; ++j) {
            middle[j] = addends[j]->GetIntermediate(input);
        }

        //these are two samples, first is return by probabilistic model
        //and second is so-called true aleatoric data
        auto sample = slidingWindow->GetSample(middle, nModels);
        auto monteCarlo = dataHolder->GetEnsemble(input, 1024);
        ///////////////////////////////////////////////////////

        //below we evaluate accuracy
        double meanSample = GetMean(sample, nU);
        double meanMonteCarlo = GetMean(monteCarlo, 1024);
        double stdSample = GetSTD(sample, nU, meanSample);
        double stdMonteCarlo = GetSTD(monteCarlo, 1024, meanMonteCarlo);
        if (meanMonteCarlo > maxMean) maxMean = meanMonteCarlo;
        if (meanMonteCarlo < minMean) minMean = meanMonteCarlo;
        if (stdMonteCarlo > maxSTD) maxSTD = stdMonteCarlo;
        if (stdMonteCarlo < minSTD) minSTD = stdMonteCarlo;
        meanError += (meanSample - meanMonteCarlo) * (meanSample - meanMonteCarlo);
        stdError += (stdSample - stdMonteCarlo) * (stdSample - stdMonteCarlo);
    }
    meanError /= N;
    meanError = sqrt(meanError);
    meanError /= (maxMean - minMean);
    stdError /= N;
    stdError = sqrt(stdError);
    stdError /= (maxSTD - minSTD);
    printf("Mean relative error for mean ensemble of %d and mean MonteCarlo %6.4f\n", nU, meanError);
    printf("Mean relative error for STD  ensemble of %d and STD  MonteCarlo %6.4f\n", nU, stdError);
}