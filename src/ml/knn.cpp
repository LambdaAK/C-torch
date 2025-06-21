#include "knn.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace ml
{

    // Constructor
    KNN::KNN(size_t k, Matrix xTr, Matrix yTr) : xTr(xTr), yTr(yTr), k(k)
    {
        // Make sure that $k$ is not 0
        if (k == 0)
        {
            throw std::invalid_argument("k must be greater than 0");
        }
    }

    int KNN::predict(Matrix x) const
    {
        // sample is a row vector
        // check that it has the same number of columns as xTr
        if (x.numCols() != xTr.numCols())
        {
            throw std::invalid_argument("Number of training samples and labels must be equal.");
        }

        // compute the Euclidean distance between the sample and each row in xTr

        // get the row vectors as matrices from xTr

        std::vector<Matrix> xTr_rows = xTr.rowsAsMatrices();

        // compute the Euclidean distance between the sample and each row in xTr

        std::map<double, int> distances;

        for (size_t i = 0; i < xTr.numRows(); ++i)
        {
            double distance = x.euclideanDistance(xTr_rows[i]);
            distances[distance] = i;
        }

        // sort the distances and get the k nearest neighbors

        std::vector<int> nearest_neighbors;

        for (const auto &pair : distances)
        {
            nearest_neighbors.push_back(pair.second);
        }

        nearest_neighbors.resize(k); // keep only the k nearest neighbors

        // get the labels of the k nearest neighbors

        std::vector<int> labels;

        for (int neighbor : nearest_neighbors)
        {
            labels.push_back(yTr(0, neighbor));
        }

        // count the frequency of each label

        std::map<int, int> label_counts;

        for (int label : labels)
        {
            label_counts[label]++;
        }

        // get the label with the highest frequency

        int max_count = 0;

        int predicted_label = -1;

        for (const auto &pair : label_counts)
        {
            if (pair.second > max_count)
            {
                max_count = pair.second;
                predicted_label = pair.first;
            }
        }

        return predicted_label;
    }

    // Calculate accuracy score on test data
    double KNN::score(const Matrix &xTe, const Matrix &yTe)
    {
        if (xTe.numRows() != yTe.numCols())
        {
            throw std::invalid_argument("Number of test samples and test labels must be equal.");
        }
        int correct = 0;

        std::vector<Matrix> xTe_rows = xTe.rowsAsMatrices();

        // classify each row

        for (size_t i = 0; i < xTe.numRows(); ++i)
        {
            int prediction = predict(xTe_rows[i]);
            if (prediction == yTe.at(0, i))
            { // yTe is a row vector, so row number is 0 and col number is i
                ++correct;
            }
        }

        return static_cast<double>(correct) / static_cast<double>(yTe.numCols());
    }

    // Getters and setters
    size_t KNN::getK() const
    {
        return k;
    }

    void KNN::setK(size_t new_k)
    {
        if (new_k == 0)
        {
            throw std::invalid_argument("k must be greater than 0");
        }
        k = new_k;
    }

} // namespace ml