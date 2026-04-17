#ifndef KNN_HPP
#define KNN_HPP

#include <vector>
#include "../math/matrix.hpp"

namespace ml
{
    class KNN
    {
    private:
        // Training data
        Matrix xTr; // Features
        Matrix yTr; // Labels
        size_t k; // number of nearest neighbors

    public:
        // Constructor
        KNN(size_t k, Matrix xTr, Matrix yTr);
        // Predict the label of a test sample
        int predict(const Matrix &x) const;
        // Compute the accuracy of a model over some test set of samples
        double score(const Matrix &xTe, const Matrix &yTe);
        // return k
        size_t getK() const;
        // set k
        void setK(size_t new_k);
    };

}

#endif // KNN_HPP
