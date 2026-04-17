#include <vector>
#include "math/matrix.hpp"

namespace ml
{
    /**
     * @brief K-Means clustering model trained at construction time.
     *
     * Input matrix layout is sample-major (`num_samples x num_features`).
     */
    class KMeans
    {
    private:
        std::vector<int> assignments;  ///< Cluster assignment per sample index.
        std::vector<Matrix> centroids; ///< Current centroid vectors (each a row matrix).
        int k;                         ///< Number of clusters.
        Matrix xTr;                    ///< Cached training features.

        /**
         * @brief Reassigns points to nearest centroid, then recomputes centroids.
         * @return `true` if state changed, otherwise `false`.
         */
        bool fix_assignments_assign_centroids();

        /**
         * @brief Recomputes centroids from current assignments, then reassigns points.
         * @return `true` if state changed, otherwise `false`.
         */
        bool fix_centroids_assign_assignments();

    public:
        /**
         * @brief Builds and trains K-Means on the given dataset.
         * @param k Number of clusters.
         * @param xTr Training samples `(num_samples, num_features)`.
         * @param max_iter Maximum alternating-optimization iterations.
         */
        KMeans(int k, Matrix xTr, int max_iter);

        /**
         * @brief Returns learned cluster index per training sample.
         * @return Vector of assignment ids in `[0, k - 1]`.
         */
        std::vector<int> getAssignments() const;
    };
}
