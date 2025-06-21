#include <vector>
#include "../math/matrix.hpp"

namespace ml
{
    class KMeans
    {
    private:
        std::vector<int> assignments;  // map each point to a cluster
        std::vector<Matrix> centroids; // cluster centroid of each cluster
        int k;                         // number of clusters
        Matrix xTr;
        bool fix_assignments_assign_centroids();
        bool fix_centroids_assign_assignments();

    public:
        // Create and train the model
        KMeans(int k, Matrix xTr, int max_iter);

        // get the cluster assignments
        std::vector<int> getAssignments() const;
    };
}