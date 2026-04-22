#include "kmeans.hpp"
#include "math/parallel.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_set>
#include <stdexcept>
#include <limits>
#include <random>

namespace ml
{

    KMeans::KMeans(int k, Matrix xTr, int max_iter)
    {

        /*
            There are two steps that we repeat until convergence or max_iter:

            1. Make each centroid the mean of the points assigned to it
            2. Assign each point to the cluster with the nearest centroid
        */

        if (k <= 0 || static_cast<size_t>(k) > xTr.numRows())
            throw std::invalid_argument("Invalid number of clusters.");

        this->k = k;
        this->xTr = xTr;

        // std::cout << "Training data: " << std::endl;

        // xTr.print();

        std::cout << "cols:             " << xTr.numCols() << std::endl;

        this->assignments = std::vector<int>(xTr.numRows());
        this->centroids = std::vector<Matrix>();
        const std::vector<Matrix> training_rows = xTr.rowsAsMatrices();

        // initialize the centroids to random points in the data
        std::random_device rand;
        std::mt19937 gen(rand());
        std::uniform_int_distribution<> distribution(0, xTr.numRows() - 1);
        std::unordered_set<int> do_not_reuse; // points that we don't want to add to the centroids again
        while (this->centroids.size() < static_cast<size_t>(k))
        {
            int idx = distribution(gen);
            if (do_not_reuse.insert(idx).second)
                this->centroids.emplace_back(training_rows[idx]);
        }

        // assign each point to a random cluster
        for (size_t i = 0; i < xTr.numRows(); ++i)
        {
            this->assignments[i] = static_cast<int>(i % static_cast<size_t>(k));
        }

        for (int i = 0; i < max_iter; i++)
        {

            bool b1 = fix_assignments_assign_centroids();
            bool b2 = fix_centroids_assign_assignments();
            if (!b1 && !b2)
            {
                break;
            }
        }
    }

    /*
        Below are two helper functions that perform the KMeans algorithm training
    */

    bool KMeans::fix_assignments_assign_centroids()
    {
        /*
            Fix the assignments
            Make each centroid the mean of the points assigned to it
        */
        struct CentroidAccumulator
        {
            std::vector<Matrix> centroids;
            std::vector<int> cluster_sizes;
        };

        const size_t feature_count = xTr.numCols();
        CentroidAccumulator totals = ctorch::parallel::parallel_reduce_items<CentroidAccumulator>(
            xTr.numRows(),
            feature_count,
            CentroidAccumulator{},
            [&](size_t begin, size_t end)
            {
                CentroidAccumulator partial;
                partial.centroids.assign(k, Matrix(1, feature_count));
                partial.cluster_sizes.assign(k, 0);

                for (size_t i = begin; i < end; ++i)
                {
                    const int cluster = assignments[i];
                    for (size_t j = 0; j < feature_count; ++j)
                    {
                        partial.centroids[cluster](0, j) += xTr(i, j);
                    }
                    ++partial.cluster_sizes[cluster];
                }

                return partial;
            },
            [](CentroidAccumulator lhs, const CentroidAccumulator &rhs)
            {
                if (lhs.centroids.empty())
                {
                    return rhs;
                }

                for (int cluster = 0; cluster < static_cast<int>(rhs.centroids.size()); ++cluster)
                {
                    for (size_t j = 0; j < rhs.centroids[cluster].numCols(); ++j)
                    {
                        lhs.centroids[cluster](0, j) += rhs.centroids[cluster](0, j);
                    }
                    lhs.cluster_sizes[cluster] += rhs.cluster_sizes[cluster];
                }

                return lhs;
            });

        bool changed = false;
        std::vector<Matrix> new_centroids(k, Matrix(1, feature_count));
        std::vector<int> cluster_sizes(k, 0);

        if (!totals.centroids.empty())
        {
            for (int cluster = 0; cluster < k; ++cluster)
            {
                new_centroids[cluster] = totals.centroids[cluster];
                cluster_sizes[cluster] = totals.cluster_sizes[cluster];
            }
        }

        for (int cluster = 0; cluster < this->k; ++cluster)
        {
            if (cluster_sizes[cluster] > 0)
            {
                for (size_t j = 0; j < feature_count; ++j)
                    new_centroids[cluster](0, j) /= cluster_sizes[cluster];
            }
            else
            {
                // If there are no points in the cluster, just make the centroid a random point in the data
                static thread_local std::mt19937 gen(std::random_device{}());
                std::uniform_int_distribution<size_t> distribution(0, this->xTr.numRows() - 1);
                const size_t random_idx = distribution(gen);
                for (size_t j = 0; j < feature_count; ++j)
                {
                    new_centroids[cluster](0, j) = this->xTr(random_idx, j);
                }
            }

            if (!(centroids[cluster] == new_centroids[cluster]))
                changed = true;
        }
        centroids = std::move(new_centroids);
        return changed;
    }

    bool KMeans::fix_centroids_assign_assignments()
    {
        std::vector<int> new_assignments(assignments.size());
        const bool changed = ctorch::parallel::parallel_reduce_items<bool>(
            this->xTr.numRows(),
            this->xTr.numCols() * static_cast<size_t>(k),
            false,
            [&](size_t begin, size_t end)
            {
                bool local_changed = false;

                // For each point, assign it to the cluster with the closest centroid.
                for (size_t point_num = begin; point_num < end; ++point_num)
                {
                    double min = std::numeric_limits<double>::max();
                    int best_cluster = 0;

                    for (int cluster = 0; cluster < k; ++cluster)
                    {
                        double distance_from_centroid = 0.0;
                        for (size_t j = 0; j < xTr.numCols(); ++j)
                        {
                            const double diff = xTr(point_num, j) - centroids[cluster](0, j);
                            distance_from_centroid += diff * diff;
                        }
                        distance_from_centroid = sqrt(distance_from_centroid);
                        if (distance_from_centroid < min)
                        {
                            min = distance_from_centroid;
                            best_cluster = cluster;
                        }
                    }

                    new_assignments[point_num] = best_cluster;
                    if (new_assignments[point_num] != assignments[point_num])
                    {
                        local_changed = true;
                    }
                }

                return local_changed;
            },
            [](bool lhs, bool rhs)
            {
                return lhs || rhs;
            });
        assignments = std::move(new_assignments);
        return changed;
    }

    std::vector<int> KMeans::getAssignments() const
    {
        return assignments;
    }
}
