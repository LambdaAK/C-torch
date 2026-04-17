#include "kmeans.hpp"
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

        if (k <= 0 || k > xTr.numRows())
            throw std::invalid_argument("Invalid number of clusters.");

        this->k = k;
        this->xTr = xTr;

        // std::cout << "Training data: " << std::endl;

        // xTr.print();

        std::cout << "cols:             " << xTr.numCols() << std::endl;

        this->assignments = std::vector<int>(xTr.numRows());
        this->centroids = std::vector<Matrix>();

        // initialize the centroids to random points in the data
        std::random_device rand;
        std::mt19937 gen(rand());
        std::uniform_int_distribution<> distribution(0, xTr.numRows() - 1);
        std::unordered_set<int> do_not_reuse; // points that we don't want to add to the centroids again
        while (this->centroids.size() < k)
        {
            int idx = distribution(gen);
            if (do_not_reuse.insert(idx).second)
                this->centroids.emplace_back(xTr.rowsAsMatrices()[idx]);
        }

        // assign each point to a random cluster
        for (int i = 0; i < xTr.numRows(); ++i)
        {
            this->assignments[i] = i % k;
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
        bool changed = false;
        std::vector<Matrix> new_centroids(k, Matrix(1, xTr.numCols()));
        std::vector<int> cluster_sizes(k, 0);

        for (size_t i = 0; i < xTr.numRows(); ++i)
        {
            int cluster = assignments[i];
            for (size_t j = 0; j < xTr.numCols(); ++j)
            {
                new_centroids[cluster](0, j) += xTr(i, j);
            }
            ++cluster_sizes[cluster];
        }

        for (int cluster = 0; cluster < this->k; ++cluster)
        {
            if (cluster_sizes[cluster] > 0)
            {
                for (size_t j = 0; j < xTr.numCols(); ++j)
                    new_centroids[cluster](0, j) /= cluster_sizes[cluster];
            }
            else
            {
                // If there are no points in the cluster, just make the centroid a random point in the data
                this->centroids[cluster] = this->xTr.rowsAsMatrices()[rand() % this->xTr.numRows()];
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
        bool changed = false;

        // for each point, assign it to the cluster with the closest cluster centroid
        // iterate through the points
        for (int point_num = 0; point_num < this->xTr.numRows(); ++point_num)
        {
            double min = std::numeric_limits<double>::max();
            int best_cluster = 0;

            // find the cluster with the closest centroid
            // iterate through the clusters
            for (int cluster = 0; cluster < k; ++cluster)
            {
                double distance_from_centroid = 0.0;
                for (size_t j = 0; j < xTr.numCols(); ++j)
                {
                    double diff = xTr(point_num, j) - centroids[cluster](0, j);
                    distance_from_centroid += diff * diff;
                }
                distance_from_centroid = sqrt(distance_from_centroid);
                if (distance_from_centroid < min)
                {
                    min = distance_from_centroid;
                    best_cluster = cluster;
                }
            }
            // assign the point to the cluster with the closest centroid
            new_assignments[point_num] = best_cluster;
            if (new_assignments[point_num] != assignments[point_num])
                changed = true;
        }
        assignments = std::move(new_assignments);
        return changed;
    }

    std::vector<int> KMeans::getAssignments() const
    {
        return assignments;
    }
}