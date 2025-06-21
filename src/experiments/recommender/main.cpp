#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <curl/curl.h>

#include "json/single_include/nlohmann/json.hpp" // or use <nlohmann/json.hpp> if preferred
#include "csv.h"
#include "matplotlibcpp.h"

#include "../../math/matrix.hpp"
#include "../../ml/kmeans.hpp"
#include "../../ml/mab.hpp"
#include "../../ml/ucb.hpp"
#include "../../ml/pca.hpp"
#include "../../ml/pca.hpp"

// The following code is adapted from https://github.com/smaltby/spotify-api-plusplus
// This is to get Spotify queries using C++

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *output)
{
    size_t total_size = size * nmemb;
    output->append((char *)contents, total_size);
    return total_size;
}

std::string urlEncode(const std::string &str)
{
    CURL *curl = curl_easy_init();
    char *encoded = curl_easy_escape(curl, str.c_str(), str.length());
    std::string result(encoded);
    curl_free(encoded);
    curl_easy_cleanup(curl);
    return result;
}

class SimpleJsonParser
{
public:
    static std::string getStringValue(const std::string &json, const std::string &key)
    {
        std::string keyPattern = "\"" + key + "\"";
        size_t keyPos = json.find(keyPattern);

        if (keyPos == std::string::npos)
        {
            return "";
        }

        size_t colonPos = json.find(":", keyPos);
        if (colonPos == std::string::npos)
        {
            return "";
        }

        size_t valueStart = json.find("\"", colonPos);
        if (valueStart == std::string::npos)
        {
            return "";
        }

        size_t valueEnd = json.find("\"", valueStart + 1);
        if (valueEnd == std::string::npos)
        {
            return "";
        }

        return json.substr(valueStart + 1, valueEnd - valueStart - 1);
    }

    static std::string extractTrackUrl(const std::string &json)
    {
        size_t tracksPos = json.find("\"tracks\"");
        if (tracksPos == std::string::npos)
        {
            return "";
        }

        size_t itemsPos = json.find("\"items\"", tracksPos);
        if (itemsPos == std::string::npos)
        {
            return "";
        }

        size_t itemsStartPos = json.find("[", itemsPos);
        if (itemsStartPos == std::string::npos)
        {
            return "";
        }

        if (json.find("[]", itemsPos) == itemsStartPos)
        {
            return "";
        }

        size_t hrefPos = json.find("\"href\"", itemsStartPos);
        if (hrefPos == std::string::npos)
        {
            return "";
        }

        size_t externalUrlsPos = json.find("\"external_urls\"", itemsStartPos);
        if (externalUrlsPos == std::string::npos || externalUrlsPos > json.find("]", itemsStartPos))
        {
            return "";
        }

        size_t spotifyPos = json.find("\"spotify\"", externalUrlsPos);
        if (spotifyPos == std::string::npos || spotifyPos > json.find("}", externalUrlsPos))
        {
            return "";
        }

        size_t urlValueStart = json.find("\"", spotifyPos + 10) + 1;
        size_t urlValueEnd = json.find("\"", urlValueStart);

        if (urlValueStart == std::string::npos || urlValueEnd == std::string::npos)
        {
            return "";
        }

        std::string url = json.substr(urlValueStart, urlValueEnd - urlValueStart);

        if (url.find("/track/") == std::string::npos)
        {
            size_t trackUrlPos = json.find("/track/", urlValueEnd);
            if (trackUrlPos != std::string::npos)
            {
                size_t trackUrlStart = json.rfind("\"", trackUrlPos) + 1;
                size_t trackUrlEnd = json.find("\"", trackUrlPos);

                if (trackUrlStart != std::string::npos && trackUrlEnd != std::string::npos)
                {
                    return json.substr(trackUrlStart, trackUrlEnd - trackUrlStart);
                }
            }
        }

        return url;
    }

    static std::string extractImageUrl(const std::string &json)
    {
        size_t imagesPos = json.find("\"images\"");
        if (imagesPos == std::string::npos)
        {
            return "";
        }

        size_t imagesArrayStart = json.find("[", imagesPos);
        if (imagesArrayStart == std::string::npos)
        {
            return "";
        }

        size_t firstImageStart = json.find("{", imagesArrayStart);
        if (firstImageStart == std::string::npos)
        {
            return "";
        }

        size_t urlPos = json.find("\"url\"", firstImageStart);
        if (urlPos == std::string::npos)
        {
            return "";
        }

        size_t urlValueStart = json.find("\"", urlPos + 6) + 1;
        size_t urlValueEnd = json.find("\"", urlValueStart);

        if (urlValueStart == std::string::npos || urlValueEnd == std::string::npos)
        {
            return "";
        }

        return json.substr(urlValueStart, urlValueEnd - urlValueStart);
    }

    static void extractTrackInfo(const std::string &json, std::string &trackName, std::string &artistName)
    {
        size_t tracksPos = json.find("\"tracks\"");
        if (tracksPos == std::string::npos)
            return;

        size_t itemsPos = json.find("\"items\"", tracksPos);
        if (itemsPos == std::string::npos)
            return;

        size_t itemsStartPos = json.find("[", itemsPos);
        if (itemsStartPos == std::string::npos)
            return;

        size_t trackObjStart = json.find("{", itemsStartPos);
        if (trackObjStart == std::string::npos)
            return;

        size_t namePos = json.find("\"name\"", trackObjStart);
        if (namePos != std::string::npos)
        {
            size_t nameStart = json.find("\"", namePos + 6) + 1;
            size_t nameEnd = json.find("\"", nameStart);
            if (nameStart != std::string::npos && nameEnd != std::string::npos)
            {
                trackName = json.substr(nameStart, nameEnd - nameStart);
            }
        }

        size_t artistsPos = json.find("\"artists\"", trackObjStart);
        if (artistsPos != std::string::npos)
        {
            size_t artistObjStart = json.find("{", artistsPos);
            if (artistObjStart != std::string::npos)
            {
                size_t artistNamePos = json.find("\"name\"", artistObjStart);
                if (artistNamePos != std::string::npos)
                {
                    size_t artistNameStart = json.find("\"", artistNamePos + 6) + 1;
                    size_t artistNameEnd = json.find("\"", artistNameStart);
                    if (artistNameStart != std::string::npos && artistNameEnd != std::string::npos)
                    {
                        artistName = json.substr(artistNameStart, artistNameEnd - artistNameStart);
                    }
                }
            }
        }
    }
};

bool downloadImage(const std::string &url, const std::string &outputPath)
{
    CURL *curl;
    FILE *fp;
    CURLcode res;
    bool success = false;

    curl = curl_easy_init();
    if (curl)
    {
        fp = fopen(outputPath.c_str(), "wb");
        if (fp)
        {
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
            curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

            res = curl_easy_perform(curl);
            if (res == CURLE_OK)
            {
                success = true;
            }
            else
            {
                std::cerr << "Failed to download image: " << curl_easy_strerror(res) << std::endl;
            }

            fclose(fp);
        }
        curl_easy_cleanup(curl);
    }

    return success;
}

void searchSpotifyTrack(const std::string &trackName, const std::string &authToken)
{
    CURL *curl;
    CURLcode res;
    std::string readBuffer;
    std::string spotifyUrl = "";
    std::string imageUrl = "";

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl)
    {
        std::string encodedTrackName = urlEncode(trackName);

        std::string url = "https://api.spotify.com/v1/search?q=" + encodedTrackName + "&type=track&limit=1";

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        struct curl_slist *headers = NULL;
        std::string authHeader = "Authorization: Bearer " + authToken;
        headers = curl_slist_append(headers, authHeader.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK)
        {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
        else
        {
            spotifyUrl = SimpleJsonParser::extractTrackUrl(readBuffer);

            imageUrl = SimpleJsonParser::extractImageUrl(readBuffer);

            std::string foundTrackName, artistName;
            SimpleJsonParser::extractTrackInfo(readBuffer, foundTrackName, artistName);

            if (!foundTrackName.empty() && !artistName.empty())
            {
                std::cout << "Found: " << foundTrackName << " by " << artistName << std::endl;

                if (!spotifyUrl.empty())
                {
                    std::cout << "Spotify URL: " << spotifyUrl << std::endl;
                }

                if (!imageUrl.empty())
                {
                    std::cout << "Album Image URL: " << imageUrl << std::endl;

                    // Save the image to a file
                    std::string imageFilename = "album_cover.jpg";
                    if (downloadImage(imageUrl, imageFilename))
                    {
                        std::cout << "Album cover saved as: " << imageFilename << std::endl;

// On Linux/macOS systems, you could try to open the image with a system command
#ifdef __unix__
                        std::cout << "Attempting to display the image..." << std::endl;
                        std::string cmd = "xdg-open " + imageFilename + " &";
                        system(cmd.c_str());
#elif defined(_WIN32)
                        std::cout << "Attempting to display the image..." << std::endl;
                        std::string cmd = "start " + imageFilename;
                        system(cmd.c_str());
#elif defined(__APPLE__)
                        std::cout << "Attempting to display the image..." << std::endl;
                        std::string cmd = "open " + imageFilename;
                        system(cmd.c_str());
#else
                        std::cout << "Image display not supported on this platform." << std::endl;
#endif
                    }
                }
            }
            else if (spotifyUrl.empty())
            {
                std::cout << "No tracks found for the given query." << std::endl;
            }
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }

    curl_global_cleanup();
}

// The following code is adapted from https://github.com/lava/matplotlib-cpp
// Used for visualizing data

namespace plt = matplotlibcpp;

void visualize_2d_data(const Matrix &data, const std::vector<int> &assignments, int k)
{
    try
    {
        if (data.numCols() != 2)
        {
            throw std::runtime_error("Data must be n x 2 for 2D visualization");
        }

        if (data.numRows() != assignments.size())
        {
            throw std::runtime_error("Data and assignments must have the same size");
        }

        std::vector<std::string> colors = {
            "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
            "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"};

        std::vector<double> full_x;

        for (int cluster = 0; cluster < k; cluster++)
        {
            std::vector<double> x, y;

            for (size_t i = 0; i < data.numRows(); i++)
            {
                if (assignments[i] == cluster)
                {
                    full_x.push_back(data(i, 0));
                    x.push_back(data(i, 0));
                    y.push_back(data(i, 1));
                }
            }

            if (x.empty())
            {
                std::cout << "CLUSTER: " << cluster << std::endl;
                continue;
            }

            std::map<std::string, std::string> keywords;
            keywords["label"] = "Cluster " + std::to_string(cluster);

            try
            {
                std::map<std::string, std::string> kwargs;
                kwargs["color"] = colors[cluster];
                plt::scatter(x, y, 3.0, kwargs);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error in scatter plot: " << e.what() << std::endl;
                plt::plot(x, y, "o");
            }
        }

        try
        {
            plt::title("2D Data Visualization");
            plt::xlabel("Dimension 1");
            plt::ylabel("Dimension 2");
            plt::legend();
            plt::grid(true);

            std::sort(full_x.begin(), full_x.end());

            size_t n = full_x.size();
            double lower = full_x[n * 1 / 100];
            double upper = full_x[n * 99 / 100];

            double padding = 0.05 * (upper - lower);
            lower -= padding;
            upper += padding;
            plt::xlim(lower, upper);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Warning: Could not set labels: " << e.what() << std::endl;
        }

        plt::show();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Visualization error: " << e.what() << std::endl;
    }
}

// End of adapted code

double euclidean_distance(const std::vector<double> &a, const std::vector<double> &b)
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(sum);
}

std::vector<double> get_row(const Matrix &data, int i)
{
    std::vector<double> row(data.numCols());
    for (int j = 0; j < data.numCols(); ++j)
        row[j] = data(i, j);
    return row;
}

std::pair<double, double> calculate_column_stats(const Matrix &data, size_t column_index)
{
    if (data.numCols() <= column_index)
    {
        throw std::out_of_range("Column index out of range");
    }

    const size_t num_rows = data.numRows();
    if (num_rows == 0)
    {
        return {0.0, 0.0};
    }

    double sum = 0.0;
    for (size_t i = 0; i < num_rows; ++i)
    {
        sum += data(i, column_index);
    }
    double mean = sum / num_rows;

    double sum_squared_diff = 0.0;
    for (size_t i = 0; i < num_rows; ++i)
    {
        double diff = data(i, column_index) - mean;
        sum_squared_diff += diff * diff;
    }
    double variance = sum_squared_diff / num_rows;
    double std_dev = std::sqrt(variance);

    return {mean, std_dev};
}

std::unordered_map<int, std::vector<int>> get_indices_map(const std::vector<int> &assignments)
{
    std::unordered_map<int, std::vector<int>> get_indices;

    for (size_t i = 0; i < assignments.size(); ++i)
    {
        get_indices[assignments[i]].push_back(i);
    }

    return get_indices;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double compute_reward(const std::vector<double> &user_vector, const std::vector<double> song_vector)
{
    if (user_vector.size() != 7 || song_vector.size() != 7)
    {
        throw std::invalid_argument("Both vectors must be of size 7.");
    }

    double dot_product = 0.0;
    double norm_user = 0.0;
    double norm_song = 0.0;

    for (size_t i = 0; i < 7; ++i)
    {
        dot_product += user_vector[i] * song_vector[i];
        norm_user += user_vector[i] * user_vector[i];
        norm_song += song_vector[i] * song_vector[i];
    }

    if (norm_user == 0.0 || norm_song == 0.0)
    {
        return 1.0;
    }

    double cosine_similarity = dot_product / (std::sqrt(norm_user) * std::sqrt(norm_song));

    double score = sigmoid(20.0 * (cosine_similarity - 0.55));

    return 1.0 + 9.0 * score;
}

using json = nlohmann::json;

int main()
{

    json results;

    bool use_eps_greedy = false;

    std::string access_token = "";

    int k = 20;
    float initial_eps = 0.9f, final_eps = 0.1f;
    float decay_rate = 0.01f;
    float epsilon = initial_eps;
    int num_iters = 500;
    int num_samples = 170000;
    int reupdate_iter = 5;
    int sliding_window = 10;
    bool run_auto = true;
    int max_iters = 1000;
    int max_recent = 200;
    bool proj_before_cluster = true;
    int proj_dim = 6;
    std::unordered_map<int, std::deque<int>> recently_used;

    float total_reward = 0.0f;
    std::vector<float> total_rewards;
    std::vector<float> normalized_rewards;
    std::vector<float> sliding_window_avgs;

    io::CSVReader<19>
        in("./processed_data.csv");
    in.read_header(io::ignore_extra_column, "valence", "year", "acousticness", "artists", "danceability", "duration_ms", "energy", "explicit", "id", "instrumentalness", "key", "liveness", "loudness",
                   "mode", "name", "popularity", "release_date", "speechiness", "tempo");

    float col1;
    float col2;
    float col3;
    std::string col4; // artists
    float col5;
    float col6;
    float col7;
    float col8;
    std::string col9; // id
    float col10;
    float col11;
    float col12;
    float col13;
    float col14;
    std::string col15; // name
    float col16;
    std::string col17;
    float col18;
    float col19;

    Matrix xTr(num_samples, 7);

    size_t row = 0;

    // name and artist
    std::vector<std::pair<std::string, std::string>> non_numerical;

    while (in.read_row(col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19))
    {
        non_numerical.push_back(std::make_pair(col15, col4));

        xTr(row, 0) = col1;  // valence
        xTr(row, 1) = col3;  // acousticness
        xTr(row, 2) = col5;  // danceability
        xTr(row, 3) = col7;  // energy
        xTr(row, 4) = col10; // instrumentalness
        xTr(row, 5) = col12; // liveliness
        xTr(row, 6) = col18; // speechiness

        row++;

        if (row == num_samples)
        {
            break;
        }
    }

    // define user vector here
    std::vector<double> user_vec = {0.7, 0.2, 0.2, 0.3, 0.9, 0.0, 0.5};

    Matrix xTr_k_means = Matrix::l2_norm_cols(xTr);
    Matrix xTr_PCA = Matrix::center_cols(xTr);

    std::vector<int> assignments;

    // clusters are one-indexed
    if (!proj_before_cluster)
    {
        ml::KMeans clusters(k, xTr_k_means, num_iters);
        assignments = clusters.getAssignments();
    }
    else
    {
        ml::PCA projector(xTr_PCA);
        Matrix W = projector.compute_projection_mat(proj_dim);
        Matrix xTr_PCA_transformed = xTr_PCA * W;
        ml::KMeans clusters(k, xTr_PCA_transformed, num_iters);
        assignments = clusters.getAssignments();
    }

    std::vector<double> cluster_sum(k, 0.0);
    std::vector<double> cluster_sum_sq(k, 0.0); // sum of squares for std dev
    std::vector<int> cluster_count(k, 0);

    for (int i = 0; i < num_samples; ++i)
    {
        int cluster_id = assignments[i];

        std::vector<double> song_vec(7);
        song_vec[0] = xTr(i, 0); // valence
        song_vec[1] = xTr(i, 1); // acousticness
        song_vec[2] = xTr(i, 2); // danceability
        song_vec[3] = xTr(i, 3); // energy
        song_vec[4] = xTr(i, 4); // instrumentalness
        song_vec[5] = xTr(i, 5); // liveness
        song_vec[6] = xTr(i, 6); // speechiness

        double reward = compute_reward(user_vec, song_vec);
        cluster_sum[cluster_id] += reward;
        cluster_sum_sq[cluster_id] += reward * reward;
        cluster_count[cluster_id] += 1;
    }

    std::cout << "\nMean reward and standard deviation per cluster:\n";
    for (int i = 0; i < k; ++i)
    {
        if (cluster_count[i] > 0)
        {
            double mean = cluster_sum[i] / cluster_count[i];
            double mean_sq = cluster_sum_sq[i] / cluster_count[i];
            double std_dev = std::sqrt(mean_sq - mean * mean);
            std::cout << "Cluster " << (i + 1) << ": mean = " << mean
                      << ", std = " << std_dev << std::endl;
        }
        else
        {
            std::cout << "Cluster " << (i + 1) << ": no samples.\n";
        }
    }

    Matrix xTr_2D;
    if (!run_auto)
    {
        ml::PCA visualize(xTr_PCA);
        Matrix W = visualize.compute_projection_mat(2);

        Matrix xTr_2D = xTr_PCA * W;
        visualize_2d_data(xTr_2D, assignments, k);
    }

    ml::UCB ucb_bandit(k);
    ml::MAB eps_greedy(k, epsilon);

    std::string algorithm_name = use_eps_greedy ? "eps_greedy" : "ucb";
    std::cout << "\nRunning experiment with " << algorithm_name << std::endl;

    std::unordered_map<int, std::vector<int>> get_indices = get_indices_map(assignments);
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<float> rew_window;
    int turn = 1;

    results["parameters"] = {
        {"algorithm", algorithm_name},
        {"k", k},
        {"initial_eps", initial_eps},
        {"final_eps", final_eps},
        {"decay_rate", decay_rate},
        {"max_iters", max_iters},
        {"sliding_window", sliding_window}};

    while (turn <= max_iters)
    {
        if (turn % reupdate_iter == 0)
        {
            if (use_eps_greedy)
            {
                epsilon = std::max(final_eps, initial_eps * std::exp(-decay_rate * turn));
                eps_greedy.set_epsilon(epsilon);
                std::cout << "Epsilon: " << epsilon << std::endl;
            }

            float window_avg = 0.0f;
            if (!rew_window.empty())
            {
                window_avg = std::accumulate(rew_window.begin(), rew_window.end(), 0.0f) / rew_window.size();
                sliding_window_avgs.push_back(window_avg);
                std::cout << "Window Average: " << window_avg << std::endl;
            }
        }

        int arm;
        if (use_eps_greedy)
        {
            arm = eps_greedy.select_arms();
        }
        else
        {
            arm = ucb_bandit.select_arms();
        }

        if (turn % reupdate_iter == 0)
        {
            std::cout << "ARM: " << arm << std::endl;
        }

        if (get_indices[arm].size() == 0)
        {
            continue;
        }

        const auto &indices = get_indices[arm];
        std::vector<int> unused;
        for (int idx : indices)
        {
            auto &deque_ref = recently_used[arm];
            if (std::find(deque_ref.begin(), deque_ref.end(), idx) == deque_ref.end())
            {
                unused.push_back(idx);
            }
        }

        if (unused.empty())
        {
            std::cout << "Recommendations exhausted for arm " << arm << std::endl;
            continue;
        }

        std::uniform_int_distribution<> dist(0, unused.size() - 1);
        int rec_index = unused[dist(gen)];
        auto &deque_ref = recently_used[arm];
        deque_ref.push_back(rec_index);
        if (deque_ref.size() > max_recent)
        {
            deque_ref.pop_front();
        }

        int choice = compute_reward(user_vec, {xTr(rec_index, 0), xTr(rec_index, 1), xTr(rec_index, 2),
                                               xTr(rec_index, 3), xTr(rec_index, 4), xTr(rec_index, 5),
                                               xTr(rec_index, 6)});

        total_reward += choice;
        total_rewards.push_back(total_reward);
        normalized_rewards.push_back(total_reward / turn);

        rew_window.push_back(choice);
        if (rew_window.size() > sliding_window)
        {
            rew_window.erase(rew_window.begin());
        }

        if (use_eps_greedy)
        {
            eps_greedy.update(arm, choice);
        }
        else
        {
            ucb_bandit.update(arm, choice);
        }

        turn++;
    }

    results["results"] = {
        {"total_rewards", total_rewards},
        {"normalized_rewards", normalized_rewards},
        {"sliding_window_avgs", sliding_window_avgs},
        {"final_reward", total_reward},
        {"average_reward", total_reward / max_iters}};

    std::string filename = algorithm_name + "_k=" + std::to_string(k) + "_dim=" + std::to_string(proj_dim) + "_results.json";
    std::ofstream file(filename);
    file << std::setw(4) << results << std::endl;
    file.close();

    std::cout << "Experiment complete for " << algorithm_name << std::endl;
    std::cout << "Final total reward: " << total_reward << std::endl;
    std::cout << "Final normalized reward: " << (total_reward / max_iters) << std::endl;
    std::cout << "Results saved to " << filename << std::endl;

    return 0;
}
