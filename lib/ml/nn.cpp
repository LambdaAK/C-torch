#include "nn.hpp"

#include <cmath>
#include <stdexcept>


namespace ml {
    Matrix ReLULayer::forward(const Matrix& input) {
        return input.relu();
    }

    Matrix ReLULayer::backward(const Matrix& input) {
        return input.relu_deriv();
    }

    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> ReLULayer::get_params() {
        return {nullptr, nullptr};
    }

    Matrix SigmoidLayer::forward(const Matrix& input) {
        return input.sigmoid();
    }

    Matrix SigmoidLayer::backward(const Matrix& input) {
        return input.sigmoid_deriv();
    }

    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> SigmoidLayer::get_params() {
        return {nullptr, nullptr};
    }

    Matrix TanhLayer::forward(const Matrix& input) {
        return input.tanh();
    }

    Matrix TanhLayer::backward(const Matrix& input) {
        return input.tanh_deriv();
    }

    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> TanhLayer::get_params() {
        return {nullptr, nullptr};
    }

    LinearLayer::LinearLayer(int input_dim, int output_dim) : input_dim(input_dim), output_dim(output_dim) {
        Matrix weights_mat(output_dim, input_dim);
        Matrix bias_mat(output_dim, 1);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        
        double limit = std::sqrt(6.0 / (input_dim + output_dim));
        std::uniform_real_distribution<double> dist(-limit, limit);
        
        for (size_t i = 0; i < output_dim; i++) {
            for (size_t j = 0; j < input_dim; j++) {
                weights_mat(i, j) = dist(gen);
            }
        }

        weights = std::make_shared<Matrix>(weights_mat);
        bias = std::make_shared<Matrix>(bias_mat);
    }

    int LinearLayer::get_input_dim() {
        return input_dim;
    }

    int LinearLayer::get_output_dim() {
        return output_dim;
    }

    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> LinearLayer::get_params() {
        return {weights, bias};
    }

    Matrix LinearLayer::forward(const Matrix& input) {
        return (*weights) * input + (*bias);
    }

    Matrix LinearLayer::backward(const Matrix& input) {
        return input;
    }

    void Sequential::add_layer(std::shared_ptr<Layer> layer) {
        layers.push_back(layer);
    }

    std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> Sequential::parameters() {
        if (grads.first.size() == 0) { // initialize gradient pointers
            for (int i = layers.size() - 1; i >= 0; i -= 2) {
                if (auto linear = std::dynamic_pointer_cast<LinearLayer>(layers[i])) {
                    grads.first.insert(grads.first.begin(), std::make_shared<Matrix>(linear->get_output_dim(), linear->get_input_dim()));
                    grads.second.insert(grads.second.begin(), std::make_shared<Matrix>(linear->get_output_dim(), 1));
                }
            }
        }
        std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> params;
        for (size_t i = 0; i < layers.size(); i += 2) {
            auto [p1, p2] = layers[i]->get_params(); // pointer to W and b
            params.push_back(std::make_pair(p1, grads.first[std::floor(i / 2)]));
            params.push_back(std::make_pair(p2, grads.second[std::floor(i / 2)]));
        }

        return params;
    }

    bool Sequential::copy_parameters_from(const Sequential& other) {
        if (layers.size() != other.layers.size()) {
            return false;
        }

        for (size_t i = 0; i < layers.size(); ++i) {
            auto* this_linear = dynamic_cast<LinearLayer*>(layers[i].get());
            auto* other_linear = dynamic_cast<LinearLayer*>(other.layers[i].get());

            if ((this_linear == nullptr) != (other_linear == nullptr)) {
                return false;
            }

            if (this_linear && other_linear) {
                if (this_linear->get_input_dim() != other_linear->get_input_dim() ||
                    this_linear->get_output_dim() != other_linear->get_output_dim()) {
                    return false;
                }

                auto [this_w, this_b] = this_linear->get_params();
                auto [other_w, other_b] = other_linear->get_params();
                *this_w = *other_w;
                *this_b = *other_b;
            }
        }

        return true;
    }

    Matrix Sequential::forward(const Matrix& x) {
        // apply each layer in sequence
        std::vector<Matrix> activations;
        Matrix output = x;
        activations.push_back(x); // z0
        for (size_t i = 0; i < layers.size(); i++) {
            output = layers[i]->forward(output);
            activations.push_back(output);
        }
        
        train_activations = activations;
        
        return output;
    }
    
    // takes as input list of activations from training (a_L, z_L are last)
    // Note: Must have called forward first!
    void Sequential::backward(const Matrix& dL_daL) {
        // add check to ensure alternating
        for (size_t i = 0; i < layers.size(); ++i) {
            if (i % 2 == 0) {
                if (dynamic_cast<LinearLayer*>(layers[i].get()) == nullptr) {
                    throw std::runtime_error("Layer at index " + std::to_string(i) + " is expected to be a LinearLayer.");
                }
            } 
            else {
                if (dynamic_cast<ActivationLayer*>(layers[i].get()) == nullptr) {
                    throw std::runtime_error("Layer at index " + std::to_string(i) + " is expected to be an ActivationLayer.");
                }
            }
        }

        // these were computed in the most recent forward pass
        const std::vector<Matrix>& activations = train_activations;

        std::pair<std::vector<Matrix>, std::vector<Matrix>> out;
        // Matrix dzL_daL = layers.back()->backward(activations[activations.size() - 1]);
        // std::cout << dL_dzL.numRows() << " " << dL_dzL.numCols() << " " << dzL_daL.numRows() << " " << dzL_daL.numCols() << std::endl;
        // std::shared_ptr<Matrix> deltal = std::make_shared<Matrix>(dL_dzL.elm_wise_product(dzL_daL));
        Matrix deltal = dL_daL;
        for (int l = layers.size() - 1; l >= 0; l -= 2) {
            // std::cout << "L HERE: " << l << std::endl;
            Matrix dL_dWl = deltal * (activations[l].transpose());
            out.first.insert(out.first.begin(), dL_dWl);
            // (*dL_dWl).print();
            out.second.insert(out.second.begin(), deltal);
            // (*deltal).print();
            if (l > 1) {
                LinearLayer* linear_layer = dynamic_cast<LinearLayer*>(layers[l].get());
                if (linear_layer) {
                    // can access weights
                    // std::cout << "==" << std::endl;
                    auto [p1, _] = linear_layer->get_params();
                    // (*p1).print();
                    // std::cout << "==" << std::endl;
                    Matrix dL_dz = (*p1).transpose() * deltal;
                    deltal = dL_dz.elm_wise_product(layers[l - 1]->backward(activations[l - 1]));
                } 
                else {
                    throw std::runtime_error("Layer is not of type LinearLayer");
                }
            }
        }
        
        for (size_t i = 0; i < grads.first.size(); ++i) {
            auto p1 = grads.first[i];
            auto p2 = grads.second[i];
            *p1 = *p1 + out.first[i]; //accumulate the gradients
            *p2 = *p2 + out.second[i];
        }
    }

    // load and save are below
    bool Sequential::save(const std::string& filepath) {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) return false;
        
        size_t layer_count = 0;
        for (const auto& layer : layers) {
            if (dynamic_cast<LinearLayer*>(layer.get())) {
                layer_count++;
            }
        }
        file.write(reinterpret_cast<const char*>(&layer_count), sizeof(layer_count));
        
        for (const auto& layer : layers) {
            if (auto linear = dynamic_cast<LinearLayer*>(layer.get())) {
                auto [weights, bias] = linear->get_params();
                
                int input_dim = linear->get_input_dim();
                int output_dim = linear->get_output_dim();
                file.write(reinterpret_cast<const char*>(&input_dim), sizeof(input_dim));
                file.write(reinterpret_cast<const char*>(&output_dim), sizeof(output_dim));
                
                for (size_t i = 0; i < weights->numRows(); i++) {
                    for (size_t j = 0; j < weights->numCols(); j++) {
                        double val = (*weights)(i, j);
                        file.write(reinterpret_cast<const char*>(&val), sizeof(double));
                    }
                }
                
                for (size_t i = 0; i < bias->numRows(); i++) {
                    double val = (*bias)(i, 0);
                    file.write(reinterpret_cast<const char*>(&val), sizeof(double));
                }
            }
        }
        
        file.close();
        return true;
    }
    
    bool Sequential::load(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: File does not exist at path: " << filepath << std::endl;
            return false;
        }
        
        size_t layer_count;
        file.read(reinterpret_cast<char*>(&layer_count), sizeof(layer_count));
        
        size_t linear_count = 0;
        for (const auto& layer : layers) {
            if (dynamic_cast<LinearLayer*>(layer.get())) {
                linear_count++;
            }
        }
        
        if (layer_count != linear_count) {
            std::cerr << "Error: Model architecture mismatch." << std::endl;
            return false;
        }
        
        for (const auto& layer : layers) {
            if (auto linear = dynamic_cast<LinearLayer*>(layer.get())) {
                auto [weights_ptr, bias_ptr] = linear->get_params();
                
                int input_dim, output_dim;
                file.read(reinterpret_cast<char*>(&input_dim), sizeof(input_dim));
                file.read(reinterpret_cast<char*>(&output_dim), sizeof(output_dim));
                
                if (input_dim != linear->get_input_dim() || output_dim != linear->get_output_dim()) {
                    std::cerr << "Error: Layer dimensions mismatch." << std::endl;
                    return false;
                }
                
                for (size_t i = 0; i < weights_ptr->numRows(); i++) {
                    for (size_t j = 0; j < weights_ptr->numCols(); j++) {
                        double val;
                        file.read(reinterpret_cast<char*>(&val), sizeof(double));
                        (*weights_ptr)(i, j) = val;
                    }
                }
                
                for (size_t i = 0; i < bias_ptr->numRows(); i++) {
                    double val;
                    file.read(reinterpret_cast<char*>(&val), sizeof(double));
                    (*bias_ptr)(i, 0) = val;
                }
            }
        }
        
        file.close();
        return true;
    }

};
