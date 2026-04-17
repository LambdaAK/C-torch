#pragma once

#include <random>
#include <fstream>

#include "math/matrix.hpp"

namespace ml {

  class Layer {
      public:
        virtual ~Layer() = default;
        virtual Matrix forward(const Matrix& input) = 0;
        virtual Matrix backward(const Matrix& input) = 0;
        virtual std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> get_params() = 0;
  };

  class ActivationLayer : public Layer {
    public:
      virtual Matrix forward(const Matrix& input) = 0;
      virtual Matrix backward(const Matrix& input) = 0;
      virtual std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> get_params() = 0;
  };

  class ReLULayer : public ActivationLayer {
    public:
      virtual Matrix forward(const Matrix& input) override;
      virtual Matrix backward(const Matrix& input) override;
      virtual std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> get_params() override;
    };

  class SigmoidLayer : public ActivationLayer {
    public:
      virtual Matrix forward(const Matrix& input) override;
      virtual Matrix backward(const Matrix& input) override;
      virtual std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> get_params() override;
  };
  

  class TanhLayer : public ActivationLayer {
    public:
      virtual Matrix forward(const Matrix& input) override;
      virtual Matrix backward(const Matrix& input) override;
      virtual std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> get_params() override;
  };
  
  class LinearLayer : public Layer {
    private:
      int input_dim;
      int output_dim;
      std::shared_ptr<Matrix> weights;
      std::shared_ptr<Matrix> bias;

    public:
      LinearLayer(int input_dim, int output_dim);
      int get_input_dim();
      int get_output_dim();
      virtual Matrix forward(const Matrix& input) override;
      virtual Matrix backward(const Matrix& input) override;
      virtual std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> get_params() override;
  };

  class Sequential {

    private:
      std::vector<std::shared_ptr<Layer>> layers;
      std::vector<Matrix> train_activations; // training activations, used in backward only, alternating between linear and nonlinear
      std::pair<std::vector<std::shared_ptr<Matrix>>, std::vector<std::shared_ptr<Matrix>>> grads; // used for optimizer
    
    public:
      Sequential() {}

      // Assumption: Network is of alternating linear and nonlinear layers, ending with a linear layer
      void add_layer(std::shared_ptr<Layer> layer);

      Matrix forward(const Matrix& x);
      
      void backward(const Matrix& dL_daL);

      // Must have called backward first
      std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> parameters();

      bool copy_parameters_from(const Sequential& other);

      bool save(const std::string& filepath);

      bool load(const std::string& filepath);
  };

  class NN_SGD {

    private:
      float lr;
      size_t batch_size;
      std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> params;

    public:
      NN_SGD() = default;
      NN_SGD(std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> p, float lr, size_t batch_size);
      void zero_grad();
      void step();
  };
};
