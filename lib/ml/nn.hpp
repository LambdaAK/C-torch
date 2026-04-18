#pragma once

#include <fstream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "math/matrix.hpp"
#include "nn_optim.hpp"

namespace ml {

  /**
   * @brief Abstract neural-network layer interface.
   *
   * Layers are expected to support a forward pass, a backward pass, and
   * parameter exposure for optimizer integration.
   */
  class Layer {
      public:
        /**
         * @brief Virtual destructor for polymorphic deletion.
         */
        virtual ~Layer() = default;

        /**
         * @brief Runs the forward pass on layer input.
         * @param input Input activation matrix.
         * @return Layer output activation matrix.
         */
        virtual Matrix forward(const Matrix& input) = 0;

        /**
         * @brief Runs backward pass for gradient propagation.
         * @param input Upstream gradient or stored activation context.
         * @return Gradient to pass to previous layer.
         */
        virtual Matrix backward(const Matrix& input) = 0;

        /**
         * @brief Returns pointers to parameter and gradient matrices.
         * @return Pair `(parameter_ptr, gradient_ptr)` for optimizer updates.
         */
        virtual std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> get_params() = 0;
  };

  /**
   * @brief Marker base class for activation-only layers.
   */
  class ActivationLayer : public Layer {
    public:
      /**
       * @brief Activation forward pass.
       * @param input Input activation matrix.
       * @return Activated output matrix.
       */
      virtual Matrix forward(const Matrix& input) = 0;

      /**
       * @brief Activation backward pass.
       * @param input Upstream gradient or cached activation input.
       * @return Gradient propagated to previous layer.
       */
      virtual Matrix backward(const Matrix& input) = 0;

      /**
       * @brief Returns parameter references for this activation layer.
       * @return Pair of matrix pointers; non-parametric layers may return placeholders.
       */
      virtual std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> get_params() = 0;
  };

  /**
   * @brief ReLU activation layer.
   */
  class ReLULayer : public ActivationLayer {
    public:
      /**
       * @brief Applies ReLU activation.
       * @param input Input activation matrix.
       * @return Element-wise `max(0, x)`.
       */
      virtual Matrix forward(const Matrix& input) override;

      /**
       * @brief Applies derivative of ReLU during backpropagation.
       * @param input Upstream gradient or activation context.
       * @return Backpropagated gradient matrix.
       */
      virtual Matrix backward(const Matrix& input) override;

      /**
       * @brief Returns parameter/gradient handles for optimizer.
       * @return Pair of matrix pointers (typically unused for ReLU).
       */
      virtual std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> get_params() override;
    };

  /**
   * @brief Sigmoid activation layer.
   */
  class SigmoidLayer : public ActivationLayer {
    public:
      /**
       * @brief Applies sigmoid activation.
       * @param input Input activation matrix.
       * @return Element-wise sigmoid output.
       */
      virtual Matrix forward(const Matrix& input) override;

      /**
       * @brief Applies derivative of sigmoid during backpropagation.
       * @param input Upstream gradient or activation context.
       * @return Backpropagated gradient matrix.
       */
      virtual Matrix backward(const Matrix& input) override;

      /**
       * @brief Returns parameter/gradient handles for optimizer.
       * @return Pair of matrix pointers (typically unused for sigmoid).
       */
      virtual std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> get_params() override;
  };
  

  /**
   * @brief Hyperbolic tangent activation layer.
   */
  class TanhLayer : public ActivationLayer {
    public:
      /**
       * @brief Applies tanh activation.
       * @param input Input activation matrix.
       * @return Element-wise tanh output.
       */
      virtual Matrix forward(const Matrix& input) override;

      /**
       * @brief Applies derivative of tanh during backpropagation.
       * @param input Upstream gradient or activation context.
       * @return Backpropagated gradient matrix.
       */
      virtual Matrix backward(const Matrix& input) override;

      /**
       * @brief Returns parameter/gradient handles for optimizer.
       * @return Pair of matrix pointers (typically unused for tanh).
       */
      virtual std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> get_params() override;
  };
  
  /**
   * @brief Fully-connected linear layer.
   */
  class LinearLayer : public Layer {
    private:
      int input_dim; ///< Expected input feature dimension.
      int output_dim; ///< Produced output feature dimension.
      std::shared_ptr<Matrix> weights; ///< Trainable weight matrix.
      std::shared_ptr<Matrix> bias; ///< Trainable bias vector.

    public:
      /**
       * @brief Creates a linear transformation layer.
       * @param input_dim Input dimension.
       * @param output_dim Output dimension.
       */
      LinearLayer(int input_dim, int output_dim);

      /**
       * @brief Returns expected input feature dimension.
       * @return Input dimension.
       */
      int get_input_dim() const;

      /**
       * @brief Returns produced output feature dimension.
       * @return Output dimension.
       */
      int get_output_dim() const;

      /**
       * @brief Computes affine forward pass.
       * @param input Input activation matrix.
       * @return `input * weights + bias`.
       */
      virtual Matrix forward(const Matrix& input) override;

      /**
       * @brief Backpropagates gradient through affine transform.
       * @param input Upstream gradient or cached activation context.
       * @return Gradient to previous layer.
       */
      virtual Matrix backward(const Matrix& input) override;

      /**
       * @brief Exposes trainable parameters for optimizer updates.
       * @return Pair containing parameter and gradient pointers.
       */
      virtual std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> get_params() override;
  };

  /**
   * @brief Sequential container for stacking layers.
   *
   * The implementation assumes alternating linear and non-linear layers,
   * ending in a linear layer for output logits.
   */
  class Sequential {

    private:
      std::vector<std::shared_ptr<Layer>> layers; ///< Layer stack in execution order.
      std::vector<Matrix> train_activations; ///< Cached activations for backpropagation.
      std::pair<std::vector<std::shared_ptr<Matrix>>, std::vector<std::shared_ptr<Matrix>>> grads; ///< Gradient buffers for optimizer.
    
    public:
      /**
       * @brief Constructs an empty sequential model.
       */
      Sequential() {}

      /**
       * @brief Appends a layer to the model.
       * @param layer Layer instance to append.
       */
      void add_layer(std::shared_ptr<Layer> layer);

      /**
       * @brief Runs forward pass through all layers.
       * @param x Input batch matrix.
       * @return Final network output.
       */
      Matrix forward(const Matrix& x);
      
      /**
       * @brief Runs full backpropagation from output gradient.
       * @param dL_daL Gradient of loss with respect to final activations.
       */
      void backward(const Matrix& dL_daL);

      /**
       * @brief Returns model parameters and associated gradients.
       * @return Vector of `(parameter_ptr, gradient_ptr)` pairs.
       */
      std::vector<std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>> parameters();

      /**
       * @brief Returns number of exposed trainable parameter matrices.
       * @return Number of parameter tensors (`2 * linear_layer_count()`).
       */
      size_t parameter_count() const;

      /**
       * @brief Copies a parameter tensor by flat parameter index.
       * @param index Zero-based parameter index in `[W0, b0, W1, b1, ...]`.
       * @param out_parameter Destination matrix copy.
       * @return `true` when index is valid and output is assigned.
       */
      bool get_parameter(size_t index, Matrix& out_parameter) const;

      /**
       * @brief Replaces a parameter tensor by flat parameter index.
       * @param index Zero-based parameter index in `[W0, b0, W1, b1, ...]`.
       * @param value New parameter matrix. Shape must match target tensor.
       * @return `true` on success, `false` for out-of-range index or shape mismatch.
       */
      bool set_parameter(size_t index, const Matrix& value);

      /**
       * @brief Counts linear layers in execution order.
       * @return Number of `LinearLayer` instances in the stack.
       */
      size_t linear_layer_count() const;

      /**
       * @brief Returns dimensions of a linear layer by linear-only index.
       * @param linear_index Zero-based index among linear layers.
       * @param out_input_dim Output input dimension.
       * @param out_output_dim Output output dimension.
       * @return `true` when index is valid and outputs are assigned.
       */
      bool linear_layer_dims(size_t linear_index, int& out_input_dim, int& out_output_dim) const;

      /**
       * @brief Copies parameters from another model if layer topology matches.
       * @param other Source model to copy parameters from.
       * @return `true` on success, `false` if model shapes are incompatible.
       */
      bool copy_parameters_from(const Sequential& other);

      /**
       * @brief Serializes model parameters to disk.
       * @param filepath Destination file path.
       * @return `true` when save succeeds.
       */
      bool save(const std::string& filepath);

      /**
       * @brief Loads model parameters from disk.
       * @param filepath Source file path.
       * @return `true` when load succeeds.
       */
      bool load(const std::string& filepath);
  };

} // namespace ml
