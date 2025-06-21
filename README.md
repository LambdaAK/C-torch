# CS-4701-Project

# Project architecture

Three main folders
1. Math - Mathematical foundations
    - Matrix class
    - Matrix operations
        - Addition, subtraction, scalar multiplication, matrix multiplication
    - Eigenvalue algorithms
        - QR algorithm for computing eigenvalues
    - Matrix factorizations
        - SVD
    - Language for mathmatical expressions
        - AST
        - Lexer/Parser
        - Evaluator
    - Numerical differentiation
2. ML
    - Optimization
        - Minimizing loss functions expressed in the language
    - Clustering
        - K-Means
        - DBSCAN
    - Supervised learning
        - K-NN
        - Perceptron/SVM
        - Linear regression
        - Logistic regression
    - Feature engineering
        - PCA
        - Kernels
3. Applications
    - TBD
    
# Applications

These are demonstrations that show how our models work. They involve real-world data and showcase the capabilities of the mathematical and machine learning components we've built.

Each application is self-contained and focuses on a specific use case, providing a clear example of how to apply the library.

# Coding style guide

- For each class, there is a header `.hpp` and implementation `.cpp` file.
- Practice test-driven development!

- Documentation

  - Every single function and class must be documented with the following:
  
    - Description  
      Briefly explain what the function or class does.
  
    - Parameters  
      List each parameter, including:  
      - Name  
      - Type  
      - Description (expected values, units, constraints)
  
    - Return Value  
      Explain what the function returns, including the type and meaning.
  
    - Exceptions (if applicable)  
      Describe any exceptions that can be thrown and under what conditions.
  
    - Examples  
      Provide a minimal, clear example showing how to use the function or class.
  
    - Complexity (optional but recommended)  
      State the time and space complexity.
  
    - Notes (optional)  
      Mention special implementation details, warnings, or anything important to know.
  
  - All documentation must use Doxygen-style comments to support auto-generation of docs.
  
  - Public methods and classes must be fully documented.  
  - Private methods should be documented if their behavior is obvious from their signature.


# Proposal

Our proposal is found in `/proposal`.

# Compiling and running code

There is a `Makefile` in each subdirectory of `src/experiments` for compiling the experiment code.