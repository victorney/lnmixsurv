#include "rng_utils.hpp"

// Squares a double
double square(const double& x) {
  return x * x;
}

/* AUXILIARY FUNCTIONS */

// Function to make a square matrix symmetric
arma::mat makeSymmetric(const arma::mat& A) {
  return (0.5 * (A + A.t()));
}

// Creates a sequence from start to end with 1 step
arma::ivec seq(const int& start, const int& end) {
  arma::vec out_vec = arma::linspace<arma::vec>(start, end, end - start + 1);
  return arma::conv_to<arma::ivec>::from(out_vec);
}

// Function for replicating a numeric value K times.
arma::vec repl(const double& x, const int& times) {
  return arma::ones<arma::vec>(times) * x;
}
