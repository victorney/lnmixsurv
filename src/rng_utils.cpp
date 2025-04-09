#include "rng_utils.hpp"

// Function used to set a seed
void setSeed(const long long int& seed, std::mt19937& rng_device) {
  rng_device.seed(seed);
}

// Generates a random observation from Uniform(0, 1) 
double runif_0_1(std::mt19937& rng_device) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng_device);
}

// Generates a random observation from Normal(mu, sd^2)
double rnorm_(const double& mu, const double& sd, std::mt19937& rng_device) {
  std::normal_distribution<double> dist(mu, sd);
  return dist(rng_device);
}

// Generates a random observation from Gamma(alpha, beta), with mean alpha/beta
double rgamma_(const double& alpha, const double& beta, std::mt19937& rng_device) {
  std::gamma_distribution<double> dist(alpha, 1.0 / beta);
  return dist(rng_device);
}

// Sample one value (k-dimensional) from a 
// Dirichlet(alpha_1, alpha_2, ..., alpha_k)
arma::vec rdirichlet(const arma::vec& alpha, std::mt19937& rng_device) {
  int K = alpha.n_elem;
  arma::vec sample(K);
  
  for (int k = 0; k < K; ++k) {
    sample(k) = rgamma_(alpha(k), 1.0, rng_device);
  }
  
  sample /= arma::sum(sample);
  return sample;
}

// Generates a random observation from a MultivariateNormal(mean, covariance)
arma::vec rmvnorm(const arma::vec& mean, const arma::mat& covariance, std::mt19937& rng_device) {
  int numDims = mean.n_elem;
  arma::vec sample(numDims);
  
  arma::mat L = arma::chol(covariance, "lower");
  
  arma::vec Z(numDims);
  
  for (int j = 0; j < numDims; j++) {
    Z(j) = rnorm_(0.0, 1.0, rng_device);
  }
  
  sample = mean + L * Z;
  return sample;
}
