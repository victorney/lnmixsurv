#ifndef RNG_UTILS_HPP
#define RNG_UTILS_HPP

#include <RcppArmadillo.h>
#include <random>

void setSeed(const long long int& seed, std::mt19937& rng_device);

double runif_0_1(std::mt19937& rng_device);

double rnorm_(const double& mu, const double& sd, std::mt19937& rng_device);

double rgamma_(const double& alpha, const double& beta, std::mt19937& rng_device);

arma::vec rdirichlet(const arma::vec& alpha, std::mt19937& rng_device);

arma::vec rmvnorm(const arma::vec& mean, const arma::mat& covariance, std::mt19937& rng_device);

#endif
