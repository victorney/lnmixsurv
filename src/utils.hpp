#ifndef UTILS_HPP
#define UTILS_HPP

#include <RcppArmadillo.h>

double square(const double& x);

arma::mat makeSymmetric(const arma::mat& A);

arma::ivec seq(const int& start, const int& end);

arma::vec repl(const double& x, const int& times);

#endif