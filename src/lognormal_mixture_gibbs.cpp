// -*- mode: C++; c-indent-level: 2; c-basic-offset: 2; indent-tabs-mode: nil; -*-

#include <RcppArmadillo.h>
#include <RcppParallel.h>

#include "rng_utils.hpp"
#include "utils.hpp"

#include <unistd.h> // aqui por conta do usleep, trocar por std::this_thread::sleep_for
#include <iostream>
#include <cmath>

using namespace Rcpp;

// Importing the RcppParallelLibs Function from RcppParallel Package to NAMESPACE
//' @importFrom RcppParallel RcppParallelLibs
 
// Sample a random object from a given vector
// Note: it just samples numeric objects (because of c++ class definition) and just one object per time.
int numeric_sample(const arma::ivec& groups,
                   const arma::vec& probs, std::mt19937& rng_device) {
  double u = runif_0_1(rng_device);
  double cumulativeProb = 0.0;
  int n = probs.n_elem;
  for (int i = 0; i < n; ++i) {
    cumulativeProb += probs(i);
    
    if (u <= cumulativeProb) {
      return groups(i);
    }
  }
  
  // This point should never be reached and it's here just for compiling issues
  return 0;
}

double S(const double& y, const double& mu, const double& sd) {
  return R::pnorm(y, mu, sd, false, false);
}

// Function used to sample the latent groups for each observation.
void sample_groups(const int& G, const arma::vec& y, const arma::vec& eta, 
                   const arma::vec& sd, arma::ivec& vec_groups,
                   const bool& data_augmentation, const arma::mat& means,
                   const arma::ivec& delta, std::mt19937& rng_device) {
  arma::vec probs(G);
  double denom;
  int n = y.n_elem;
  
  if(data_augmentation) {
    for (int i = 0; i < n; i++) {
      denom = 0.0;
      
      for (int g = 0; g < G; g++) {
        probs(g) = eta(g) * R::dnorm(y(i), arma::as_scalar(means(i, g)), sd(g), false);
        denom += probs(g);
      }
      
      probs = (denom == 0) * (repl(1.0 / G, G)) + (denom != 0) * (probs / denom);
      
      vec_groups(i) = numeric_sample(seq(0, G - 1), probs, rng_device);
    }
  } else {
    for (int i = 0; i < n; i++) {
      denom = 0.0;
      
      if(delta(i) == 1) {
        for (int g = 0; g < G; g++) {
          probs(g) = eta(g) * R::dnorm(y(i), arma::as_scalar(means(i, g)), sd(g), false);
          denom += probs(g);
        }
      } else {
        for (int g = 0; g < G; g++) {
          probs(g) = eta(g) * S(y(i), arma::as_scalar(means(i, g)), sd(g));
          denom += probs(g);
        }
      }
      
      probs = (denom == 0) * (repl(1.0 / G, G)) + (denom != 0) * (probs / denom);
      
      vec_groups(i) = numeric_sample(seq(0, G - 1), probs, rng_device);
    }
  }
}

// Function used to sample random groups for each observation proportional to the eta parameter
arma::ivec sample_groups_start(const int& G, const arma::vec& y, const arma::vec& eta,
                               std::mt19937& rng_device) {
  int n = y.n_elem;
  arma::ivec vec_groups(n);
  
  for (int i = 0; i < n; i++) {
    vec_groups(i) = numeric_sample(seq(0, G - 1), eta, rng_device);
  }
  
  return(vec_groups);
}

// Function used to simulate survival time for censored observations.
arma::vec augment(const int& G, const arma::vec& y, const arma::ivec& groups,
                  const arma::ivec& delta, const arma::vec& sd,
                  std::mt19937& rng_device, const arma::mat& means) {
  arma::vec out = y;
  arma::uvec censored_indexes = arma::find(delta == 0); // finding which observations are censored
  
  double out_i;
  int count;
  double mean;
  
  for (int i : censored_indexes) {
    out_i = y(i);
    count = 0;
    mean = arma::as_scalar(means(i, groups(i)));
    
    // sample out(i) value
    while(out_i <= y(i)) {
      out_i = rnorm_(mean, sd(groups(i)), rng_device);
      
      // break if it seems like it's going to run forever
      if(count >= 10000) {
        out_i = 1.01 * y(i); // increment y(i) by 1%
        break;
      }
      
      count ++;
    }
    
    out(i) = out_i;
  }
  
  return out;
}

// Create a table for each numeric element in the vector groups.
arma::ivec groups_table(const int& G, const arma::ivec& groups) {
  arma::ivec out(G);
  arma::ivec index;
  
  for (int g = 0; g < G; g++) {
    index = groups(arma::find(groups == g));
    out(g) = index.n_rows;
  }
  
  return out;
}

/* Auxiliary functions for EM algorithm */

// Compute weights matrix
arma::mat compute_W(const arma::vec& y, const arma::mat& X, const arma::vec& eta, 
                    const arma::mat& beta, const arma::vec& sigma, 
                    const int& G, const int& n, double& denom, arma::mat& mat_denom, const arma::rowvec& repl_vec) {
  arma::mat out(n, G);
  
  for(int g = 0; g < G; g++) {
    mat_denom.col(g) = eta(g) * arma::normpdf(y,
                  X * beta.row(g).t(),
                  repl(sigma(g), n));
  }
  
  for(int i = 0; i < n; i++) {
    denom = arma::sum(mat_denom.row(i));
    if(denom > 0) {
      out.row(i) = mat_denom.row(i) / denom;
    } else {
      out.row(i) = repl_vec;
    }
  }
  
  return out;
}

// Function used to computed the expected value of a truncated normal distribution
double compute_expected_value_truncnorm(const double& alpha, const double& mean, const double& sigma) {
  double out;
  
  if (R::pnorm(alpha, 0.0, 1.0, true, false) < 1.0) {
    out = mean + sigma *
      (R::dnorm(alpha, 0.0, 1.0, false)/(R::pnorm(alpha, 0.0, 1.0, false, false)));
  } else {
    out = mean + sigma *
      (R::dnorm(alpha, 0.0, 1.0, false)/0.0001);
  }
  
  return out;
}

// Create the latent variable z for censored observations
arma::vec augment_em(const arma::vec& y, const arma::uvec& censored_indexes,
                     const arma::mat& X, const arma::mat& beta,
                     const arma::vec& sigma, const arma::mat& W,
                     const int& G, const arma::mat& mean,
                     const int& n) {
  arma::vec out = y;
  arma::mat alpha_mat(n, G);
  
  for(int g = 0; g < G; g++) {
    alpha_mat.col(g) = (y - mean.col(g))/sigma(g);
  }
  
  for (int i : censored_indexes) {
    out(i) = 0.0;
    
    for (int g = 0; g < G; g++) {
      out(i) += W(i, g) * compute_expected_value_truncnorm(arma::as_scalar(alpha_mat(i, g)), arma::as_scalar(mean(i, g)), sigma(g));
    }
  }
  
  return out;
}

// Function used to sample groups from W. It samples one group by row based on the max weight.
arma::ivec sample_groups_from_W(const arma::mat& W, const int& n) {
  arma::vec out(n);
  
  for(int i = 0; i < n; i++) {
    out(i) = W.row(i).index_max();
  }
  
  return(arma::conv_to<arma::ivec>::from(out));
}

// Sample initial values for the EM parameters
void sample_initial_values_em(arma::vec& eta, arma::vec& phi, arma::mat& beta, arma::vec& sd, const int& G, const int& k, std::mt19937& rng_device) {
  eta = rdirichlet(repl(rgamma_(1.0, 1.0, rng_device), G), rng_device);
  
  for (int g = 0; g < G; g++) {
    phi(g) = rgamma_(0.1, 0.1, rng_device);
    
    for (int c = 0; c < k; c++) {
      beta(g, c) = rnorm_(0.0, 20.0, rng_device);
    }
  }
  
  sd = 1.0 / sqrt(phi);
}

// Update the matrix beta for the group g
void update_beta_g(const arma::vec& colg, const arma::mat& X, const int& g, const arma::vec& z, arma::mat& beta,
                   arma::sp_mat& Wg) {
  Wg = arma::diagmat(colg);
  arma::mat S = X.t() * Wg * X; 
  
  if(arma::det(makeSymmetric(S)) < 1e-10) { // regularization if matrix is poorly conditioned
    S += 1e-8 * arma::eye(S.n_cols, S.n_cols);
  }
  
  beta.row(g) = arma::solve(makeSymmetric(S), X.t() * Wg * z, arma::solve_opts::likely_sympd).t();
}

// Update the parameter phi(g)
void update_phi_g(const double& denom, const arma::uvec& censored_indexes, const arma::mat& X, const arma::vec& colg, const arma::vec& y, const arma::vec& z,
                  const arma::vec& sd, const arma::mat& beta, const arma::vec& var, const int& g, const int& n, arma::vec& phi, std::mt19937& rng_device,
                  double& alpha, double& quant) {
  alpha = 0.0;
  quant = arma::as_scalar(arma::square(z - (X * beta.row(g).t())).t() * colg);
  
  for(int i : censored_indexes) {
    alpha = (y(i) - arma::as_scalar(X.row(i) * beta.row(g).t())) / sd(g);
    
    if(R::pnorm(alpha, 0.0, 1.0, true, false) < 1.0) {
      quant += colg(i) * var(g) * (1.0 + alpha * R::dnorm(alpha, 0.0, 1.0, false)/(R::pnorm(alpha, 0.0, 1.0, false, false)) - square(R::dnorm(alpha, 0.0, 1.0, false)/(R::pnorm(alpha, 0.0, 1.0, false, false))));
    } else {
      quant += colg(i) * var(g) * (1.0 + alpha * R::dnorm(alpha, 0.0, 1.0, false)/0.0001 - square(R::dnorm(alpha, 0.0, 1.0, false)/0.0001));
    }
  }
  
  // to avoid numerical problems
  if (quant == 0.0) {
    phi(g) = rgamma_(0.5, 0.5, rng_device); // resample phi
  } else {
    phi(g) = denom / quant;
  }
  
  // to avoid numerical problems
  if(phi(g) > 1e5 || phi.has_nan()) {
    phi(g) = rgamma_(0.5, 0.5, rng_device); // resample phi
  }
}

// Update the model parameters with EM
void update_em_parameters(const int& n, const int& G, arma::vec& eta, arma::mat& beta, arma::vec& phi, const arma::mat& W, const arma::mat& X, 
                          const arma::vec& y, const arma::vec& z, const arma::uvec& censored_indexes, const arma::vec& sd, std::mt19937& rng_device,
                          double& quant, double& denom, double& alpha, arma::sp_mat& Wg, arma::vec& colg) {
  arma::vec var = arma::square(sd);
  
  for (int g = 0; g < G; g++) {
    colg = W.col(g);
    
    eta(g) = arma::sum(colg) / n; // updating eta(g)
    
    if (arma::any(eta == 0.0)) { // if there's a group with no observations
      eta = rdirichlet(repl(1.0, G), rng_device);
    }
    
    update_beta_g(colg, X, g, z, beta, Wg); // updating beta for the group g
    update_phi_g(arma::sum(colg), censored_indexes, X, colg, y, z, sd, beta, var, g, n, phi, rng_device, alpha, quant);
  }
}

// Compute model's log-likelihood to select the EM initial values
double loglik_em(const arma::vec& eta, const arma::vec& sd, const arma::mat& W, const arma::vec& z, const int& G, const int& N, const arma::mat& mean, const arma::uvec& censored_indexes) {
  double loglik = 0.0;
  
  for(int i = 0; i < N; i++) {
    if(arma::any(censored_indexes == i)) {
      for (int g = 0; g < G; g++) {
        if (eta(g) * R::pnorm((z(i) - mean(i, g))/sd(g), 0.0, 1.0, false, false) == 0.0) {
          loglik += W(i, g) * log(0.00001);
        } else {
          loglik += W(i, g) * log(eta(g) * R::pnorm((z(i) - mean(i, g))/sd(g), 0.0, 1.0, false, false));
        }
      }
    } else {
      for(int g = 0; g < G; g++) {
        if (eta(g) * R::dnorm(z(i), arma::as_scalar(mean(i, g)), sd(g), false) == 0.0) {
          loglik += W(i, g) * log(0.00001);
        } else {
          loglik += W(i, g) * log(eta(g) * R::dnorm(z(i), arma::as_scalar(mean(i, g)), sd(g), false));
        }
      }
    }
  }
  
  return loglik;
}

// EM for the lognormal mixture model.
arma::field<arma::mat> lognormal_mixture_em(const int& Niter, const int& G, const arma::vec& t, const arma::ivec& delta, const arma::mat& X,
                                            const bool& better_initial_values, const int& N_em,
                                            const int& Niter_em, const bool& internal, const bool& show_output, std::mt19937& rng_device) {
  
  int n = X.n_rows;
  int k = X.n_cols;
  double quant, denom, alpha;
  
  // initializing objects used on EM algorithm
  arma::vec y = log(t);
  arma::vec eta(G);
  arma::vec phi(G);
  arma::vec sd(G);
  arma::vec z(n);
  arma::mat W(n, G);
  arma::mat beta(G, k);
  arma::mat mean(n, k);
  arma::mat out(Niter, G * k + (G * 2));
  arma::uvec censored_indexes = arma::find(delta == 0); // finding which observations are censored
  arma::vec colg(n);
  arma::sp_mat Wg;
  arma::field<arma::mat> out_internal_true(6);
  arma::field<arma::mat> out_internal_false(2);
  arma::field<arma::mat> em_params(6);
  arma::field<arma::mat> best_em(6);
  arma::mat mat_denom(n, G);
  arma::rowvec repl_vec = repl(1.0 / G, G).t();
  
  for(int iter = 0; iter < Niter; iter++) {
    if(iter == 0) { // sample starting values
      
      if(better_initial_values) {
        for (int init = 0; init < N_em; init ++) {
          em_params = lognormal_mixture_em(Niter_em, G, t, delta, X, false, 0, 0, true, false, rng_device);
          
          if(init == 0) {
            best_em = em_params;
            if(show_output) {
              Rcout << "Initial LogLik: " << arma::as_scalar(best_em(5)) << "\n";
            }
          } else {
            if(arma::as_scalar(em_params(5)) > arma::as_scalar(best_em(5))) { // comparing logliks
              if(show_output) {
                Rcout << "Previous maximum: " << arma::as_scalar(best_em(5)) << " | New maximum: " << arma::as_scalar(em_params(5))  << "\n";
              }
              best_em = em_params;
            }
          }
        }
        
        eta = best_em(0);
        beta = best_em(1);
        phi = best_em(2);
        W = best_em(3);
        if(show_output) {
          Rcout << "Starting EM with better initial values" << "\n";
        }
      } else {
        sample_initial_values_em(eta, phi, beta, sd, G, k, rng_device);
        W = compute_W(y, X, eta, beta, sd, G, n, denom, mat_denom, repl_vec);
      }
      
    } else {
      mean = X * beta.t();
      sd = 1.0 / sqrt(phi);
      z = augment_em(y, censored_indexes, X, beta, sd, W, G, mean, n);
      W = compute_W(z, X, eta, beta, sd, G, n, denom, mat_denom, repl_vec);
      update_em_parameters(n, G, eta, beta, phi, W, X, y, z, censored_indexes, sd, rng_device, quant, denom, alpha, Wg, colg);
      
      if(show_output) {
        if((iter + 1) % 20 == 0) {
          Rcout << "EM Iter: " << (iter + 1) << " | " << Niter << "\n";
        }
      }
    }
    
    // Fill the out matrix
    arma::rowvec newRow = 
      arma::join_rows(eta.row(0), 
                      beta.row(0),
                      phi.row(0));
    
    for (int g = 1; g < G; g++) {
      newRow = 
        arma::join_rows(newRow, 
                        eta.row(g), 
                        beta.row(g),
                        phi.row(g));
    }
    
    out.row(iter) = newRow;
  }
  
  mean = X * beta.t();
  
  if(internal) {
    out_internal_true(0) = eta;
    out_internal_true(1) = beta;
    out_internal_true(2) = phi;
    out_internal_true(3) = W;
    out_internal_true(4) = augment_em(y, censored_indexes, X, beta, 1.0 / sqrt(phi), W, G, mean, n);
    out_internal_true(5) = loglik_em(eta, 1.0 / sqrt(phi), compute_W(y, X, eta, beta, 1.0 / sqrt(phi), G, n, denom, mat_denom, repl_vec), y, G, n, mean, censored_indexes);
    
    return out_internal_true;
  } else {
    out_internal_false(0) = out;
    out_internal_false(1) = loglik_em(eta, 1.0 / sqrt(phi), compute_W(y, X, eta, beta, 1.0 / sqrt(phi), G, n, denom, mat_denom, repl_vec), y, G, n, mean, censored_indexes);
    
    return out_internal_false;
  }
  
  return out_internal_false; // should never be reached
}

// Setting parameter's values for the first Gibbs iteration
void first_iter_gibbs(const arma::field<arma::mat>& em_params, arma::vec& eta,
                      arma::mat& beta, arma::vec& phi, const int& em_iter,
                      const int& G, const arma::vec& y,
                      arma::vec& sd, arma::ivec& groups, 
                      const arma::mat& X, const arma::ivec& delta,
                      std::mt19937& rng_device) {
  if (em_iter != 0) {
    // we are going to start the values using the last EM iteration
    eta = em_params(0);
    beta = em_params(1);
    phi = em_params(2);
    sd = 1.0 / sqrt(phi);
    groups = sample_groups_from_W(em_params(3), y.n_rows);
  } else {
    // sampling initial values
    eta = rdirichlet(repl(1.0, G), rng_device);
    
    for (int g = 0; g < G; g++) {
      phi(g) = rgamma_(0.5, 0.5, rng_device);
      beta.row(g) = rmvnorm(repl(0.0, X.n_cols),
               arma::diagmat(repl(20.0 * 20.0, X.n_cols)),
               rng_device).t();
    }
    
    sd = 1.0 / sqrt(phi);
    
    groups = sample_groups_start(G, y, eta, rng_device);
  }
}

// Avoiding groups with zero number of observations in it (causes numerical issues)
void avoid_group_with_zero_allocation(arma::ivec& n_groups, arma::ivec& groups, const int& G, const int& N, std::mt19937& rng_device) {
  int idx = 0;
  int m;
  
  for(int g = 0; g < G; g++) {
    if(n_groups(g) == 0) {
      m = 0;
      while(m < 5) {
        idx = numeric_sample(seq(0, N),
                             repl(1.0 / N, N),
                             rng_device);
        
        if(n_groups(groups(idx)) > 5) {
          groups(idx) = g;
          m += 1;
        } 
      }
      
      // recalculating the number of groups
      n_groups = groups_table(G, groups);
    }
  }
}

double update_phi_g_gibbs(const int& n_groups_g, const arma::vec& linearComb, std::mt19937& rng_device) {
  return rgamma_(static_cast<double>(n_groups_g)  / 2.0 + 0.01, (1.0 / 2.0) * arma::as_scalar(linearComb.t() * linearComb) + 0.01, rng_device);
}

arma::rowvec update_beta_g_gibbs(const double& phi_g, const arma::mat& Xg, const arma::mat& Xgt, const arma::vec& yg, std::mt19937& rng_device) {
  arma::rowvec out;
  arma::mat comb = phi_g * Xgt * Xg + arma::diagmat(repl(1.0 / 1000.0, Xg.n_cols));
  arma::mat Sg;
  arma::vec mg;
  
  if(arma::det(comb) != 0) {
    if(arma::det(makeSymmetric(comb)) < 1e-10) { // regularization if matrix is poorly conditioned
      comb += 1e-8 * arma::eye(Xg.n_cols, Xg.n_cols);
    }
    
    Sg = arma::solve(makeSymmetric(comb),
                     arma::eye(Xg.n_cols, Xg.n_cols),
                     arma::solve_opts::likely_sympd);
    mg = phi_g * (Sg * Xgt * yg);
    out = rmvnorm(mg, Sg, rng_device).t();
  }
  
  return out;
}

// update all the Gibbs parameters
void update_gibbs_parameters(const int& G, const arma::mat& X, const arma::vec& y_aug, const arma::ivec& n_groups, const arma::ivec& groups, 
                             arma::vec& eta, arma::mat& beta, arma::vec& phi, std::mt19937& rng_device) {
  
  arma::mat Xg;
  arma::mat Xgt;
  arma::vec yg;
  arma::vec linearComb;
  arma::uvec indexg;
  
  // updating eta
  eta = rdirichlet(arma::conv_to<arma::Col<double>>::from(n_groups) + 150.0, 
                   rng_device);
  
  // For each g, sample new phi[g] and beta[g, _]
  for (int g = 0; g < G; g++) {
    indexg = arma::find(groups == g);
    Xg = X.rows(indexg);
    Xgt = Xg.t();
    yg = y_aug(indexg);
    linearComb = yg - Xg * beta.row(g).t();
    
    // updating phi(g)
    // the priori used was Gamma(0.01, 0.01)
    phi(g) = update_phi_g_gibbs(n_groups(g), linearComb, rng_device);
    
    // updating beta.row(g)
    // the priori used was MNV(vec 0, diag 1000)
    beta.row(g) = update_beta_g_gibbs(phi(g), Xg, Xgt, yg, rng_device);
  }
}

double update_phi_g_gibbs_augF(const double& phi_actual, const arma::vec& linearComb,
                               std::mt19937& rng_device, const arma::ivec& delta,
                               double& proposal_var, double& adapt_rate, const double& t) {
  double psi_actual = log(phi_actual);
  double lambda = log(proposal_var);
  double psi_prop = rnorm_(psi_actual, proposal_var, rng_device);
  double phi_prop = exp(psi_prop);
  double a0 = 0.01;
  double b0 = 0.01;
  double dccp_actual = (a0 - 1) * psi_actual - b0 * phi_actual;
  double dccp_prop = (a0 - 1) * psi_prop - b0 * phi_prop;
  double decision;
  double decision_outcome; // 1 if proposed value is accepted, 0 otherwise
  
  for(int i = 0; i < linearComb.n_elem; i++) {
    dccp_actual += (delta(i) == 1) * ((1.0/2.0) * psi_actual - (phi_actual/2) * square(linearComb(i))) +
      (delta(i) == 0) * log(S(sqrt(phi_actual) * linearComb(i), 0.0, 1.0));
    dccp_prop += (delta(i) == 1) * ((1.0/2.0) * psi_prop - (phi_prop/2) * square(linearComb(i))) +
      (delta(i) == 0) * log(S(sqrt(phi_prop) * linearComb(i), 0.0, 1.0));
  }
  
  double log_alpha = dccp_prop - dccp_actual + psi_prop - psi_actual;
  
  if(log(runif_0_1(rng_device)) < log_alpha) {
    decision = phi_prop;
    decision_outcome = 1.0;
  } else {
    decision = phi_actual;
    decision_outcome = 0.0;
  }
  
  adapt_rate = 1.0 / pow(t + 1.0, 0.55);
  
  proposal_var = exp(lambda + adapt_rate * (decision_outcome - 0.44));
  
  return decision;
}

arma::rowvec update_beta_g_gibbs_augF(const arma::rowvec beta_actual, const double& phi, const arma::mat& X,
                                      const arma::vec& y, std::mt19937& rng_device, const arma::ivec& delta,
                                      double& proposal_var, double& adapt_rate, const double& t,
                                      const arma::vec& linear_actual) {
  
  int p = beta_actual.n_elem;
  arma::mat Sigma0 = arma::diagmat(repl(1.0/1000.0, p));
  arma::rowvec beta_prop = rmvnorm(beta_actual.t(), arma::diagmat(repl(proposal_var, p)), rng_device).t();
  arma::vec linear_prop = y - X * beta_prop.t();
  
  double decision_outcome;
  arma::rowvec decision;
  double lambda = log(proposal_var);
  
  double dccp_actual = -(1.0 / 2.0) * arma::as_scalar(beta_actual * Sigma0 * beta_actual.t());
  double dccp_prop = -(1.0 / 2.0) * arma::as_scalar(beta_prop * Sigma0 * beta_prop.t());
  
  for(int i = 0; i < X.n_rows; i++) {
    dccp_actual += (delta(i) == 1) * ((1.0 / 2.0) * log(phi) - (phi / 2.0) * square(linear_actual(i))) +
      (delta(i) == 0) * log(S(sqrt(phi) * linear_actual(i), 0.0, 1.0));
    dccp_prop += (delta(i) == 1) * ((1.0 / 2.0) * log(phi) - (phi / 2.0) * square(linear_prop(i))) +
      (delta(i) == 0) * log(S(sqrt(phi) * linear_prop(i), 0.0, 1.0));
  }
  
  if(log(runif_0_1(rng_device)) < dccp_prop - dccp_actual) {
    decision = beta_prop;
    decision_outcome = 1.0;
  } else {
    decision = beta_actual;
    decision_outcome = 0.0;
  }
  
  adapt_rate = 1.0 / pow(t + 1.0, 0.55);
  
  proposal_var = exp(lambda + adapt_rate * (decision_outcome - 0.44));
  
  return decision;
}

void update_gibbs_parameters_augF(const int& G, const arma::mat& X, const arma::vec& y, const arma::ivec& n_groups, const arma::ivec& groups, 
                                  arma::vec& eta, arma::mat& beta, arma::vec& phi, std::mt19937& rng_device, const arma::ivec& delta,
                                  arma::vec& proposal_var_phi, arma::vec& adapt_rate_phi, arma::vec& proposal_var_beta, arma::vec& adapt_rate_beta,
                                  const double& t) {
  
  arma::mat Xg;
  arma::vec yg;
  arma::vec linearComb;
  arma::uvec indexg;
  arma::ivec deltag;
  
  // updating eta
  eta = rdirichlet(arma::conv_to<arma::Col<double>>::from(n_groups) + 1.5, 
                   rng_device);
  
  // For each g, sample new phi[g] and beta[g, _]
  for (int g = 0; g < G; g++) {
    indexg = arma::find(groups == g);
    Xg = X.rows(indexg);
    yg = y(indexg);
    deltag = delta(indexg);
    linearComb = yg - Xg * beta.row(g).t();
    
    // updating phi(g)
    // the priori used was Gamma(0.01, 0.01)
    phi(g) = update_phi_g_gibbs_augF(phi(g), linearComb, rng_device, deltag, proposal_var_phi(g), adapt_rate_phi(g), t);
    
    // updating beta.row(g)
    // the priori used was MNV(vec 0, diag 1000)
    beta.row(g) = update_beta_g_gibbs_augF(beta.row(g), phi(g), Xg, yg, rng_device, deltag, proposal_var_beta(g), adapt_rate_beta(g), t, linearComb);
  }
}

// Internal implementation of the lognormal mixture model via Gibbs sampler
arma::mat lognormal_mixture_gibbs_implementation(const int& Niter, const int& em_iter, const int& G, 
                                                 const arma::vec& t, const arma::ivec& delta, 
                                                 const arma::mat& X,
                                                 long long int starting_seed,
                                                 const bool& show_output, const int& chain_num,
                                                 const bool& better_initial_values, const int& Niter_em,
                                                 const int& N_em, const bool& data_augmentation) {
  
  std::mt19937 global_rng;
  
  // setting global seed to start the sampler
  setSeed(starting_seed, global_rng);
  
  // Calculating number of columns of the output matrix:
  // Each group has p (#cols X) covariates, 1 mixture component and
  // 1 precision. This implies:
  int p = X.n_cols;
  int nColsOutput = (p + 2) * G;
  int N = X.n_rows;
  
  arma::vec y = log(t);
  
  // The output matrix should have Niter rows (1 row for each iteration) and
  // nColsOutput columns (1 column for each element).
  arma::mat out(Niter, nColsOutput);
  
  // The order of filling the output matrix matters a lot, since we can
  // make label switching accidentally. Latter this is going to be defined
  // so we can always fill the matrix in the correct order (by columns, always).
  arma::mat Xt = X.t();
  arma::vec y_aug(N);
  arma::ivec n_groups(G);
  arma::mat means(N, G);
  arma::vec sd(G);
  
  // Starting other new values for MCMC algorithms
  arma::vec eta(G);
  arma::vec phi(G);
  arma::mat beta(G, p);
  arma::ivec groups(N);
  arma::vec log_eta_new(G);
  
  arma::rowvec newRow;
  arma::field<arma::mat> em_params(6);
  
  arma::vec proposal_var_phi(G, arma::fill::value(1.0));
  arma::vec adapt_rate_phi(G, arma::fill::value(1.0));
  
  arma::vec proposal_var_beta(G, arma::fill::value(1.0));
  arma::vec adapt_rate_beta(G, arma::fill::value(1.0));
  
  int step = static_cast<int>(std::ceil(static_cast<double>(Niter) / 10.0));

  if(em_iter > 0) {
    // starting EM algorithm to find values close to the MLE
    em_params = lognormal_mixture_em(em_iter, G, t, delta, X, better_initial_values, N_em, Niter_em, true, false, global_rng);
  } else if(show_output) {
    Rcout << "Skipping EM Algorithm" << "\n";
  }
  
  for (int iter = 0; iter < Niter; iter++) {
    // Starting empty objects for Gibbs Sampler
    if (iter == 0) {
      first_iter_gibbs(em_params, eta, beta, phi, em_iter, G, y, sd, groups, X, delta, global_rng);
    }
    
    means = X * beta.t();
    sd = 1.0 / sqrt(phi);
    
    // Data augmentation (if desired)
    if (data_augmentation) {
      y_aug = augment(G, y, groups, delta, sd, global_rng, means);
    } else {
      y_aug = y;
    }
    
    // Updating Groups
    sample_groups(G, y_aug, eta, sd, groups, data_augmentation, means, delta, global_rng);
    
    // Computing number of observations allocated at each class
    n_groups = groups_table(G, groups);
    
    // Ensuring that every class have, at least, 5 observations
    avoid_group_with_zero_allocation(n_groups, groups, G, N, global_rng);
    
    // Updating all parameters
    if(data_augmentation) {
      update_gibbs_parameters(G, X, y_aug, n_groups, groups, eta, beta, phi, global_rng);
    } else {
      double t = static_cast<double>(iter);
      update_gibbs_parameters_augF(G, X, y, n_groups, groups, eta, beta, phi, global_rng, delta, proposal_var_phi, adapt_rate_phi, proposal_var_beta, adapt_rate_beta, t);
    }
    
    // filling the ith iteration row of the output matrix
    // the order of filling will always be the following:
    
    // First Mixture: proportion, betas, phi
    // Second Mixture: proportion, betas, phi
    // ...
    // Last Mixture: proportion, betas, phi
    
    // arma::uvec sorteta = arma::sort_index(eta, "descend");
    // beta = beta.rows(sorteta);
    // phi = phi.rows(sorteta);
    // eta = eta.rows(sorteta);
    
    newRow = arma::join_rows(beta.row(0),
                             phi.row(0),
                             eta.row(0));
    for (int g = 1; g < G; g++) {
      newRow = arma::join_rows(newRow, beta.row(g),
                               phi.row(g),
                               eta.row(g));
    }
    
    out.row(iter) = newRow;
    
    if(((iter + 1) % step == 0) && show_output) {
      Rcout << "(Chain " << chain_num << ") MCMC Iter: " << iter + 1 << "/" << Niter << "\n";
    }
  }
  
  if(show_output) {
    Rcout << "Chain " << chain_num << " finished sampling." << "\n";
  }
  
  return out;
}

struct GibbsWorker : public RcppParallel::Worker {
  const arma::vec& seeds; // starting seeds for each chain
  arma::cube& out; // store matrix iterations for each chain
  
  // other parameters used to fit the model
  const int& Niter;
  const int& em_iter;
  const int& G;
  const arma::vec& t;
  const arma::ivec& delta;
  const arma::mat& X;
  const bool& show_output;
  const bool& better_initial_values;
  const int& N_em;
  const int& Niter_em;
  const bool& data_augmentation;
  
  // Creating Worker
  GibbsWorker(const arma::vec& seeds, arma::cube& out, const int& Niter, const int& em_iter, const int& G, const arma::vec& t,
              const arma::ivec& delta, const arma::mat& X, const bool& show_output, const bool& better_initial_values,
              const int& N_em, const int& Niter_em, const bool& data_augmentation) :
    seeds(seeds), out(out), Niter(Niter), em_iter(em_iter), G(G), t(t), delta(delta), X(X), show_output(show_output), better_initial_values(better_initial_values), N_em(N_em), Niter_em(Niter_em), data_augmentation(data_augmentation) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      usleep(5000 * i); // avoid racing conditions
      out.slice(i) = lognormal_mixture_gibbs_implementation(Niter, em_iter, G, t, delta, X, seeds(i), show_output, i + 1, better_initial_values, Niter_em, N_em, data_augmentation);
    }
  }
};

// Function to call lognormal_mixture_gibbs_implementation with parallellization
// [[Rcpp::export]]
arma::cube lognormal_mixture_gibbs(const int& Niter, const int& em_iter, const int& G,
                                   const arma::vec& t, const arma::ivec& delta, 
                                   const arma::mat& X, const arma::vec& starting_seed,
                                   const bool& show_output, const int& n_chains,
                                   const bool& better_initial_values, const int& N_em, const int& Niter_em,
                                   const bool& data_augmentation) {
  arma::cube out(Niter, (X.n_cols + 2) * G, n_chains); // initializing output object
  
  // Fitting in parallel
  GibbsWorker worker(starting_seed, out, Niter, em_iter, G, t, delta, X, show_output, better_initial_values, N_em, Niter_em, data_augmentation);
  RcppParallel::parallelFor(0, n_chains, worker);
  
  return out;
}

//[[Rcpp::export]]
arma::field<arma::mat> lognormal_mixture_em_implementation(const int& Niter, const int& G, const arma::vec& t,
                                                           const arma::ivec& delta, const arma::mat& X, 
                                                           long long int starting_seed,
                                                           const bool& better_initial_values, const int& N_em,
                                                           const int& Niter_em, const bool& show_output) {
  
  std::mt19937 global_rng;
  
  // setting global seed to start the sampler
  setSeed(starting_seed, global_rng);
  
  arma::field<arma::mat> out = lognormal_mixture_em(Niter, G, t, delta, X, better_initial_values, N_em, Niter_em, false, show_output, global_rng);
  
  return out;
}
