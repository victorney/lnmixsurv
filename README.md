
<!-- README.md is generated from README.Rmd. Please edit that file -->

# persistencia

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![Codecov test
coverage](https://codecov.io/gh/vitorcapdeville/persistencia/branch/master/graph/badge.svg)](https://app.codecov.io/gh/vitorcapdeville/persistencia?branch=master)
<!-- badges: end -->

The goal of lnmixsurv is provide an easy interface for the Bayesian
lognormal mixture model as described in article Lapse risk modelling in insurance:
a Bayesian mixture approach. An usual formula-type model is implemented in `survival_ln_mixture`, with
the usual `suvival::Surv()` interface. The model follows the
[conventions for R modeling
packages](https://tidymodels.github.io/model-implementation-principles/),
and uses the [hardhat](https://hardhat.tidymodels.org/) structure.

The underlying algorithm implementation is via Gibbs sampler techniques and is implemmented in `C++`, using
`RcppArmadillo` for the linear algebra operations and `OpenMP` to
provide a parallelized way of generating multiple chains.

## Installation

You can install the development version of persistencia from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("vivianalobo/lnmixsurv")
```

## parsnip and censored extension

An extension to the models defined by
[parsnip](https://parsnip.tidymodels.org/index.html) and
[censored](https://censored.tidymodels.org/articles/examples.html) is
also provided, adding the `survival_ln_mixture` engine to the
`parsnip::survival_reg()` model.

The following models, engines, and prediction type are
available/extended trhough `persistencia`:

| model        | engine              | time | survival | linear_pred | raw | quantile | hazard |
|:-------------|:--------------------|:-----|:---------|:------------|:----|:---------|:-------|
| survival_reg | survival_ln_mixture | ✖    | ✔        | ✖           | ✖   | ✔        | ✔      |
