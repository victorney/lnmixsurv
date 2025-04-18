% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tidiers.R
\name{tidy.survival_ln_mixture_em}
\alias{tidy.survival_ln_mixture_em}
\title{Tidying method for a Lognormal Mixture model (fitted via Expectation-Maximization algorithm).}
\usage{
\method{tidy}{survival_ln_mixture_em}(x, effects = "fixed", digits = NULL, ...)
}
\arguments{
\item{x}{Fitted model object (survival_ln_mixture_em).}

\item{effects}{A character vector including one or more of \code{"fixed"} and \verb{"auxiliary}.}

\item{digits}{How many significant digits should be displayed?}

\item{...}{Not used.}
}
\value{
A \code{data.frame} without rownames. When \code{effects="fixed"} (the default), tidy.survival_ln_mixutre
returns one row for each coefficient for each component of the mixture with two columns:
\item{term}{The name of the corresponding term in the model.}
\item{estimate}{A point estimate of the coefficient (last iteration value).}

Setting \code{effects="auxiliary"} will select the precision and proportion of mixture components parameters.
}
\description{
These method tidy the estimates from \code{survival_ln_mixture} fits into a short summary. It doesn't contain uncertainty estimates since it's a likelihood maximization algorithm.
}
\examples{

require(survival)
lung$sex <- factor(lung$sex)
set.seed(1)
mod2 <- survival_ln_mixture_em(Surv(time, status == 2) ~ sex, lung)
tidy(mod2)
tidy(mod2, effects = c("fixed", "auxiliary"))

}
