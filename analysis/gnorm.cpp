#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA: design matrix (X) and response vector (y)
  DATA_MATRIX(X);
  DATA_VECTOR(y);
  
  // PARAMETERS: fixed effects coefficients, log(sigma) and log(shape parameter)
  PARAMETER_VECTOR(beta);
  PARAMETER(log_sigma);
  PARAMETER(log_k);
  
  // Convert log parameters to the positive scale
  Type sigma = exp(log_sigma);
  Type k = exp(log_k);
  
  // Linear predictor (mu)
  vector<Type> mu = X * beta;
  
  // Negative log likelihood (nll)
  Type nll = 0.0;
  int n = y.size();
  for(int i = 0; i < n; i++){
    Type diff = fabs(y(i) - mu(i));
    nll -= log(k) - log(2 * sigma) - lgamma(1/k) - pow(diff/sigma, k);
  }
  return nll;
}
