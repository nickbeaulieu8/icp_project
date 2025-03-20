library(RTMB)
library(readr)

# ---------------------
# 1. Load and prepare the data
# ---------------------
data <- read_csv("../data/cps_clean_v2.csv")

# Subset to individuals aged 15+ and remove rows with missing key features
#data <- subset(data, AGE >= 15)
#data <- na.omit(data)

# Convert specified categorical features to factors
cat_vars <- c("RELATE", "SEX", "RACE", "MARST", "VETSTAT", "FTYPE", 
              "FAMKIND", "FAMREL", "BPL", "CITIZEN", "NATIVITY", "HISPAN", 
              "EMPSTAT", "LABFORCE", "OCC", "IND", "CLASSWKR", "WKSTAT", 
              "EDUC", "SCHLCOLL", "DIFFHEAR", "DIFFEYE", "DIFFREM", "DIFFPHYS", 
              "DIFFMOB", "DIFFCARE", "DIFFANY", "OCCLY", "INDLY", "CLASSWLY", 
              "WORKLY", "FULLPART", "PENSION", "WANTJOB", "MIGRATE1", "DISABWRK", "QUITSICK")
for (var in cat_vars) {
  if (var %in% names(data))
    data[[var]] <- as.factor(data[[var]])
}

# Create a model formula and the corresponding design matrix.
formula <- INCPER_DELTA ~ AGE + SEX + RACE + MARST + VETSTAT +
  FAMSIZE + NCHILD + EDUC + EMPSTAT + LABFORCE + YEAR
X <- model.matrix(formula, data)

# Define the response variable (target) from the data.
y <- data$INCPER_DELTA

# ---------------------
# 2. Define the negative log-likelihood function
# ---------------------
# The generalized normal (Subbotin) density is:
#   f(y; μ, σ, k) = k/(2σ Γ(1/k)) * exp{ - (|y–μ|/σ)^k }
# so its log density is:
#   log(k) - log(2σ) - lgamma(1/k) - (|y–μ|/σ)^k
#
# Note: We let RTMB know which variable is the response by using the OBS function.
nll <- function(pars) {
  # Allow reference to parameters by name using RTMB
  pars |> RTMB::getAll()
  
  # Tell RTMB that our response is being observed.
  y_obs <- y |> RTMB::OBS()
  
  # Linear predictor
  mu <- as.vector(X %*% beta)
  
  # Convert log-scale parameters to their natural scale.
  sigma <- exp(log_sigma)
  k     <- exp(log_k)
  
  # Set a small constant to avoid taking log(0)
  epsilon <- 1e-10
  
  # Instead of computing (abs(y_obs-mu)/sigma) directly in the log,
  # we add epsilon to protect against the 0 case.
  ratio <- (abs(y_obs - mu) + epsilon) / sigma
  
  # Compute the log density contributions for each observation using
  # the generalized normal (Subbotin) density
  ll <- sum( log(k) - log(2 * sigma) - lgamma(1/k) - ratio^k )
  
  # Return the negative log likelihood (for minimization)
  return(-ll)
}

# ---------------------
# 3. Define starting parameters and create RTMB AD object
# ---------------------
# Here, beta is a vector whose length equals the number of columns in X.
pars <- list(
  beta      = rep(0, ncol(X)),
  log_sigma = 0,    # starting value: sigma = exp(0) = 1
  log_k     = 0     # starting value: k = exp(0) = 1
)

# RTMB will take our nll function and return an object that provides:
#    - the value of the function (fn)
#    - gradient (gr)
#    - hessian, etc.
obj <- nll |> RTMB::MakeADFun(pars, silent = FALSE)

# ---------------------
# ERROR CHECKING
# ---------------------
# Evaluate the function value at the initial parameters
init_fn <- obj$fn(obj$par)
cat("Initial function value:", init_fn, "\n")

# Evaluate the gradient at the initial parameters
init_grad <- obj$gr(obj$par)
cat("Initial gradient:\n")
print(init_grad)

# Check if any gradient values are NA/NaN
if (any(is.na(init_grad)) || any(is.nan(init_grad))) {
  stop("Your initial gradient contains NA/NaN. Check your nll function and data.")
}
# ---------------------


# ---------------------
# 4. Optimize and report results
# ---------------------
# We use nlminb with the function and gradient from RTMB.
opt <- with(obj, nlminb(par, fn, gr))
print(opt)

# Compute standard errors from the Hessian using RTMB’s sdreport
sdr <- obj |> RTMB::sdreport(opt$par)
print(sdr)

# ---------------------
# 5. Compute residuals and check model fit
# ---------------------
# RTMB can compute one-step-ahead residuals.
residuals <- obj |> oneStepPredict(method = "fullGaussian")
str(oneStepPredict(obj, method = "fullGaussian"))

# Compute one-step predictions (returned as an S4 object)
residuals_obj <- oneStepPredict(obj, method = "fullGaussian")
# Convert the object to a numeric matrix
residuals <- as.matrix(residuals_obj)

qqnorm(residuals[, 1])
abline(0, 1)