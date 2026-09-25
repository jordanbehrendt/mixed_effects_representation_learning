# ------------------------------------------------------------------
# Linear Mixed Model Script (nlme backend)
# ------------------------------------------------------------------
# This script is called from Python via subprocess.
# It fits a linear mixed model and writes results to CSV files.
#
# Expected arguments (in order):
# 1. input_file        : path to input CSV data
# 2. fixed_form        : fixed-effects formula (e.g., "y ~ x1 + x2")
# 3. group_var         : grouping variable (e.g., subject ID)
# 4. random_form       : random-effects formula (e.g., "1 + time")
# 5. random_covariance  : covariance structure for random effects ("diagonal", "full")
# 6. residual_covariance        : covariance structure ("independence", "ar1", "cs")
# 7. time_var          : time variable (required for AR1)
# 8. estimation_method : "REML" or "ML"
# 9. out_dir           : output directory for results
# ------------------------------------------------------------------

library(nlme)

# ------------------------------------------------------------------
# Parse command line arguments
# ------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 9) {
  stop("Expected 8 arguments but received fewer.")
}

input_file              <- args[1]
fixed_form              <- args[2]
group_var               <- args[3]
random_form             <- args[4]
random_covariance       <- args[5]
residual_covariance     <- tolower(args[6])
time_var                <- args[7]
estimation_method       <- args[8]
out_dir                 <- args[9]

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
df <- read.csv(input_file)

if (!(group_var %in% colnames(df))) {
  stop(paste("Grouping variable not found in data:", group_var))
}

# ------------------------------------------------------------------
# Define model components
# ------------------------------------------------------------------

# Fixed effects
fixed_formula <- as.formula(fixed_form)

# Random effects
if (random_covariance == "diagonal") {

  random_formula <- list(
    pdDiag(
      as.formula(
        paste("~", random_form)
      )
    )
  )

  names(random_formula) <- group_var

} else if (random_covariance == "full") {

  random_formula <- as.formula(
    paste("~", random_form, "|", group_var)
  )

} else {

  stop(
    paste(
      "Unsupported random covariance:",
      random_covariance
    )
  )
}

# ------------------------------------------------------------------
# Correlation structure
# ------------------------------------------------------------------
cor_struct <- NULL

if (residual_covariance == "ar1") {

  # AR(1) requires a time variable
  if (!(time_var %in% colnames(df))) {
    stop(paste("Time variable required for AR1 but not found:", time_var))
  }

  cor_struct <- corAR1(
    form = as.formula(paste("~", time_var, "|", group_var))
  )

} else if (residual_covariance == "cs") {

  cor_struct <- corCompSymm(
    form = as.formula(paste("~ 1 |", group_var))
  )

} else if (residual_covariance != "independence") {

  stop(paste("Unsupported covariance structure:", residual_covariance))
}

# ------------------------------------------------------------------
# Fit model
# ------------------------------------------------------------------
model <- lme(
  fixed = fixed_formula,
  data = df,
  random = random_formula,
  correlation = cor_struct,
  method = estimation_method
)

# ------------------------------------------------------------------
# Extract fixed effects
# ------------------------------------------------------------------
fe <- fixed.effects(model)

fe_df <- data.frame(
  term = names(fe),
  estimate = as.numeric(fe)
)

write.csv(
  fe_df,
  file.path(out_dir, "fixed_effects.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------------
# Extract random effects
# ------------------------------------------------------------------
re <- ranef(model)

re_df <- data.frame(
  group = rownames(re),
  re
)

# rename 'group' → actual group_var
colnames(re_df)[1] <- group_var

# Make column names explicit (e.g., re_intercept, re_time)
colnames(re_df)[2:ncol(re_df)] <- paste0("re_", colnames(re))

write.csv(
  re_df,
  file.path(out_dir, "random_effects.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------------
# Extract model metrics
# ------------------------------------------------------------------
residual_sd <- sigma(model)
residual_variance <- residual_sd^2

metrics_df <- data.frame(
  aic = AIC(model),
  bic = BIC(model),
  log_likelihood = as.numeric(logLik(model)),
  residual_sd = residual_sd,
  residual_variance = residual_variance
)

write.csv(
  metrics_df,
  file.path(out_dir, "metrics.csv"),
  row.names = FALSE
)