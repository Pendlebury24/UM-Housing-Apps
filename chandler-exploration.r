library(readxl)
library(rpart)
library(rpart.plot)
library(janitor)
library(tidyverse)
library(tidymodels)
library(themis)


data <- read_excel("Data 2019-2023.xlsx") %>% 
  clean_names()

set.seed(20240407)

md <- data %>% 
  filter(m_f != "U", 
         age <= 21)  %>% # let's focus the model.
  mutate(
    no_show = as_factor(no_show),
    academic_year = as_factor(academic_year)
  ) %>% 
  filter(status != "Moved Out", # clean out very low frequency bits
         class != "Graduate",
         academic_year != 2018) %>% 
  select(-id,-name,-room,-rate,-b_date,-completed,
         -cancelled,-reason,-returner_13,-returner_24,
         -admit_term,-month,-canceled,-resident,-admit_year) %>% 
  mutate(
    hall = if_else(is.na(hall),"Missing",hall), # just creating a missing all
    hall = fct_relevel(hall,"Craig Hall")
  )

splits <- initial_split(md, prop = 0.7, strata = no_show)

housing_rec <- recipe(no_show ~ .,
                      data=training(splits)) %>% 
  step_other(admit_status,threshold=0.1) %>% 
  step_dummy(all_nominal(),-all_outcomes()) %>% 
  step_impute_linear(all_predictors()) %>% 
  step_upsample(all_outcomes(),
                over_ratio = 0.25) %>% 
  step_nzv()

#housing_rec_num <- housing_rec %>% 
#  step_mutate(no_show = as.numeric(no_show))

glmnet_mod <- logistic_reg(penalty=0.1,mixture=1) %>% 
  set_engine("glmnet") 
# These should be tuned.

rf_mod <- rand_forest() %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

xgb_mod <- boost_tree() %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")


glmnet_workflow <- workflow() %>% 
  add_recipe(housing_rec) %>% 
  add_model(glmnet_mod) 

rf_workflow <- workflow() %>% 
  add_recipe(housing_rec) %>% 
  add_model(rf_mod)

xgb_workflow <- workflow() %>% 
  add_recipe(housing_rec) %>% 
  add_model(xgb_mod)


train_data <- training(splits)
test_data <- testing(splits)

# Fit the  workflows
glm_fit <- glmnet_workflow %>% 
  fit(data=train_data)

rf_fit <- rf_workflow %>%
  fit(data = train_data)

xgb_fit <- xgb_workflow %>%
  fit(data = train_data)


# Function to evaluate a model
evaluate_model <- function(fit, data,mod_name) {

  metrics <- metric_set(accuracy,kap,f_meas)
  
  predictions <- predict(fit, new_data = data) %>% 
    bind_cols(data) 
  
  results <- metrics(predictions,truth=no_show,estimate=.pred_class) %>% 
    mutate(model = mod_name)
  
  return(results)
}



# Evaluate metrics for glmnet model
glmnet_metrics <- evaluate_model(glm_fit, test_data,"glm")

# Evaluate metrics for random forest model
rf_metrics <- evaluate_model(rf_fit, test_data,"rf")

# Evaluate metrics for XGBoost model
xgb_metrics <- evaluate_model(xgb_fit, test_data,"xgb")

# Combine all metrics for comparison
all_metrics <- bind_rows(glmnet_metrics, rf_metrics, xgb_metrics)

all_metrics %>% 
  select(-.estimator) %>% 
  pivot_wider(names_from=.metric,
              values_from=.estimate) %>% 
  arrange(kap)
# glm looks fishy
