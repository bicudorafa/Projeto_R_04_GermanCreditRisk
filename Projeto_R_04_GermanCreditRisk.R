## Analise de Credito - German Credit Data

## Obtencao dos dados

# Carrega o dataset antes da transformacao
german_credit_l <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
Credit <- read.table(german_credit_l)

# Nome das variaveis
names(Credit) <- c('CheckingAcctStat', 'Duration', 'CreditHistory', 'Purpose', 'CreditAmount', 
                   'SavingsBonds', 'Employment', 'InstallmentRatePecnt', 'SexAndStatus', 
                   'OtherDetorsGuarantors', 'PresentResidenceTime', 'Property', 'Age', 
                   'OtherInstallments', 'Housing', 'ExistingCreditsAtBank', 'Job', 'NumberDependents', 
                   'Telephone', 'ForeignWorker', 'CreditStatus')

# Analise do dataframe
str(Credit)
summary(Credit)

## Data Cleaning

# Definicao variavel de interesse 
Credit[, 'CreditStatus'] <- factor(Credit[, 'CreditStatus'], labels = c('Good', 'Bad'))

# Funcao para automatizar "fatorizacao" das variaveis
to.factor <- function(df, features) {
  for (feature in features) {
    df[[feature]] <- as.factor(df[[feature]])
  }
  return(df)
}

# Criacao do string vector das variaveis a  serem fatorizadas e sua fatorizacao
categorical_vars <- c('CheckingAcctStat', 'CreditHistory', 'Purpose', 
                     'SavingsBonds', 'Employment', 'InstallmentRatePecnt', 'SexAndStatus', 
                     'OtherDetorsGuarantors', 'PresentResidenceTime', 'Property',  
                     'OtherInstallments', 'Housing', 'ExistingCreditsAtBank', 'Job', 'NumberDependents', 
                     'Telephone', 'ForeignWorker', 'CreditStatus')

Credit <- to.factor(Credit, categorical_vars)

## Preparacao dos dados para modelo preditivo

# Carregando pacote necessario e iniciando seed do projeto
library(caret) 
set.seed(666)

# Funcao para automatizar normalizacao
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center=T, scale=T)
  }
  return(df)
}

# Normalizacao das variaveis
numeric_vars <- c("Duration", "Age", "CreditAmount")
scale_Credit <- scale.features(Credit, numeric_vars)

# Separacao dos Sets de Treino e Teste
sample <- createDataPartition(scale_Credit$CreditStatus, times = 1, list = F, p = .6)
train_sample <- scale_Credit[sample, ]
test_sample <- scale_Credit[-sample, ]

## Feature Selection

# Carregando pacotes necessarios
library(randomForest)
library(ggplot2)

# Funcao para selecao das variaveis
rfe.feature.selection <- function(num_iters=20, features, target){
  variable_sizes <- 1:10
  control <- rfeControl(functions = rfFuncs, method = "cv", 
                        verbose = FALSE, returnResamp = "all", 
                        number = num_iters)
  rfe_results <- rfe(x = features, y = target, 
                     sizes = variable_sizes, 
                     rfeControl = control)
  return(rfe_results)
}

# Executando a funcao e para obter features mais explicativas
rfe_results <- rfe.feature.selection(features = train_sample[,-21], 
                                 target = train_sample[,21])

# Selecao das Features mais significates e visualizacao da significancia
rfe_results
varImp((rfe_results), scale = F)
plot(rfe_results, type=c("g", "o"))
optVariables <- rfe_results[["optVariables"]]
optVariables

# Plot das variaveis otimas

# Categoricas
plots_cat<- list()
for (i in c('CheckingAcctStat', 'CreditHistory','OtherDetorsGuarantors', 'SavingsBonds', 'Purpose', 
            'Employment')) {
  plots_cat[[i]] <- ggplot(Credit, aes_string(x = i, fill = 'CreditStatus')) + 
  geom_bar(alpha=0.8, colour='black', position = 'dodge') + ggtitle(paste(i, 'x CreditStatus')) +
  theme_minimal()
  print(plots_cat[[i]])
}

# Continua (Duration, CreditAmount)
plots_cont<- list()
for (i in c('Duration', 'CreditAmount')) {
  plots_cont[[i]] <- ggplot(Credit, aes_string(x = 'CreditStatus', y = i, fill = 'CreditStatus')) + 
    geom_boxplot(alpha=0.8, colour='black', position = 'dodge') + ggtitle(paste(i, 'x CreditStatus')) +
    theme_classic()
  print(plots_cont[[i]])
}

## Comparacao entre modelos com e sem feature selection

# Controles para certificar testes imparciais

# Particoes da data para reutilizar nos cvs
myFolds <- createFolds(train_sample$CreditStatus, k = 5)

# Controle dos modelos de treino
train_control <- trainControl(method = "cv", 
                              index = myFolds, 
                              returnResamp = "all",
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary,
                              verboseIter = TRUE,
                              savePredictions = TRUE)

## Modelo de classificacao sem feature selection
glmModel_full <- train(CreditStatus ~ ., data = train_sample,
                       method = 'glm',
                       metric = "ROC",
                       trControl = train_control)

# Avaliacao 1
glmModel_full_pred <-predict(glmModel_full, test_sample)
confusionMatrix(glmModel_full_pred, test_sample$CreditStatus)

## Modelo de classificacao com feature selection
glmModel_fs <- train(CreditStatus ~ 
                       CheckingAcctStat +
                       Duration + 
                       OtherDetorsGuarantors +
                       CreditAmount +
                       CreditHistory +    
                       SavingsBonds +
                       Purpose +
                       Employment, 
                       data = train_sample,
                       method = 'glm',
                       metric = "ROC",
                       trControl = train_control)

# Avaliacao 2
glmModel_fs_pred <-predict(glmModel_fs, test_sample)
confusionMatrix(glmModel_fs_pred, test_sample$CreditStatus)

## Comparacao dos 2 modelos

# Lista dos modelos
model_list <- list(All_Features = glmModel_full, Feature_Selection = glmModel_fs)

# Pass model_list to resamples(): resamples
resamples <- resamples(model_list)

# Summarize the results
summary(resamples)

# Create dotplot
dotplot(resamples, metric  = "ROC")

## Plotagem das ROC Curves

# Pacote necesario
library(ROCR)

# Lista com as predicoes
full_pred <- predict(glmModel_full, newdata = test_sample, type = "prob")
fs_pred <- predict(glmModel_fs, newdata = test_sample, type = "prob")
pred_list <- list(full_pred$Good, fs_pred$Good)

# Lista dos valores de fato (mesmo para todos)
m <- length(pred_list)
test_list <- rep(list(test_sample$CreditStatus), m)

# ROC curves
pred <- prediction(pred_list, test_list)
rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves")
legend(x = "bottomright", 
       legend = c("All_Features", "Feature_Selection"),
       fill = 1:m)