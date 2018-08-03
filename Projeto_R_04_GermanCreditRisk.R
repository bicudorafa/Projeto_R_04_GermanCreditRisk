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

## Regressao Classica
SimpleModel <- train(CreditStatus ~ ., data = train_sample, 
                     family = binomial(),
                     method = 'glm',
                     trControl = trainControl(method = 'none'))

# Avaliando o modelo
SimpleModel_pred <-predict(SimpleModel, test_sample)
confusionMatrix(SimpleModel_pred, test_sample$CreditStatus)

# Melhores Variaveis
FS_formula = as.formula(CreditStatus ~ CheckingAcctStat + Duration + OtherDetorsGuarantors +CreditAmount +
                          CreditHistory + SavingsBonds + Purpose + Employment)

# Controle dos modelos de treino
train_control <- trainControl(method="repeatedcv", 
                              number=10, 
                              repeats=3, 
                              returnResamp = "all",
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary,
                              verboseIter = TRUE,
                              savePredictions = TRUE)

## Modelos com tratamento

# Glm com melhores variaveis
FSModel <- glmModel_fs <- train(FS_formula, 
                                data = train_sample,
                                method = 'glm',
                                metric = "ROC",
                                trControl = train_control)

## Glmnet

# Grid para escolher melhores hiper parametros
tuning_grid_glm <-  expand.grid(alpha =  c(0, 0.25, 0.75, 1), # indica o peso (0 <- 1) entre L1 e L2
                                lambda = seq(0.0001, 1, length = 5)) # indica o tamanho da penalidade

# Modelo
GlmnetModel <- train(FS_formula, 
                     data = train_sample,
                     metric = "ROC", 
                     method = "glmnet",
                     trControl = train_control,
                     tuneGrid = tuning_grid_glm)

## Random Forest

# Grid para escolher melhores hiper parametros (no caso, so mtry - resto e default)
tuning_grid_rf <- expand.grid(.mtry = c(2, 4, 6, 8),.splitrule = "gini", .min.node.size = 1)

# Modelo
RfModel <- train(FS_formula, 
                 data = train_sample,
                 metric = "ROC", 
                 method = "ranger",
                 trControl = train_control,
                 tuneGrid = tuning_grid_rf)

## KNN

# Grid para escolher melhores hiper parametros (no caso, so mtry - resto e default)
tuning_grid_knn <- expand.grid(.k = 1:25)

# Modelo
KnnModel <- train(FS_formula, 
                  data = train_sample,
                  metric = "ROC", 
                  method = "knn",
                  trControl = train_control,
                  tuneGrid = tuning_grid_knn)

# Create model_list
model_list <- list(GLM_FS = FSModel, GLMNET = GlmnetModel, RF = RfModel, KNN = KnnModel)

# Pass model_list to resamples(): resamples
resamples <- resamples(model_list)

# Summarize the results
#summary(resamples)

# Create bwplot
dotplot(resamples, metric = 'ROC')

# Previsao dos melhores
GlmnetModel_pred <-predict(GlmnetModel, test_sample)
FSModel_pred <- predict(FSModel, test_sample)
confusionMatrix(GlmnetModel_pred, test_sample$CreditStatus)
confusionMatrix(FSModel_pred, test_sample$CreditStatus)