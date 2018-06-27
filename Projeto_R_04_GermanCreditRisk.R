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
## Data Munging

# Definicao variavel de interesse 
Credit[, 'CreditStatus'] <- factor(as.character(Credit[, 'CreditStatus']), labels = c('Good', 'Bad'))

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
set.seed(69)
sample <- caTools::sample.split(scale_Credit$CreditStatus, SplitRatio = 0.70)
train_sample <- as.data.frame(subset(Credit, sample == T))
test_sample <- as.data.frame(subset(Credit, sample == F))

## Feature Selection

# Carregando pacotes necessarios
library(caret) 
library(randomForest) 

# Funcao para selecao das variaveis
rfe.feature.selection <- function(num_iters=20, features, target){
  set.seed(69)
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


# Visualizando os resultados
rfe_results
varImp((rfe_results))