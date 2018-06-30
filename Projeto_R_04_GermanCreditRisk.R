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
set.seed(69)

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
sample <- createDataPartition(scale_Credit$CreditStatus, times = 1, list = F, p = .7)
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

# Visualizando os resultados
rfe_results
varImp((rfe_results), scale = T)
plot(rfe_results, type=c("g", "o"))

# Plot das 8 variaveis mais significantes
# Atuais: CheckingAcctStat, Duration, OtherInstallments, CreditAmount, CreditHistory, SavingsBonds,
# Purpose, Age

# Categoricas
plots_cat<- list()
for (i in c('CheckingAcctStat', 'CreditHistory','OtherInstallments', 'SavingsBonds', 'Purpose')) {
  plots_cat[[i]] <- ggplot(Credit, aes_string(x = i, fill = 'CreditStatus')) + 
  geom_bar(alpha=0.8, colour='black', position = 'dodge') + ggtitle(paste(i, 'x CreditStatus')) +
  theme_minimal()
  print(plots_cat[[i]])
}

# Continua (Duration, CreditAmount, Age)
plots_cont<- list()
for (i in c('Duration', 'CreditAmount','Age')) {
  plots_cont[[i]] <- ggplot(Credit, aes_string(x = 'CreditStatus', y = i, fill = 'CreditStatus')) + 
    geom_boxplot(alpha=0.8, colour='black', position = 'dodge') + ggtitle(paste(i, 'x CreditStatus')) +
    theme_classic()
  print(plots_cont[[i]])
}

## Comparacao entre modelos sem e com pre processamento

# Construindo um modelo de regressao logistica classico
glmModel_full <- train(CreditStatus ~ ., data = train_sample, 
                       family = binomial(),
                       method = 'glm')

# Visualizando o modelo
summary(glmModel_full)

# Testando o modelo nos dados de teste
glmModel_full_pred <-predict(glmModel_full, test_sample)

# Avaliando o modelo
confusionMatrix(glmModel_full_pred, test_sample$CreditStatus)
