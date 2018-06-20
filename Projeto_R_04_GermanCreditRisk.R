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

# Fatorizando features obtidas como integers
Credit[, 'NumberDependents'] <- factor(Credit[, 'NumberDependents'])
Credit[, 'CreditStatus'] <- factor(as.character(Credit[, 'CreditStatus']), labels = c('Good', 'Bad'))