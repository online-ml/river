rm(list=objects())

library(reticulate)

library(opera)

source_python("R/utils.py")
source("R/utils.R")

options("scipen"=1)

alpha = 0.1

tab_gamma = c(0,
              0.000005,
              0.00005,
              0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,
              0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
              0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09)

### Simulations

n_rep = 500
n = 300
train_size = 200

aggregation_gamma <- function(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg){
  
  test_size = n - train_size
  
  name = get_name_data_R(n, regression='Friedman', noise='ARMA', params_noise=params_noise, seed=n_rep)
  path = "/Users/mzaffran/Documents/Code/CP/cp-epf/data_cluster"
  data = read_pickle(paste(path,"/",name,".pkl",sep=""))
  
  methods = c()
  for(gamma in tab_gamma){
    methods = c(methods,paste('ACP_',gamma,sep=""))}
  
  experts_low = array(NA, dim=c(length(tab_gamma),n_rep,test_size))
  experts_high = array(NA, dim=c(length(tab_gamma),n_rep,test_size))
  
  for(idm in 1:length(methods)){
    method = methods[idm]
    path = "/Users/mzaffran/Documents/Code/CP/cp-epf/results_cluster"
    names = get_name_results_R(method, n, regression='Friedman', noise='ARMA', 
                               params_noise=params_noise)
    results = read_pickle(paste(path,"/",names[['directory']],"/",
                                names[['method']],".pkl",sep=""))
    experts_low[idm,,] = results$Y_inf
    experts_high[idm,,] = results$Y_sup
    experts_low[idm,,][experts_low[idm,,] == -Inf] = -1000
    experts_high[idm,,][experts_high[idm,,] == Inf] = 1000
  }
  
  experts_low_pred = array(NA, dim=c(n_rep,test_size))
  experts_high_pred = array(NA, dim=c(n_rep,test_size))
  
  for(k in 1:n_rep){
    mlpol_grad_low <- mixture(Y=data$Y[k,(train_size+1):dim(data$Y)[2]], experts=t(experts_low[,k,]), model = agg, loss.gradient = T ,
                              loss.type = list(name="pinball", tau=alpha/2))
    mlpol_grad_high <- mixture(Y=data$Y[k,(train_size+1):dim(data$Y)[2]], experts=t(experts_high[,k,]), model = agg, loss.gradient = T ,
                               loss.type = list(name="pinball", tau=1-alpha/2))
    experts_low_pred[k,] = mlpol_grad_low$prediction
    experts_high_pred[k,] = mlpol_grad_high$prediction
  }
  
  results = list('Y_inf'=experts_low_pred, 'Y_sup'=experts_high_pred)
  
  path = "/Users/mzaffran/Documents/Code/CP/cp-epf/results_cluster"
  names = get_name_results_R(paste('Aggregation_',agg,'_Gradient',sep=""), n, regression='Friedman', noise='ARMA', 
                           params_noise=params_noise)
  write_pickle(results, paste(path,"/",names[['directory']],"/",
                              names[['method']],".pkl",sep=""))
}

agg = "BOA"
var = 10

params_noise = list('ar'=c(1,-0.1), 'ma'=c(1,0.1), 'process_variance'=var)
aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg)
params_noise = list('ar'=c(1,-0.8), 'ma'=c(1,0.8), 'process_variance'=var)
aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg)
params_noise = list('ar'=c(1,-0.9), 'ma'=c(1,0.9), 'process_variance'=var)
aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg)
params_noise = list('ar'=c(1,-0.95), 'ma'=c(1,0.95), 'process_variance'=var)
aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg)
params_noise = list('ar'=c(1,-0.99), 'ma'=c(1,0.99), 'process_variance'=var)
aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg)

params_noise = list('ar'=c(1,-0.1), 'ma'=c(1), 'process_variance'=var)
aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg)
params_noise = list('ar'=c(1,-0.8), 'ma'=c(1), 'process_variance'=var)
aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg)
params_noise = list('ar'=c(1,-0.9), 'ma'=c(1), 'process_variance'=var)
aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg)
params_noise = list('ar'=c(1,-0.95), 'ma'=c(1), 'process_variance'=var)
aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg)
params_noise = list('ar'=c(1,-0.99), 'ma'=c(1), 'process_variance'=var)
aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg)

params_noise = list('ar'=c(1), 'ma'=c(1,0.1), 'process_variance'=var)
aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg)
params_noise = list('ar'=c(1), 'ma'=c(1,0.8), 'process_variance'=var)
aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg)
params_noise = list('ar'=c(1), 'ma'=c(1,0.9), 'process_variance'=var)
aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg)
params_noise = list('ar'=c(1), 'ma'=c(1,0.95), 'process_variance'=var)
aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg)
params_noise = list('ar'=c(1), 'ma'=c(1,0.99), 'process_variance'=var)
aggregation_gamma(params_noise, n, train_size, n_rep, tab_gamma, alpha, agg)


### Prices

aggregation_gamma_real <- function(target, dataset, train_size, tab_gamma, alpha, agg, gradient){

  Y = target
  n = length(Y)
  test_size = n - train_size
  
  methods = c()
  for(gamma in tab_gamma){
    methods = c(methods,paste('ACP_',gamma,sep=""))}
  
  experts_low = array(NA, dim=c(length(tab_gamma),test_size))
  experts_high = array(NA, dim=c(length(tab_gamma),test_size))
  
  for(idm in 1:length(methods)){
    method = methods[idm]
    path = "/Users/mzaffran/Documents/Code/CP/cp-epf/results"
    names = get_name_results(method, dataset=dataset)
    results = read_pickle(paste(path,"/",names[[1]],"/",names[[2]],".pkl",sep=""))
    experts_low[idm,] = results$Y_inf
    experts_high[idm,] = results$Y_sup
    experts_low[idm,][experts_low[idm,] == -Inf] = -100
    experts_high[idm,][experts_high[idm,] == Inf] = 1000
  }

  mlpol_grad_low <- mixture(Y=Y[(train_size+1):length(Y)], experts=t(experts_low), model = agg, loss.gradient = gradient ,
                            loss.type = list(name="pinball", tau=alpha/2))
  mlpol_grad_high <- mixture(Y=Y[(train_size+1):length(Y)], experts=t(experts_high), model = agg, loss.gradient = gradient ,
                             loss.type = list(name="pinball", tau=1-alpha/2))
  experts_low_pred = mlpol_grad_low$prediction
  experts_high_pred = mlpol_grad_high$prediction
  
  results = list('Y_inf'=experts_low_pred, 'Y_sup'=experts_high_pred)
  
  path = "/Users/mzaffran/Documents/Code/CP/cp-epf/results"
  if(gradient){names = get_name_results(paste('Aggregation_',agg,'_Gradient',sep=""), dataset=dataset)}
    else{names = get_name_results(paste('Aggregation_',agg,sep=""), dataset=dataset)}
  write_pickle(results, paste(path,"/",names[[1]],"/",names[[2]],".pkl",sep=""))
}

dataset = 'Spot_France_ByHour_train_2019-01-01'
data = read.csv("../data_prices/Prices_2016_2019_extract.csv")
target = data$Spot
train_size = 26136

aggregation_gamma_real(target, dataset, train_size, tab_gamma, alpha, "BOA", gradient = TRUE)
aggregation_gamma_real(target, dataset, train_size, tab_gamma, alpha, "MLpol", gradient = TRUE)
aggregation_gamma_real(target, dataset, train_size, tab_gamma, alpha, "EWA", gradient = TRUE)
aggregation_gamma_real(target, dataset, train_size, tab_gamma, alpha, "BOA", gradient = FALSE)
aggregation_gamma_real(target, dataset, train_size, tab_gamma, alpha, "MLpol", gradient = FALSE)
aggregation_gamma_real(target, dataset, train_size, tab_gamma, alpha, "EWA", gradient = FALSE)

data = read.csv("../data_prices/Prices_2016_2019_extract.csv")

for(h in 0:23){
  
  print(h)
  
  dataset = paste('Spot_France_Hour_',h,'_train_2019-01-01',sep="")
  target = data[data$hour==h,'Spot']
  train_size = 1089
  
  aggregation_gamma_real(target, dataset, train_size, tab_gamma, alpha, "BOA", gradient = TRUE)
  aggregation_gamma_real(target, dataset, train_size, tab_gamma, alpha, "MLpol", gradient = TRUE)
  aggregation_gamma_real(target, dataset, train_size, tab_gamma, alpha, "EWA", gradient = TRUE)
  aggregation_gamma_real(target, dataset, train_size, tab_gamma, alpha, "BOA", gradient = FALSE)
  aggregation_gamma_real(target, dataset, train_size, tab_gamma, alpha, "MLpol", gradient = FALSE)
  aggregation_gamma_real(target, dataset, train_size, tab_gamma, alpha, "EWA", gradient = FALSE)
}
