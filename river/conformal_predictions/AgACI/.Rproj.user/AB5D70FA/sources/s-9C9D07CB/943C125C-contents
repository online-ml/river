get_name_data_R <- function(n, regression='Friedman', noise='ARMA', params_reg={}, params_noise={}, seed=1){
  
  if(regression == 'Friedman'){name = 'Friedman'}
  else if(regression == 'Linear'){name = 'Linear'}

  if(noise == 'ARMA'){
    ar = params_noise['ar'][[1]]
    ma = params_noise['ma'][[1]]
  
    ar_name = 'AR'
    if(length(ar) >= 2){
      for(p in 2:length(ar)){ar_name = paste(ar_name,as.character(-ar[p]),sep='_')}
    }
  
    ma_name = 'MA'
    if(length(ma) >= 2){
      for(q in 2:length(ma)){ma_name = paste(ma_name,as.character(ma[q]),sep='_')}
    }
    
    name = paste(name,'ARMA',ar_name,ma_name,sep='_')
    
    if(!is.null(params_noise[["scale"]])){
      name = paste(name,'scale',as.character(params_noise['scale']),sep='_')
    }
      
    if(!is.null(params_noise[["process_variance"]])){
      name = paste(name,'fixed_variance',as.character(as.integer(params_noise['process_variance'])),
                   sep='_')
    }
    name = paste(name,'seed',as.character(as.integer(seed)),'n',as.character(as.integer(n)),
            sep='_')
  }
    return(name)
}

get_name_results_R <- function(method, n=NULL, online=TRUE, randomized=FALSE, 
                               params_method={}, basemodel='RF', regression=NULL, 
                               noise=NULL, params_reg={}, params_noise={}, dataset=NULL){
 
  # Results file name, depending on the method
  
  name_method = paste(method,basemodel,sep='_')
  
 # if((method == 'ACP') & (params_method != {})){
#    name_method = paste(name_method,'gamma',as.character(params_method['gamma']),sep='_')
#  }

  if(regression == 'Friedman'){name_directory = 'Friedman'}
  else if(regression == 'Linear'){name_directory = 'Linear'}
  
  if(noise == 'ARMA'){
    ar = params_noise['ar'][[1]]
    ma = params_noise['ma'][[1]]
    
    ar_name = 'AR'
    if(length(ar) >= 2){
      for(p in 2:length(ar)){ar_name = paste(ar_name,as.character(-ar[p]),sep='_')}
    }
    
    ma_name = 'MA'
    if(length(ma) >= 2){
      for(q in 2:length(ma)){ma_name = paste(ma_name,as.character(ma[q]),sep='_')}
    }
    
    name_directory = paste(name_directory,'ARMA',ar_name,ma_name,sep='_')
    
    if(!is.null(params_noise[["scale"]])){
      name_directory = paste(name_directory,'scale',as.character(params_noise['scale']),
                             sep='_')
    }
  }
    if(!is.null(params_noise[["process_variance"]])){
      name_directory = paste(name_directory,'fixed_variance',
                             as.character(as.integer(params_noise['process_variance'])),
                             sep='_')
    }

    if(!is.null(dataset)){name_directory = dataset}
    else{name_directory = paste(name_directory,'n',as.character(as.integer(n)),sep='_')}
    
    names = list("directory" = name_directory, "method" = name_method)
    return(names)
}
  
  