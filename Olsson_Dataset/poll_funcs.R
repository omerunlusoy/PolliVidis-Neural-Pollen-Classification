
get.pcorr <- function(X,gvar){ 
  # Likelihood of the true species ----
  levG0 <- levels(X[,gvar])
  levG <- paste0('P_',gsub(' ','.',levG0))
  levG <- gsub('-','.',levG)
  for (i in 1:length(levG0)){
    rws <- X[,gvar]==levG0[i]
    cls <- names(X)==levG[i]
    if (sum(cls)>0){
      X$pcorr[rws] <- X[rws,cls]
    }
  }
  return(X)
}


freqAdjp <- function(X,x,gvar,fvar){
  
  # Adjustment by frequncey - Majority vote ----
  # A reasonably fast way to calculate optimal k and minimize Deviance
  # Uses the summed frequencies (proportions) of all species for each sample, i.e. the rows in x
  # to adjust the probabilities of all individual pollen. The weight of this adjustment, k, is 
  # optimized, by finding the k that minimizes deviance. A proper minimization could be employed 
  # here instead.
  
#  x <- aggregate(X[,grep('P_',names(X),value=T)],list(fvar=X[,fvar]),sum)
#  x$n <- rowSums(x[,2:(ncol(x))])
#  x[,2:(ncol(x)-1)] <- x[,2:(ncol(x)-1)]/x$n
  
  X0 <- X[,grep('P_',names(X))]
  X0 <- cbind(X[,1:3],X0*0)
  for (i in 1:nrow(x)){
    if (i %% 10==0){print(i)}
    X0[X0[,fvar]==x[i,fvar],grep('P_',names(X0))] <- x[i,grep('P_',names(x))]
  }
  
  K <- c(seq(0,.9,.1),seq(.95, 1.75,by=.05),seq(1.8,2.3,by=.1))
  E <- data.frame(matrix(nrow=length(K),ncol=2))
  names(E) <- c('k','Deviance')
  E$k <- K
  j=0
  print('Calculate Deviance...')
  for (k in K){
    j=j+1
    print(c(j,k))
    
    Y <- X
    Y[,grep('P_',names(Y))] <- X0[,grep('P_',names(X0))]^k*Y[,grep('P_',names(Y))]
    Y[,grep('P_',names(Y))] <- Y[,grep('P_',names(Y))]/rowSums(Y[,grep('P_',names(Y))])
    Y <- get.pcorr(Y,gvar)
    Y$loglik <- log(Y$pcorr)
    E$Deviance[j] <- -2*sum(Y$loglik,na.rm=T)
  }

  kopt <- E$k[which.min(E$Deviance)]
  Y <- X
  Y[,grep('P_',names(Y))] <- X0[,grep('P_',names(X0))]^kopt*Y[,grep('P_',names(Y))]
  Y[,grep('P_',names(Y))] <- Y[,grep('P_',names(Y))]/rowSums(Y[,grep('P_',names(Y))])
  
  Y <- get.pcorr(Y,gvar)
#  Y$loglik <- log(Y$pcorr)
  
  dcl <- grep('P_',names(Y))[1]-1
  Y <- get.predSpecp(Y,dcl)
#  Y$H <- renyi(Y[,grep('P_',names(Y))],scales=1,hill=F)
  
  #x <- aggregate(X[,c(grep('P_',names(X),value=T),'pcorr')],list(bildnummer=X[,'bildnummer'],filnamn=X[,'filnamn']),sum)
  #x$n <- rowSums(x[,3:(ncol(x)-1)])
  #x[,3:(ncol(x)-1)] <- x[,3:(ncol(x)-1)]/x$n
  
  #y <- aggregate(Y[,c(grep('P_',names(Y),value=T),'pcorr')],list(bildnummer=Y[,'bildnummer'],filnamn=Y[,'filnamn']),sum)
  #y$n <- rowSums(y[,3:(ncol(y)-1)])  
  #y[,3:(ncol(y)-1)] <- y[,3:(ncol(y)-1)]/y$n
  
  #y$S <- as.numeric(renyi(y[,grep('P_',names(y))],scales=1,hill=T))  # diversity of each sample
  
  # cfY :  Confusion matrix of Y ----
  #cfY <- aggregate(Y[,grep('P_',names(Y))],list(BOSRn=Y[,'BOSRn']),mean,drop=F)
  #cfY[is.na(cfY)] <- 0
  
  
  return(Yout <- list('Y'=Y,'E'=E,'kopt'=kopt))
}


get.predSpecp <- function(X,dcl){
  # Recalculate predicted class ---- 
  # The code assumes that this information is in the variable immediatley left of the first P_-variable.

  X[,dcl] <- names(X)[dcl+apply(X[,grep('P_',names(X))],1,which.max)]
  X[,dcl] <- gsub('P_','',X[,dcl])
  X[,dcl] <- gsub('.group','-group',X[,dcl])
  X[,dcl] <- gsub('.',' ',X[,dcl],fixed=T)
  X[,dcl] <- factor(X[,dcl])
  
  return(X)
}



sampleProps <- function(X,fvar){
  # Calculate relative frequencies of classes in each sample ----
  
  x <- aggregate(X[,c(grep('P_',names(X),value=T),'pcorr')],list(fvar=X[,fvar]),sum)
  x$n <- rowSums(x[,2:(ncol(x)-1)])
  x[,2:(ncol(x)-1)] <- x[,2:(ncol(x)-1)]/x$n
  x$S <- as.numeric(renyi(x[,grep('P_',names(x))],scales=1,hill=T))  # diversity of each sample
  names(x)[grep(fvar,names(X))] <- fvar
  
  return(x)
}

