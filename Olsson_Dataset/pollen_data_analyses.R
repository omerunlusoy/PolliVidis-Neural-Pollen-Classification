library(vegan)
source('poll_funcs.R')

# Splitting experiment ----

SPD <- read.csv('split_data.csv')      # Data from simple splitting experiment
SPD$spec <- factor(SPD$spec)
SPD <- get.pcorr(SPD,'spec')           # Calculate true positive probability

# Overall accuracy in splitting experiment
print(Accuracy.Split <- mean(SPD$pcorr))

# recall rates of 83 species in splitting experiment
spd <- aggregate(pcorr~spec,SPD,mean)
print(spd <- spd[order(spd$pcorr),])

print(mean(spd$pcorr))   # This is class-averaged accuracy, but it is equivalent to overall (above), as data is balanced



# Leave-one-out experiment - Species level ----

L1oS <- read.csv('L1oS_data.csv')             # Data from leave-one-out experiment
L1oS$spec <- factor(L1oS$spec)
L1oS <- get.pcorr(L1oS,'spec')                # Calculate true positive probability

pn <- read.csv('pollen_counts.csv')         # counts of reference samples and pollen grains for all species

px <- aggregate(pcorr~round+spec,L1oS,mean)  # aggregate per species and round
ppx <- aggregate(pcorr~spec,px,mean)        # aggregate to species
ppx <- merge(ppx,pn,by='spec')              # merge with samples sizes
ppx$ntrain <- ppx$n_samples-1               # no. training images is 1 less than number of samples. The remaining is used for validation
ppx$fn <- ppx$ntrain
ppx$fn[ppx$fn>5] <- 5
ppx$fn <- as.character(ppx$fn)
ppx$fn[ppx$fn=='5'] <- '5+'
ppx$fn <- factor(ppx$fn)                    # fn is a factor with levels 1, 2, 3, 4, 5+ training samples
ppx$qcorr <- qlogis(ppx$pcorr)              # logit transform of recall rates (ratio in 0-1) to fit linear model

print(Overall.Accuarcy.L1oS <- mean(L1oS$pcorr))
print(ClassAvg.Accuracy.L1oS <- mean(ppx$pcorr))

anova(lm(qcorr~fn,ppx))  # Relation between recall and number of training samples


# Leave-one-out experiment - pollen type level ----

L1oT <- read.csv('L1oT_data.csv')             # Data from leave-one-out experiment
L1oT$ptype <- factor(L1oT$ptype)
L1oT <- get.pcorr(L1oT,'ptype')                # Calculate true positive probability

pnt <- aggregate(cbind(n_samples,n_grains)~ptype,pn,sum)       # summing no. samples per pollen type
pxt <- aggregate(pcorr~round+ptype,L1oT,mean)       # mean recall per type and round
ppxt <- aggregate(pcorr~ptype,pxt,mean)             # mean recall per type
ppxt <- merge(ppxt,pnt,by='ptype')
ppxt <- ppxt[order(ppxt$pcorr),]
ppxt$ntrain <- ppxt$n_samples-1               # no. training images is 1 less than number of samples. The remaining is used for validation
ppxt$qcorr <- qlogis(ppxt$pcorr)              # logit transform of recall rates (ratio in 0-1) to fit linear model

print(Overall.Accuarcy.L1oT <- mean(L1oT$pcorr))
print(ClassAvg.Accuracy.L1oT <- mean(ppxt$pcorr))

anova(lm(qcorr~ntrain,ppxt))              # Test of effect of number of training images on recall rate


# Cross-validating using bumblebee samples ----

BCV <- read.csv('BCV_data.csv')
BCV$ptype <- factor(BCV$ptype)
BCV <- get.pcorr(BCV,'ptype')

bcv <- aggregate(pcorr~ptype,BCV,mean)

print(Overall.Accuarcy.BCV <- mean(BCV$pcorr))
print(ClassAvg.Accuracy.BCV <- mean(bcv$pcorr))

BCV$n <- 1
bcvn <- aggregate(n~ptype,BCV,sum)     # n is number of pollen grains of each species found in the samples
bcv <- merge(bcv,bcvn)

print(Deviance.BCV <- -2*sum(log(BCV$pcorr)))   # Deviance of the model

pbcv <- sampleProps(BCV,'imno')                 # relative frequencies of each class in sample


fA <- freqAdjp(BCV,pbcv,'ptype','imno')         # Adjustment by frequency ----
BCV.adj <- fA$Y
BCV.adj <- get.pcorr(BCV.adj,'ptype')
pbcv.adj <- sampleProps(BCV.adj,'imno')         # adjusted relative frequencies

bcv.adj <- aggregate(pcorr~ptype,BCV.adj,mean)
names(bcv.adj)[2] <- 'pcorr.adj'
bcv <- merge(bcv,bcv.adj)

print(Overall.Accuarcy.BCV.adj <- mean(BCV.adj$pcorr))

with(bcv,t.test(x=pcorr,y=pcorr.adj,paired=T))          # t-test of improvement by adjustment, per species

print(Deviance.BCV.adj <- -2*sum(log(BCV.adj$pcorr)))   # Deviance of the model
print(Deviance.BCV.adj-Deviance.BCV)                    # Difference from base model (lower is better)
print(k.opt <- fA$kopt)                                 # best k to minimize Deviance

BCV$H <- renyi(BCV[,grep('P_',names(BCV))],scales=1,hill=F)              # entropy of identification for each pollen grain
BCV.adj$H <- renyi(BCV.adj[,grep('P_',names(BCV.adj))],scales=1,hill=F)  # entropy of identification for each pollen grain
(mean(BCV$H))                # mean entropy of all pollen
(mean(BCV.adj$H))            # in adjusted data

t.test(x=pbcv$pcorr,y=pbcv.adj$pcorr,paired=T)        # t-test of improvement by adjustment, per sample

m <- read.csv('manual_counts.csv')
Sm <- m[,c('imno','S')]
names(Sm)[2] <- 'S.m'            # Shannon diversity of manual counts
So <- pbcv[,c('imno','S')]
names(So)[2] <- 'S.o'            # Shannon diversity of orginal CNN classification
Sa <- pbcv.adj[,c('imno','S')]
names(Sa)[2] <- 'S.a'            # Shannon diversity of adjusted CNN classification
Sm <- merge(Sm,So,by='imno')
Sm <- merge(Sm,Sa,by='imno')

summary(lm(S.o~S.m,Sm))
summary(lm(S.a~S.m,Sm))


# model with 35 pollen types  ----

BCV35 <- read.csv('BCV35_data.csv')
nr <- nrow(BCV35)  # The following lines are a crude way to coerce all pollen types of the predicted classes onto the ptype-factor
btemp <- BCV35[1:35,]
btemp[,c(1,4:35)] <- 0
btemp[,2:3] <- unique(BCV35$predPtype)
BCV35 <- rbind(BCV35,btemp)
BCV35$ptype <- factor(BCV35$ptype)
BCV35 <- BCV35[1:nr,]
BCV35 <- get.pcorr(BCV35,'ptype')

print(Overall.Accuarcy.BCV35 <- mean(BCV35$pcorr))
print(Deviance.BCV35 <- -2*sum(log(BCV35$pcorr)))   # Deviance of the model
print(Deviance.BCV35-Deviance.BCV)                 # Difference from base model (lower is better)

ts <- table(BCV35$ptype)       # table of actual pollen types
tp <- table(BCV35$predPtype)   # table of predicted pollen types
(tf <- tp[which(ts==0)])      # table of predicted pollen in "empty classes"
sum(tf)                       # total number of pollen grains ending up in the 6 additional classes


# species -> 29 pollen types   ----
g2 <- read.csv('names_tab.csv')

BCVs <- read.csv('BCV_84spec_data.csv')                # Bumblebee crossvalidation model for all species
BCVpt <- BCVs[,1:(length(unique(g2$pt.names))+3)]
names(BCVpt)[4:ncol(BCVpt)] <- unique(g2$pt.names)

BCVpt[,4:ncol(BCVpt)] <- 0
for (i in 1:nrow(g2)){
  BCVpt[,g2$pt.names[i]] <- BCVpt[,g2$pt.names[i]] + BCVs[,g2$sp.names[i]]    # Sum the proportions for species included in the groups
}
BCVpt$ptype <- factor(BCVpt$ptype)
BCVpt <- get.pcorr(BCVpt,'ptype')
print(Overall.Accuarcy.BCVpt <- mean(BCVpt$pcorr))
print(Deviance.BCVpt <- -2*sum(log(BCVpt$pcorr)))   # Deviance of the model
print(Deviance.BCVpt-Deviance.BCV)                  # Difference from base model (lower is better)



# model based on GoogLeNet architecture ----
GLN <- read.csv('BCV_data_GLN.csv')
GLN$ptype <- factor(GLN$ptype)
GLN <- get.pcorr(GLN,'ptype')
print(Overall.Accuarcy.GLN <- mean(GLN$pcorr))
print(Deviance.GLN <- -2*sum(log(GLN$pcorr)))   # Deviance of the model
print(Deviance.GLN-Deviance.BCV)                # Difference from base model (lower is better)

# model based on Xception architecture ----
Xcp <- read.csv('BCV_data_Xcp.csv')
Xcp$ptype <- factor(Xcp$ptype)
Xcp <- get.pcorr(Xcp,'ptype')
print(Overall.Accuarcy.Xcp <- mean(Xcp$pcorr))
print(Deviance.Xcp <- -2*sum(log(Xcp$pcorr)))   # Deviance of the model
print(Deviance.Xcp-Deviance.BCV)                # Difference from base model (lower is better)

