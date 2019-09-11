#ML in drug discovery
library(Rcpi)
library(rcdk)
library(keras)
library(ggplot2)

start.time<-Sys.time()
#Function declaration

#Standarize based on previous data
scale.db<-function(df.new, df.old){
  if(ncol(df.new)!=ncol(df.old)) print("Error number of columns!")
  for (i in 1:ncol(df.new)) {
    df.new[,i]<- (df.new[,i]-mean(df.new[,i]))/sd(df.new[,i])
  }
  return(as.matrix(df.new))
}

#Folders
path<-"C:/Users/AutophagyCrusher/Documents/DrugDiscoveryWorkshop-master/SMILES_Activity.csv"
pathDB<-"C:/Users/AutophagyCrusher/Documents/DrugDiscoveryWorkshop-master/ZINCDB/"

#Loading and descriptors calculation
df<-read.csv(path)
mols <- sapply(as.character(df$SMILE), parse.smiles)
desc <- extractDrugAIO(mols,  silent=FALSE)

#Checkpoint
#write.csv(desc, "/home/nacho/StatisticalCodingClub/parameters.csv")
#desc<-read.csv("/home/nacho/StatisticalCodingClub/parameters.csv")

#Automatic cleaning
desc$X<-NULL
findNA<-apply(desc, 2, anyNA)
desc<-desc[,which(findNA==FALSE)]
anyNA(desc)

findIdentical<- apply(desc, 2, sd)
desc<-desc[,which(findIdentical!=0)]

#Manual cleaning
to.remove<-c("ATSc1",
             "ATSc2",
             "ATSc3",
             "ATSc4",
             "ATSc5",
             "BCUTw.1l",
             "BCUTw.1h",
             "BCUTc.1l",
             "BCUTc.1h",
             "BCUTp.1l",
             "BCUTp.1h",
             "khs.tCH",
             "C2SP1")

`%!in%` = Negate(`%in%`)
desc<-desc[,which(names(desc) %!in% to.remove)]

x<-scale(desc)
# normalize <- function(x) {
#   return ((x - min(x)) / (max(x) - min(x)))
# }
# 
# x<-apply(desc,2,normalize)

#y<-scale(df$Activity)
y<-(df$Activity-min(df$Activity))/(max(df$Activity)-min(df$Activity))


#Train/Test
val.ID<-sample(c(1:1452),140 )

x.val<-x[val.ID,]
y.val<-y[val.ID]
x<-x[-val.ID,]
y<-y[-val.ID]


model<-keras_model_sequential()

model %>% layer_dense(units = 80, activation = "relu",  input_shape = (ncol(x))) %>% 
  layer_dense(units = 40, activation = "relu") %>% 
  layer_dense(units = 10, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")


summary(model)  

compile(model, optimizer = "adagrad", loss = "mean_absolute_error")


history<-model %>% fit(x=x,
                       y=y,
                       validation_data=list(x.val, y.val),

                       batch_size=5,
                       epochs=5
                        )
plot(history)

#Prediction over validation

y.hat<-predict(model, x.val)

ggplot()+
  geom_jitter(aes(x=y.val, y=y.hat), colour="red", size=2, alpha=0.5)+
  geom_smooth(aes(x=y.val, y=y.hat))+
  xlab("Activity")+
  ylab("Predicted Activity")+
  theme_minimal()


#Optimization
reset_states(model)
rounds<-200
models<-as.data.frame(matrix(data = NA, ncol =9 ,nrow = rounds))
colnames(models)<-c("UnitsL1", "UnitsL2", "UnitsL3", "act", "L1.norm", "L2.norm", "L1.norm.Rate", "L2.norm.Rate", "MSE")
pb<-txtProgressBar(min=1, max = rounds, initial = 1)

for (i in 1:rounds) {

  #Hyperparams
  unitsL1<-round(runif(1, min=20, max = 100))
  unitsL2<-round(runif(1, min=10, max = 60))
  unitsL3<-round(runif(1, min=2, max = 20))
  actlist<-c("tanh","sigmoid","relu","elu","selu")
  act<-actlist[round(runif(1,min = 1, max=5))]
  L1.norm<-round(runif(1,min = 0, max = 1))
  L2.norm<-round(runif(1,min = 0, max = 1))
  L1.norm.Rate<-runif(1, min = 0.1, max = 0.5)
  L2.norm.Rate<-runif(1, min = 0.1, max = 0.5)
  ###
  
  
  
  setTxtProgressBar(pb, i)
  model.search<-keras_model_sequential()
  
  model.search %>% layer_dense(units = unitsL1, activation = act, input_shape = (ncol(x)))
  if(L1.norm==1)  model.search %>% layer_dropout(rate=L1.norm.Rate)
  model.search %>% layer_dense(units = unitsL2, activation = act)
  if(L2.norm==1)  model.search %>% layer_dropout(rate=L2.norm.Rate)   
  model.search %>% layer_dense(units = unitsL3, activation = act) %>% 
  layer_dense(units = 1, activation = "sigmoid")
  
  compile(model.search, optimizer = "adagrad", loss = "mean_squared_error")
  model.search %>% fit(x=x,
                         y=y,
                         verbose=0,
                         batch_size=10,
                         epochs=10
  )
  
  y.hat<-predict(model.search, x.val)
  MSE<-(sum((y.val-y.hat)^2))/length(y.val)
  
  #FIlling dataframe
  models$MSE[i]<-MSE
  models$act[i]<-act
  models$UnitsL1[i]<-unitsL1
  models$UnitsL2[i]<-unitsL2
  models$UnitsL3[i]<-unitsL3
  models$L1.norm[i]<-L1.norm
  models$L2.norm[i]<-L2.norm
  models$L1.norm.Rate[i]<-L1.norm.Rate
  models$L2.norm.Rate[i]<-L2.norm.Rate
  reset_states(model.search)

  }

#Validation best performer
models<-models[complete.cases(models),]
unitsL1<- models$UnitsL1[models$MSE==min(models$MSE)]
unitsL2<- models$UnitsL2[models$MSE==min(models$MSE)]
unitsL3<- models$UnitsL3[models$MSE==min(models$MSE)]

act<-models$act[models$MSE==min(models$MSE)]
L1.norm<-models$L1.norm[models$MSE==min(models$MSE)]
L2.norm<-models$L2.norm[models$MSE==min(models$MSE)]
L1.norm.Rate<-models$L1.norm.Rate[models$MSE==min(models$MSE)]
L2.norm.Rate<-models$L2.norm.Rate[models$MSE==min(models$MSE)]

model.best<-keras_model_sequential()

model.best %>% layer_dense(units = unitsL1, activation = act, input_shape = (ncol(x)))
if(L1.norm==1)  model.best %>% layer_dropout(rate=L1.norm.Rate)
model.best %>% layer_dense(units = unitsL2, activation = act)
if(L2.norm==1)  model.best %>% layer_dropout(rate=L2.norm.Rate)   
model.best %>% layer_dense(units = unitsL3, activation = act) %>% 
  layer_dense(units = 1, activation = act)

compile(model.best, optimizer = "adam", loss = "mean_squared_error")
model.best %>% fit(x=x,
                     y=y,
                     
                     batch_size=10,
                     epochs=10
)

y.hat<-predict(model.best, x.val)

ggplot()+
  geom_jitter(aes(x=y.val, y=y.hat), colour="red", size=2, alpha=0.5)+
  geom_smooth(aes(x=y.val, y=y.hat))+
  xlab("Activity")+
  ylab("Predicted Activity")+
  
  theme_minimal()

#Screening
db.completed<-vector()
db<-list.files(pathDB)
db.toprocess<-setdiff(db,db.completed)
if(length(db.toprocess>0)) continue=TRUE

while (continue) {

  batchsize<-10000
  batch.vector<-c(1,batchsize)
  
  smi.db<-read.csv(paste(pathDB,db.toprocess[1],sep=""), sep = " ", head=FALSE)
  db.completed[length(db.completed)+1]<-db.toprocess[1]

while(nrow(smi.db)>batch.vector[2]){
  mols.db <- sapply(as.character(smi.db$V1[batch.vector[1]:batch.vector[2]]), parse.smiles)  
  descs.db <- extractDrugAIO(mols.db,  silent=FALSE)
  findNA<-apply(descs.db, 2, anyNA)
  descs.db<-descs.db[,which(findNA==FALSE)]
  descs.db$khs.tCH<-NULL
  desc.db$C2SP1<-NULL
  desc.in.model<-colnames(desc)
  descs.db<-descs.db[,which(names(descs.db) %in% desc.in.model)]
  #Manual cleaning

  
  
  descs.db<-scale.db(descs.db,desc)
  
  predicted.act<-predict(model.best, descs.db)
  
  if(!exists("activity.db")) 
   {activity.db<- predicted.act}
  else{ activity.db<-c(activity.db, predicted.act)}
  
  
  
  batch.vector[1]<-sum(batch.vector)
  batch.vector[2]<-batch.vector[2]+batchsize
  if(batch.vector[2]>nrow(smi.db)) batch.vector[2]<-nrow(smi.db)
  
}

if(!exists("compound.db")) 
{compound.db<- smi.db$V2}
else{ compound.db<-c(compound.db, smi.db$V2)}   
  
db.toprocess<-setdiff(db,db.completed)  
if(length(db.toprocess==0)) continue==FALSE  
  
} 
  

end.time<-Sys.time()

print(end.time-start.time)



