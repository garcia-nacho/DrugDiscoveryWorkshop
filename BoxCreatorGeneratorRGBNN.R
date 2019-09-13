#### Computer vision in R

#library(fitdistrplus)
library(jpeg)
library(countcolors)
library(keras)
library(pushoverr)
library(imager)
library(ggplot2)
library(deepviz)
library(magrittr)

#Pushover
set_pushover_app(token = "af9mndpox5jiuhxr9ie48ap4in6drt")
set_pushover_user("uwu72orqb1gyfdnnio4881ej15761h")


#Parameters

width<-500 #width
height<-500 #height
pred.batch <- 300
cell.co<-0.5
cells.to.find<-20
#Bounding box size
sizeX<-16
sizeY<-16

#Linux
class.path<-c("C:/Users/AutophagyCrusher/Desktop/CellCV/Classes RGB/")
image.path<-c("C:/Users/AutophagyCrusher/Desktop/CellCV/PICS-LowRes/")

class.files<-list.files(class.path)
#class.files<-class.files[-1]
class.n<-length(class.files)

imageAugmRot<-FALSE
imageAugmDisp<-FALSE
PxDisp.X<-3
PxDisp.Y<-3
batch_size<-20

#Manually curated
for (i in 1:class.n) {

  df<-read.csv(paste(class.path, class.files[i], sep = ""))
  df$X.1<-NULL
  df$X.2<-NULL
  df$X.3<-NULL
  df$image<-gsub("C.-","",df$Label)
  df$image<-gsub(":.*","",df$image)
  df$class<-i-1
  
  if(!exists("training")){
    training<-df
  } else  training<-rbind(training, df)  
  

}

#training$image<-gsub(".tif",".tif.pgm",training$image)

#Check sizes
training<-subset(training, training$X>9 & training$X>9 & training$X<width-9 &  training$Y<height-9)

pic.name.training<-unique(training$image)
training$imageID<-NA
pic.training<-list()

for (i in 1:length(pic.name.training)) {
  training$imageID[training$image==pic.name.training[i]]<-i
  dummy.pic<-load.image(paste(image.path, pic.name.training[i], sep = ""))

  #Colour correction  
 for (h in 1:3) {
   dummy.pic[,,,h]<-dummy.pic[,,,h]-median(dummy.pic[,,,h])
   dummy.pic[,,,h]<-(dummy.pic[,,,h]-min(dummy.pic[,,,h]))/(max(dummy.pic[,,,h])-min(dummy.pic[,,,h]))
 }
  
  
  
  pic.training[[i]]<-dummy.pic
}
gc()

offsetY<-1
offsetX<-1

if(imageAugmDisp){
  offsetY<-PxDisp.X+1
  offsetX<-PxDisp.Y+1}

borders<-which(training$Y<round(sizeY/2)+offsetY | training$Y>height-offsetY-round(sizeY/2) |training$X<round(sizeX/2)+offsetX | training$X>width-offsetX-round(sizeX/2))

training<-training[-borders,]
training$Y<-height-training$Y

#Select a complete different image for validation
name.df<-as.data.frame(pic.name.training)
name.df$image.Number<-gsub(".*#","#",name.df$pic.name.training)
name.df$image.Number<-gsub(".jpg","",name.df$image.Number)
name.df$count<-NA

for (i in 1:nrow(name.df)) {
  name.df$count[i]<-length(which(training$image==name.df$pic.name.training[i]))
  
}
image.val<-name.df$pic.name.training[name.df$count==min(name.df$count)]

validation.count<-which(training$image %in% image.val)

validation <- training[validation.count,]
training <- training[-validation.count,]


#### Generators
`%!in%` = Negate(`%in%`)
cell_generator <- function(X, Y, batch_size) {
  
  function() {
    #Balancing classes
    class0.gen<-which(X$class==0)
    class1.gen<-which(X$class==1)
    class0.gen<-sample(class0.gen, round(batch_size/2))
    class1.gen<-sample(class1.gen, batch_size-round(batch_size/2))
    data.gen<-X[c(class0.gen,class1.gen),]
    training.gen<-array(data = 0, dim = c(batch_size,sizeY+1,sizeX+1,1))
    
    Y<-Y[c(class0.gen,class1.gen)]
    for (h in 1:batch_size) {
      #Bounding boxes
      center<-vector()
      
      center[1]<-data.gen$X[h]
      center[2]<-data.gen$Y[h]
      
      #Image displacement  
      if(imageAugmDisp==TRUE){
        center[1]<-center[1]+round(runif(1, min = -PxDisp.X, max = PxDisp.X)) #PX diplacement
        center[2]<-center[2]+round(runif(1, min = -PxDisp.X, max = PxDisp.X)) #PX diplacement
      }
      
      dummyimage<-pic.training[[X$imageID[h]]][c(center[2]-(sizeY/2)):(center[2]+(sizeY/2)) ,
                                               c(center[1]-(sizeX/2)):(center[1]+(sizeX/2)) ,1,1 ]
      
      if(imageAugmRot==TRUE){
        rotationLevel<-round(runif(1, min = 0, max = 3)) #Image Rotation
        if (rotationLevel>0)    
          for (k in 1: rotationLevel) dummyimage<- aperm(dummyimage, c(2,1))
      }
      training.gen[h,,,1]<-dummyimage[,]
      
    }
    Y<-to_categorical(Y)
    list(training.gen, training.gen)
    
  }
}

##

cell_generator.non.rnd <- function(X, Y, batch_size) {
  seen<-vector()
  function() {
    disp.row<-which(c(1:nrow(X)) %!in% seen)
    
    
    if(batch_size>length(disp.row)){
      seen<-vector
      disp.row<-which(c(1:nrow(X)) %!in% seen)
    } 
    
    
    current<-sample(disp.row, batch_size)
    data.gen<-X[current,]
    seen<-c(seen,current)
    
    training.gen<-array(data = 0, dim = c(batch_size,sizeY+1,sizeX+1,1))
    
    Y<-Y[current]
    for (h in 1:batch_size) {
      #Bounding boxes
      center<-vector()
      
      center[1]<-data.gen$X[h]
      center[2]<-data.gen$Y[h]
      
      
      
      dummyimage<-pic.training[[X$imageID[h]]][c(center[2]-(sizeY/2)):(center[2]+(sizeY/2)) ,
                                               c(center[1]-(sizeX/2)):(center[1]+(sizeX/2)) ,1,1 ]
      
      training.gen[h,,,1]<-dummyimage[,]
      
    }
    Y<-to_categorical(Y)
    list(training.gen, training.gen)
    
  }
}
  

#Validation

validation.arr<-array(data = 0, dim = c(nrow(validation),sizeY,sizeX,3))

for (i in 1:nrow(validation)) {

    center<-vector()
    center[2]<-validation$X[i]
    center[1]<-validation$Y[i]
    
    
    dummyimage<-pic.training[[validation$imageID[i]]][c(center[2]-7):(center[2]+8) , c(center[1]-7):(center[1]+8) ,1, ]
    
    validation.arr[i,,,]<-dummyimage[,,]
  
}

cells.validation<-which(validation$class==1)

#Remove class 0
#training<-subset(training, training$class==1)

#plotArrayAsImage(img2)

CNN.act<-"relu"
NN.act<-"relu"

Input<-layer_input(shape = c(sizeX,sizeY,3))

CNN<-Input %>% layer_conv_2d(filters=4, kernel_size=c(3,3), activation=CNN.act, padding='same',strides=c(1,1),data_format='channels_last')%>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters=4, kernel_size=c(3,3), activation=CNN.act, padding='same',strides=c(1,1),data_format='channels_last')%>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters=4, kernel_size=c(3,3), activation=CNN.act, padding='same',strides=c(1,1),data_format='channels_last') %>% 
  layer_flatten()

SignalExtractor<- function(arg){
  
  mean.arr<-k_mean(arg, axis = c(2,3), keepdims = TRUE)
  sd.arr<-k_std(arg, axis = c(2,3), keepdims = TRUE)
  out.arr<-k_concatenate(list(mean.arr, sd.arr), axis = -1)
  return(out.arr)
}

Extractor<-Input %>% layer_lambda(SignalExtractor) %>% layer_reshape(target_shape = c(6))

CNN.Extractor<- layer_concatenate(list(CNN,Extractor)) %>% 
  layer_dense(units = 48, activation = NN.act) %>% 
  layer_dense(units = 12, activation = NN.act) %>%
  layer_dense(units = 2, activation = "softmax")

model<-keras_model(Input,CNN.Extractor)
summary(model)

#opt<-optimizer_adam( lr= 0.0001 , decay = 1e-6 )

compile(model,optimizer ="Adadelta", loss = 'mean_squared_error' )

date<-as.character(date())
logs<-gsub(" ","_",date)
logs<-gsub(":",".",logs)
logs<-paste("logs/",logs,sep = "")

steps_per_epoch <-round(nrow(training)/batch_size)
if(imageAugmRot) steps_per_epoch<-steps_per_epoch*3
if(imageAugmDisp) steps_per_epoch<-steps_per_epoch*PxDisp.X*PxDisp.Y


#Array for training

training.arr<-array(data = 0, dim = c(nrow(training),sizeY,sizeX,3))

for (i in 1:nrow(training)) {
  
  center<-vector()
  center[2]<-training$X[i]
  center[1]<-training$Y[i]
  
  dummyimage<-pic.training[[training$imageID[i]]][c(center[2]-7):(center[2]+8) , c(center[1]-7):(center[1]+8) ,1, ]
  
  training.arr[i,,,]<-dummyimage[,,]
  
}


y<-training$class
y<-to_categorical(y)

#Normal fit (no generator function)

date<-as.character(date())
logs<-gsub(" ","_",date)
logs<-gsub(":",".",logs)
logs<-paste("logs/",logs,sep = "")

history<-model %>% fit(training.arr,y,
                       epoch=1,
                       validation_data=list(validation.arr,to_categorical(validation$class)),
                       callbacks = callback_tensorboard(logs),
                       shuffle=TRUE,
                       view_metrics=TRUE
                       )

tensorboard(logs)

predicted<-predict(model, validation.arr)
predicted<-as.data.frame(predicted)
predicted$real<-validation$class
predicted$V2<-round(predicted$V2)
predicted$V1<-NULL

#Density maps



img<-pic.training[[1]]

#Prediction 
Stop=FALSE
while(Stop==FALSE){
  
  pred<-array(data = 0, dim = c(pred.batch,sizeY,sizeX,3))
  
  for (i in 1:pred.batch) {
   P.X<-runif(pred.batch, min=36, max = width-36)
   P.Y<-runif(pred.batch, min=36, max = height-36)
   dummyimage<-img[c(P.X[i]-7):(P.X[i]+8) , c(P.Y[i]-7):(P.Y[i]+8) ,1, ]
   dummyimage<-array(dummyimage, dim=dim(dummyimage))
  
   pred[i,,,]<-dummyimage[,,]
   
   
  }
  
  
  output<-as.data.frame(predict(model, pred))
  output$X<-P.X
  output$Y<-P.Y
  output<-subset(output, output$V2>cell.co)
  
  if(!exists("bboxes")){
    bboxes<-output
  } else  bboxes<-rbind(bboxes, output)  

  
  if(nrow(bboxes)>cells.to.find) 
    {Stop=TRUE}
  

  }

bboxes<-subset(bboxes,bboxes$X+9<width & bboxes$X-9>0 & bboxes$Y-9>0 & bboxes$Y+9<height)
img2<-img

for (i in nrow(bboxes)) {
  img2[bboxes$X[i],bboxes$Y[i],1,1]<-1
  
}
plotArrayAsImage(img2[,,1,])


for (i in 1:nrow(bboxes)) {
  
  img2[(round(bboxes$Y[i])-7):(round(bboxes$Y[i])+8),(round(bboxes$X[i])+30):(round(bboxes$X[i])+35),1,1]<-1 #V.R
  img2[(round(bboxes$Y[i])-7):(round(bboxes$Y[i])+8),(round(bboxes$X[i])-35):(round(bboxes$X[i])-30),1,1]<-1 #VR
  
  img2[(round(bboxes$Y[i])-35):(round(bboxes$Y[i])-30),(round(bboxes$X[i])-35):(round(bboxes$X[i])+35),1,1]<-1
  img2[(round(bboxes$Y[i])+30):(round(bboxes$Y[i])+35),(round(bboxes$X[i])-35):(round(bboxes$X[i])+35),1,1]<-1
  
  }


img2 %>% as.raster() %>% plot()
