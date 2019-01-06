##########################################################################Create test and validation sets  NOTE: Updated 12/24/2018.
#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
     semi_join(edx, by = "movieId") %>%
     semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Learners will develop their algorithms on the edx set
# For grading, learners will run algorithm on validation set to generate ratings

validation <- validation %>% select(-rating)

# Ratings will go into the CSV submission file below:

write.csv(validation %>% select(userId, movieId) %>% mutate(rating = NA),
          "submission.csv", na = "", row.names=FALSE)
rm(dl, ratings, movies, test_index, temp, movielens, removed)
ls()
#################################################################################Process Data
library(anytime)
library(data.table)
library(scales)
library(doParallel)
cl <- makeCluster(4)
registerDoParallel(cl)


edx$genres<-as.factor(edx$genres)
edx$rating<-as.factor(edx$rating)

#time stamp as date factor
edx$timestamp<-anydate(edx$timestamp)
edx$date<-as.factor(format(edx$timestamp, "%Y-%m"))
edx$date<-as.factor(edx$date)
edx$userId<-NULL
edx$movieId<-as.integer(edx$movieId)
edx$timestamp<-NULL
edx$title<-NULL
str(edx)
#####################break out genre, use first three columns as factors, impute missing 
temp <- as.data.frame(edx$genres, stringsAsFactors=FALSE)
temp2 <- as.data.frame(tstrsplit(temp[,1], '[|]', type.convert=TRUE), stringsAsFactors=FALSE)
colnames(temp2) <- c(1:7)
rm(temp)
temp2[,4:8] <- NULL
temp2 <- as.data.frame(lapply(temp2, factor))

#impute with mode per column
imp<-names(sort(table(temp2[,2]),decreasing=TRUE)[1])
temp2[,2][is.na(temp2[,2])] <- imp 
imp1<-names(sort(table(temp2[,3]),decreasing=TRUE)[1])
temp2[,3][is.na(temp2[,3])] <- imp1
temp2[,1][temp2[,1] == "(no genres listed)"] <- "Action"
#Index <- which(temp2$X1 == "(no genres listed)")
temp2[,1]<-as.factor(temp2[,1])





#cbind to edx, remove genre
edx1<-cbind(edx, temp2)
rm(edx)
rm(temp2)
edx1$genres<-NULL

rm(cl)
rm(imp)
rm(imp1)

aa<-as.data.frame(edx1 %>%
  group_by(rating) %>%
  summarise (n = n()) %>%
  mutate(percent = n / sum(n)))
aa$percent<-percent(aa$percent)
aa


#####################################################################################################Data Visalization
N<-ggplot(edx1) + geom_bar(aes(x = rating))
N + theme(axis.text.x = element_text(angle = 60, hjust = 1)) +theme(plot.title = element_text(hjust = 0.5))+ ggtitle("Count of Ratings for all Catagories") + xlab("Rating Catagory") + ylab("Number of Ratings")

p <- ggplot(edx1, aes(x=edx1$`X1`, y=as.numeric(edx1$rating)/2)) + stat_summary(fun.y="mean", geom="bar")
p + theme(axis.text.x = element_text(angle = 60, hjust = 1)) +theme(plot.title = element_text(hjust = 0.5))+ ggtitle("Average User Rating by Genre-First Catagory") + xlab("Genre") + ylab("Average User Rating")

q <- ggplot(edx1, aes(x=edx1$`X2`, y=as.numeric(edx1$rating)/2)) + stat_summary(fun.y="mean", geom="bar")
q + theme(axis.text.x = element_text(angle = 60, hjust = 1)) +theme(plot.title = element_text(hjust = 0.5))+ ggtitle("Average User Rating by Genre-Second Catagory") + xlab("Genre") + ylab("Average User Rating")

r <- ggplot(edx1, aes(x=edx1$`X3`, y=as.numeric(edx1$rating)/2)) + stat_summary(fun.y="mean", geom="bar")
r + theme(axis.text.x = element_text(angle = 60, hjust = 1)) +theme(plot.title = element_text(hjust = 0.5))+ ggtitle("Average User Rating by Genre-Third Catagory") + xlab("Genre") + ylab("Average User Rating")
rm(p, q, r)

###############################################Partition Test and Train Sets/build Ranger Model
test_index1 <- createDataPartition(y = edx1$rating, times = 1, p = 0.3, list = FALSE)
train <- edx1[-test_index1,]
test <- edx1[test_index1,]

rm(test_index1)
rm(edx1)
str(train)

library(ranger)
mod_ranger <- ranger(rating ~ ., data = train, write.forest = TRUE, verbose = FALSE, importance = "impurity", num.trees = 100, mtry = sqrt(ncol(train)))
mod_ranger
######confusion matrix/accuracy
xx <- test[,-2]
yy<- test$rating
rm(test)
RF_pred<-predict(mod_ranger, xx)
y_hat<-RF_pred$prediction
confusionMatrix(y_hat, yy)
rm(RF_pred)
rm(xx)
rm(yy)
rm(y_hat)


######################################################predict validation set
################################data processing/imputing

#time stamp as date factor
validation$timestamp<-anydate(validation$timestamp)
validation$date<-as.factor(format(validation$timestamp, "%Y-%m"))
validation$date<-as.factor(validation$date)

validation$movieId<-as.integer(validation$movieId)
validation$timestamp<-NULL

validation$genres<-as.factor(validation$genres)
#####################break out genre, use first three columns as factors, impute missing 
temp <- as.data.frame(validation$genres, stringsAsFactors=FALSE)
temp2 <- as.data.frame(tstrsplit(temp[,1], '[|]', type.convert=TRUE), stringsAsFactors=FALSE)
colnames(temp2) <- c(1:7)
rm(temp)
temp2[,4:8] <- NULL
temp2 <- as.data.frame(lapply(temp2, factor))

#impute with mode per column
imp<-names(sort(table(temp2[,2]),decreasing=TRUE)[1])
temp2[,2][is.na(temp2[,2])] <- imp 
imp1<-names(sort(table(temp2[,3]),decreasing=TRUE)[1])
temp2[,3][is.na(temp2[,3])] <- imp1
temp2[,1][temp2[,1] == "(no genres listed)"] <- "Action"
#Index <- which(temp2$X1 == "(no genres listed)")
temp2[,1]<-as.factor(temp2[,1])
summary(temp2)

val1<-cbind(validation, temp2)
rm(val, temp2)
val1$genres<-NULL
val1$title<-NULL
val1$movieId<-as.integer(val1$movieId)
str(val1)

val_pred<-predict(mod_ranger, val1)
y_hat1<-val_pred$prediction



validation$rating<-y_hat1
validation$date<-NULL
validation$timestamp<-NULL
validation$title<-NULL
validation$genres<-NULL
write.csv(validation, file = "submission.csv", row.names=FALSE)












































