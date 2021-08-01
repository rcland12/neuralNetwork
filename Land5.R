# Load functions needed
{
  sigmoid <- function(x){1/(1+exp(-x))}
  sigmoidPrime <- function(x){sigmoid(x)*(1-sigmoid(x))}
  WgradHelper <- function(a, d){
    sum <- matrix(0, nrow = ncol(a), ncol = ncol(d))
    for (i in 1:nrow(a)) {
      sum <- sum + outer(a[i,], d[i,])
    }
    sum <- sum/nrow(a)
  }
  standardizeObservation <- function(x0, X){
    (x0 - colMeans(X))/apply(X, 2, sd)
  }
}

# Problem 2 -----------------------------------------------------------
initializeNetwork <- function(p, hidden, lev){
  K <- length(unique(lev))
  L <- length(hidden) + 1
  Wlist <- Blist <- vector(mode = "list", length = L)
  M <- c(p, hidden, length(lev))
  
  for (i in 1:(length(M)-1)) {
    Wlist[[i]] <- matrix(rnorm(M[i]*M[i+1], mean = 0, sd = 0.001), nrow = M[i], ncol = M[i+1])
    Blist[[i]] <- rep(rnorm(M[i+1], mean = 0, sd = 0.001), 1)
  }
  
  neuralNetwork <- list(Blist = Blist, Wlist = Wlist, lev = lev)
  return(neuralNetwork)
}


# Problem 3 -----------------------------------------------------------
feedForward <- function(x0, neuralNetwork, backPropagate = FALSE){
  a <- x0
  L <- length(neuralNetwork$Wlist)
  Alist <- Zlist <- vector(mode = "list", length = L)
  Zlist[[1]] <- a %*% neuralNetwork$Wlist[[1]] + neuralNetwork$Blist[[1]]
  Alist[[1]] <- sigmoid(Zlist[[1]])
  
  for (i in 2:L) {
    Zlist[[i]] <- Alist[[i-1]] %*% neuralNetwork$Wlist[[i]] + neuralNetwork$Blist[[i]]
    Alist[[i]] <- sigmoid(Zlist[[i]])
  }

  if (backPropagate == FALSE) {
    maxLevel <- lev[which.max(Alist[[L]])]
    return(maxLevel)
  }
  else {
    returnList <- list(Zlist = Zlist, Alist = Alist)
    return(returnList)
  }
}


# Problem 4 -----------------------------------------------------------
Bias1 <- c(0.89307, 31.35202, -15.53195)
Bias2 <- c(-0.05625, -1.89, 1.03616)
Blist <- list(Bias1 = Bias1, Bias2 = Bias2)

Weight1 <- matrix(c(
              0.07296, -4.04173, 1.83557,
              0.09986, -7.1124, 13.23802
            ), nrow = 2, ncol = 3, byrow = TRUE)
Weight2 <- matrix(c(
              1.38143, 1.11339, -0.0043,
              0.04027, 1.07048, -1.03302,
              -1.07657, 0.96546, -0.02152
            ), nrow = 3, ncol = 3, byrow = TRUE)
Wlist <- list(Weight1 = Weight1, Weight2 = Weight2)

lev <- c("setosa", "versicolor", "virginica")

neuralNetwork <- list(Blist = Blist, Wlist = Wlist, lev = lev)


# Problem 5 -----------------------------------------------------------
x0 <- c(6, 2.25)
feedForward(x0, neuralNetwork)     # Prediction of "virginica"


# Problem 6 -----------------------------------------------------------
transformGtoY <- function(G){
  # Attempt at one line
  t(apply(as.matrix(G), 1, function(x) x == unique(G)))
  # Thanks for the help on this one
  
  # Easier function
  # Y <- matrix(0, nrow = length(G), ncol = length(unique(G)))
  # for (i in 1:length(G)) {
  #   Y[i,] <- G[i] == unique(G)
  # }
}


# Problem 7 -----------------------------------------------------------
A_Function_Just_For_Example_9.3 <- function(X, Y, rate, initialVec, loops){
  N <- length(X)
  gradient <- theta <- matrix(0, nrow = loops, ncol = length(initialVec))
  theta[1,] <- initialVec
  gradient[1,] <- c(-2*sum(Y-theta[1,1]-theta[1,2]*X)/N, -2*sum((Y-theta[1,1]-theta[1,2]*X)*X)/N)
  
  for (i in 2:loops) {
    gradient[i,] <- c(-2*sum(Y-theta[i-1,1]-theta[i-1,2]*X)/N, -2*sum((Y-theta[i-1,1]-theta[i-1,2]*X)*X)/N)
    theta[i,] <- theta[i-1,] - rate * gradient[i,]
  }
  
  min <- sum((Y-theta[loops,1]-theta[loops,2]*X)^2)/N
  output <- list(minF = min, theta = theta[loops,])
  return(output)
}


X <- c(9.0, 2.7, 3.7, 5.7, 9.1, 2.0)
Y <- c(5.3, 7.2, 6.5, 5.9, 4.9, 7.3)
rate <- .015
initials <- c(6, .25)
# Arguments include (X, Y, learning rate, initial value vector, iterations for loop)
output <- A_Function_Just_For_Example_9.3(X, Y, rate, initials, 100000)    # Pretty fast at 100,000 loops
output$minF                                                                # A minimum f of 0.03144375
output$theta                                                               # Betas of (7.85, -0.31)


# Problem 8 -----------------------------------------------------------
# This is the function you made. This is brilliant!
backPropagate <- function(X, G, hidden, rate, iterations, batchSize){
  G <- as.matrix(G)
  bestCost <- Inf
  X <- scale(X)
  Y <- transformGtoY(G)
  neuralNetwork <- initializeNetwork(ncol(X), hidden, unique(G))
  L <- length(hidden) + 1
  
  for (i in 1:iterations) {
    indices <- sample(nrow(X), batchsize)
    tempX <- X[indices,]
    tempY <- Y[indices,]
    AandZ <- feedForward(tempX, neuralNetwork, backPropagate = TRUE)
    delta <- (AandZ$Alist[[L]]-tempY)*sigmoidPrime(AandZ$Zlist[[L]])
    Bgrad <- colMeans(delta)
    Wgrad <- WgradHelper(AandZ$Alist[[L-1]], delta)
    neuralNetwork$Blist[[L]] <- neuralNetwork$Blist[[L]] - rate*Bgrad
    neuralNetwork$Wlist[[L]] <- neuralNetwork$Wlist[[L]] - rate*Wgrad
    
    for (l in (L-1):1) {
      delta <- delta %*% t(neuralNetwork$Wlist[[l+1]])*sigmoidPrime(AandZ$Zlist[[l]])
      Bgrad <- colMeans(delta)
      
      if (l == 1) {
        Wgrad <- WgradHelper(tempX, delta)
      }
      else {
        Wgrad <- WgradHelper(AandZ$Alist[[l-1]], delta)
      }
      
      neuralNetwork$Blist[[l]] <- neuralNetwork$Blist[[l]] - rate*Bgrad
      neuralNetwork$Wlist[[l]] <- neuralNetwork$Wlist[[l]] - rate*Wgrad
    }
    
    a <- X
    
    for (l in 1:L) {
      a <- sigmoid(sweep(a %*% neuralNetwork$Wlist[[l]], 2, neuralNetwork$Blist[[l]], "+"))
    }
    
    cost <- sum((a - Y)^2)
    print(cost)
    
    if (cost < bestCost) {
      bestNetwork <- neuralNetwork
      bestCost <- cost
    }
  }
  
  return(bestNetwork)
}


# Problem 9 -----------------------------------------------------------
X <- iris[,1:4]
G <- iris$Species
hidden <- 4
rate <- 0.15
iterations <- 50000
batchsize <- 50
bestNeuralNetwork <- backPropagate(X, G, hidden, rate, iterations, batchsize) # This took 3 minutes lol

# Let's make a prediction
prediction <- c(5.2, 3.7, 1.4, 0.6)
x0 <- standardizeObservation(prediction, X)
feedForward(x0, bestNeuralNetwork)      # Prediction of "setosa"



# Let's check out the built-in function
library(neuralnet)
NN <- neuralnet(Species ~ Sepal.Width + Sepal.Length + Petal.Length + Petal.Width,
                data = iris,
                hidden = 4,
                learningrate = 0.15
                )
plot(NN, show.weights = FALSE)

# We will make the same prediction as before
pred <- data.frame(5.2, 3.7, 1.4, 0.6)
names(pred) <- colnames(X)
unique(G)[which.max(predict(NN, pred))]    # Prediction of "setosa"


