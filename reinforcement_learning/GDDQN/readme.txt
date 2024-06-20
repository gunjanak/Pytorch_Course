Implementation of Gaussian Double Deep Q-Network using PyTorch.
This algorithm extends the traditional Double DQN to handle uncertainties by predicting both the mean and standard deviation
of Q-values using a Gaussian distribution. 
It utilizes KL divergence for the loss function to better measure and minimize the 
  divergence between predicted and target Q-value distributions, 
providing a more robust approach for reinforcement
learning tasks where capturing uncertainty is important.
