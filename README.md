# HPO solver based on Variational Autoencoders

## Abstract
This is part of a coding challenge based on the work of the machine learning group at Nuremberg University at this [*repo*] {https://github.com/machinelearningnuremberg/HPO-B} .

The hyperparameter optimization is finding the hayperparameter configuration that maximize a machine learning pipeline. Hyperparameter Optimization (HPO) is a black-box search problem where the goal is to find the optimal hyperparameter configuration for a machine learning model by evaluating different configurations and measuring their validation accuracy. Unlike model parameter optimization, which uses gradients, HPO relies on black-box evaluations because the gradient of the validation accuracy with respect to hyperparameters is not easily computable. Bayesian optimization is a common approach for HPO, using a performance estimator and an acquisition function to recommend the next configuration with the highest likelihood of improving performance based on past evaluations.

In this repo I implemented a new method of black-box optimization technique for HPO based on Variational Autoencoders as a generative model.

I created a new script in ”methods” folder called [*”generative_hpo.py”*]{https://github.com/youssefnassar95/HPO-B/blob/main/methods/generative_hpo.py} which has the method ”observe_and_suggest”.
Then I created the script to run which is *[”example_gen.py”*]{https://github.com/youssefnassar95/HPO-B/blob/main/example_gen.py} that uses "HPOBHandler” class and do the evaluation with the ”evaluate continuous” method. This script iterates over the different seeds and datasets, after that it saves the results in a nested dictionary following the same structure, but without adding the x configuration in it, as for ”evaluate continuous” method does not return these configuration but only the accuracy.

## Chosen Model Architecture
The model consists of 3 main parts: 
1) The Encoder part of the VAE
2) The Decoder part of the VAE 
3) The Transformer. 

### Transformer architecture
For the Transoformer We have an embedding layer before it, I used an MLP embedding model with 2 layers, the input size is 17 which is 16 hyperparameters from x and accuracy y. The output dimension is 32 which was chosen arbitrary. 
For the MLP embedding, C as input will be in dimension BxCx17 we reshape it to be B*Cx17 pass it to the embedding layer, then the output will be B*Cx32, which we reshape it again to BxCx32.

I chose a Transformer Encoder, as we use it for context to understand the history H, we
don’t need generation of data from the Transformer as this is the task for the VAE, also it is more efficient and faster to train the Transformer encoder. We have the encoder layer with input dimension of 32 to match the output feature dimension from the embedding layer, with 4 heads, and we have 3 sub-encoder-layers. The output of the Transformer encoder is then averaged over the sequence dimension giving us a tensor of dimension Bx32, we average it to be able to be passed to the VAE as it takes only fixed number of inputs, and since the C in the training and evaluation is changing then we do the averaging as a method to fix the dimension, there can be also that I choose only the first vector of the sequence but in here I did the first option. Then the output of the Transformer encoder is concatenated with the sampled x and I giving a tensor of dimension Bx49 that will be passed to the VAE.

### VAE architecture
The VAE is consists of encoder and decoder, the encoder is consisted of 2 linear layers,
has input of 49, hidden dimension of 32 and latent dimension of 64. Then, we have 2
linear layers to get μ and σ that each is 2 dimensional. The output of the encoder will be
passed to these 2 layers, then do the re-parameterization to get the z which is then passed to the decoder alongside the I and the output of the Transformer encoder. The decoder is a 3 layer model that has input dimension of 35 which is 32 + 2 + 1 which are, the Transformer output, z dimension and dimension of I respectively, latent dimension of 64, hidden dimension of 32 and output dimension of 16 which will be the reconstructed x which has 16 hyperparameters.