# MSc Meeting

## Admin

- With SORA, deadline is 27/09/21 - 13 weeks
- LSA is 16/08 - 03/09

- https://arxiv.org/pdf/2106.09647.pdf

- Office hours are 7am - 9am Fridays

## Plan

- run sample through network, calculate grads of weights wrt output as a proxy for relevance of that parameter to the output
- Flatten weight grads, calc cosine similarity between train and test samples
- computationally difficult, basically a copy of the model for each sample
- could use sparse representation by selecting top n contributions and zeroing everything else
- is this functionally equivalent to just comparing the raw samples? Can we compare the raw samples or attribution scores as a baseline?

