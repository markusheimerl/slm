# slm
A small language model implementation

It could be so simple. Just do exactly what is done in ssm/gpu/ssm.h and ssm/gpu/ssm.c with these differences:

Data isn't readily prepared, because embeddings will change in float values between batches. That's why we can't produce a csv before starting training.

But other than that it is so simple:
You just pick a character from BATCH_SIZE locations of the combined corpus, embed that character to a vector. Thats the input, the X. This vector has EMBED_DIM also sometimes called TOKEN_DIM values.

Then you take the following character and use its ASCII value to create a one hot encoded vector, where there is a single 1.0 at the index that represents the ASCII value of that character. That's the output.

Now you group all of these examples. That's our first batch. Then after that one is through forward and backward pass, we move the BATCH_SIZE locations in the corpus one forward and do the same thing again. This works because we accumulate the context inside of the hidden state of the state space model.
