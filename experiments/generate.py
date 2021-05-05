from _context import former
from former import util, GTransformer

from util import d, here, tic, toc

import torch
import math
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, sys, math, gzip
from tqdm import tqdm

# NB, the enwik8 data contains tokens from 9 to 240, but well round up to the nearest
# power of two.
NUM_TOKENS = 256
# Used for converting between nats and bits
LOG2E = math.log2(math.e)

def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()

def loadCoco(dir):
    '''
    Loads in the coco dataset and returns an array of arrays of characters that have been converted to ints
    eg. 'Hello there' -> ['H','e','l','l','o', ' ', 't', 'h', 'e', 'r', 'e'] -> [72, 101, 108, 108, 111, 32, 116, 104, 101, 114, 101]
    '''
    # First open the file and read every line into an array of sentences
    file = open(dir, 'r')
    sentences = [line for line in file]

    # Second, loop through every sentence and create another array of integers
    converted_sentences = []
    for sentence in sentences:
        temp_array = [ord(character) for character in sentence]
        if len(temp_array) > 1: converted_sentences.append(temp_array)

    # Helper function to sort by length
    def byLength(array):
        return(len(array))

    converted_sentences.sort(key = byLength) #sort in acsending order
    
    return converted_sentences

class BatchSize(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class BatchSizeNone(BatchSize):
    def __init__(self):
        super().__init__("Please specifiy a batch size")

class BatchSizeNegative(BatchSize):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        super().__init__("cannot be negative")

    def __str__(self):
        return f'Batch size ({self.batch_size}) {self.message}'

class BatchSizeZero(BatchSize):
    def __init__(self):
        super().__init__("Batch Size must be bigger than 0")

def batchByTokens(array, batchSize = None):
    '''
    Batch the data by the length of tokens so that each batch has equal length arrays
    '''
    if batchSize is None:
        batchSize = 1
    else:
        if batchSize is None:
            raise BatchSizeNone()
        elif batchSize < 0:
            raise BatchSizeNegative(batchSize)
        elif batchSize == 0:
            raise BatchSizeZero() 

    batches = []
    start = 0 
    while start < len(array):
        end = start
        total = 0
        while total < batchSize and end < len(array):
            end += 1 
            total += len(array[end-1])

        # Give warning of sequence skip:
        if start == end:
            print('skipping sequence', start, 'of length',  len(array[start]))
            continue

        batches.append(array[start:end])

        start = end

    return batches

def pad(batch):
    '''
    Pad the batch with zero values and return a Tensored batch
    '''
    tokenLength = len(batch[-1])
    for seq in batch:
        while len(seq) < tokenLength:
            seq.append(0)

    return torch.Tensor(batch).to(torch.int64)
        
def splitArray(array, trainSplit = 0.9, valSplit = 0.05, testSplit = 0.05):
    '''
    Splits the data into specifies train, validation and test sizes, and returns a 3 Tensors representing them
    '''
    trainLength = round(len(array) * trainSplit) 
    valLength = round(len(array) * valSplit) 

    trainData = array[:trainLength]
    valData = array[trainLength:trainLength+valLength]
    testData = array[trainLength+valLength:]
    
    return trainData, valData, testData

def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """
    Load the enwik8 dataset from the Hutter challenge.

    Adapted from https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py
    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """
    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.fromstring(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

def sample_batch(data, length, batch_size):
    """
    Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model.

    For each input instance, it also slices out the sequence that is shofted one position to the right, to provide as a
    target for the model.

    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch.
    :param batch_size: The number of subsequences in the batch
    :return: A pair (input, target) of minteger matrices representing the input and target for the model.
    """

    # Sample the starting indices of the sequences to slice out.
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - length - 1)

    # Slice out the input sequences
    seqs_inputs  = [data[start:start + length] for start in starts]
    # -- the start index is the one we just sampled, and the end is exactly 'lentgh' positions after that.
    seqs_target = [data[start + 1:start + length + 1] for start in starts]
    # -- The target is the same sequence as input, except one character ahead (we are asking the model to predict the
    #    next character at each position)

    # We now have two lists of torch vectors, which we can concatenate into matrices of batch_size-by-length
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    # -- Note that we add a singleton dimenson to each vector, s[None.,:], and then concatenate along that dimension.

    return inputs, target

def compute_compression(model, data, context, batch_size):
    """
    Compute the _compression_ of a dataset under a model. That is, given a model, in how many bits could we represent
    the dataset. This requires us to turn a given probability distribution into a code for the outcomes.

    See [this video](https://youtu.be/mSneVjDvzNQ) for an explanation.

    :param model: A sequence-to-sequence model that takes as input a (sub) sequence of integers and produces a probability
    distributuion on the output.
    :param data: A singe list of integers representing the  data
    :return: The result of the computation in "bits per byte". That is, how many bits does the compressed representation
    spend on each byte (=ASCII character) of the raw data.
    """

    bits, tot = 0.0, 0
    batch = []
    # Buffer, every time it fills up, we run it through the model
    # --- For the sake of speed we want to process the data in batches. For each token in the data, we make a
    #     prediction based on all the `context` tokens before it. This means that for each subsequence in the batch, we
    #     need to shift the start/end indices ahead by one token.
    #
    #     After we pass the batch through the model, we look at only the probabilities predicted for the last token.

    for current in range(data.size(0)):

        fr = max(0, current - context)
        to = current + 1

        instance = data[fr:to].to(torch.long) # the subsequence of the data to add to the batch
        if instance.size(0) < context + 1:
            pad = torch.zeros(size=(context + 1 - instance.size(0),), dtype=torch.long)
            instance = torch.cat([pad, instance], dim=0)
            # -- the first tokens don't have enough tokens preceding them, so we pad them to the right size.

            assert instance.size(0) == context + 1 # all instances should be `context` + 1 long

        if torch.cuda.is_available():
            instance = instance.cuda()

        batch.append(instance[None, :])
        # -- We add a singleton dimension to concatenate along later.

        if len(batch) == batch_size or current == data.size(0) - 1:
            # batch is full or we are at the last instance, run it through the model

            b = len(batch)

            all = torch.cat(batch, dim=0)
            inputs = all[:, :-1]  # input
            target = all[:, -1]  # target values

            output = model(inputs)

            lnprobs = output[torch.arange(b, device=d()), -1, target]
            log2probs = lnprobs * LOG2E
            # -- The model produces natural logarithms of probabilities, but we need base-2 logarithms of the
            #    probabilities, since these give us bits.

            bits += - log2probs.sum() # Add the bits for each character (the negative log_2 probabilties) to the running total
            batch = []  # clear the buffer

    return bits / data.size(0) # bits-per-byte

def sample_sequence(model, seed, max_context, length=600, temperature=0.5, verbose=False):
    """
    Sequentially samples a sequence from the model, token by token.

    :param model:
    :param seed: The sequence to start with.
    :param length: The total number of characters to sample.
    :param temperature: The sampling temperature.
    :param verbose: If true, the sampled sequence is also printed as it is sampled.

    :return: The sampled sequence, including the seed.
    """

    file = open('former/generated_seqs/05-05-2021.txt', 'a')
    file.write('SEED:\n')
    sequence = seed.detach().clone()

    if verbose: # Print the seed, surrounded by square brackets
        # print('[', end='', flush=True)
        for c in seed:
            # print(str(chr(c)), end='', flush=True)
            file.write(str(chr(c)))
        # print(']', end='', flush=True)
        file.write('\nGENERATED:\n')

    for _ in range(length):

        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[-max_context:]

        # Run the current input through the model
        output = model(input[None, :])

        # Sample the next token from the probabilitys at the last position of the output.
        c = sample(output[0, -1, :], temperature)

        if verbose:
            # print(str(chr(c)), end='', flush=True)
            file.write(str(chr(c)))

        sequence = torch.cat([sequence, c[None]], dim=0) # Append the sampled token to the sequence

    file.write('\n')
    file.close()

    print()
    return seed

# def go(arg):

#     if arg.seed < 0:
#         seed = random.randint(0, 1000000)
#         print('random seed: ', seed)
#     else:
#         torch.manual_seed(arg.seed)

#     tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging

#     # load the data (validation unless arg.final is true, then test)
#     arg.data = here('data/enwik8.gz') if arg.data is None else arg.data

#     data_train, data_val, data_test = enwik8(arg.data)
#     data_train, data_test = (torch.cat([data_train, data_val], dim=0), data_test) \
#                             if arg.final else (data_train, data_val)

#     # create the model
#     model = GTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=arg.context, num_tokens=NUM_TOKENS, attention_type=arg.attention_type)
#     if torch.cuda.is_available():
#         model.cuda()

#     opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())

#     # Linear learning rate warmup
#     sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

#     # Training loop
#     # -- We don't loop over the data, instead we sample a batch of random subsequences each time. This is not strictly
#     #    better or worse as a training method, it's just a little simpler.
#     #
#     instances_seen = 0
#     for i in tqdm.trange(arg.num_batches):

#         opt.zero_grad()

#         source, target = sample_batch(data_train, length=arg.context, batch_size=arg.batch_size)
#         instances_seen += source.size(0)

#         if torch.cuda.is_available():
#             source, target = source.cuda(), target.cuda()

#         tic()
#         output = model(source) # forward pass
#         t = toc()

#         # Compute the loss
#         loss = F.nll_loss(output.transpose(2, 1), target, reduction='mean')

#         tbw.add_scalar('transformer/train-loss', float(loss.item()) * LOG2E, i * arg.batch_size, instances_seen)
#         tbw.add_scalar('transformer/time-forward', t, instances_seen)

#         loss.backward() # backward pass

#         # clip gradients
#         # -- If the total gradient vector has a length > x, we clip it back down to x.
#         if arg.gradient_clipping > 0.0:
#             nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

#         opt.step() # stochastic gradient descent step
#         sch.step() # update the learning rate

#         # Validate every `arg.test_every` steps. First we compute the
#         # compression on the validation data (or a subset),
#         # then we generate some random text to monitor progress.
#         if i != 0 and (i % arg.test_every == 0 or i == arg.num_batches - 1):
#             with torch.no_grad():

#                 ## Sample and print a random sequence

#                 # Slice a random seed from the test data, and sample a continuation from the model.
#                 seedfr = random.randint(0, data_test.size(0) - arg.context)
#                 seed = data_test[seedfr:seedfr + arg.context].to(torch.long)

#                 if torch.cuda.is_available():
#                     seed = seed.cuda()

#                 sample_sequence(model, seed=seed, max_context=arg.context, verbose=True, length=arg.sample_length)

#                 ## Compute validation bits per byte

#                 upto = data_test.size(0) if i == arg.num_batches - 1 else arg.test_subset
#                 data_sub = data_test[:upto]

#                 bits_per_byte = compute_compression(model, data_sub, context=arg.context, batch_size=arg.test_batchsize)
#                 # -- Since we're not computing gradients, we can increase the batch size a little from what we used in
#                 #    training.

#                 print(f'epoch{i}: {bits_per_byte:.4} bits per byte')
#                 tbw.add_scalar(f'transformer/eval-loss', bits_per_byte, i * arg.batch_size, instances_seen)
#                 # -- 0.9 bit per byte is around the state of the art.

def go(arg):
    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging

    # load the data (validation unless arg.final is true, then test)
    data = loadCoco('former/data/coco.valannotations.txt')
    batches = batchByTokens(data, batchSize=arg.batch_size)
    trainBatches, valBatches, testBatches = splitArray(batches)

    # create the model
    model = GTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=arg.context, num_tokens=NUM_TOKENS, attention_type=arg.attention_type)
    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())

    # Linear learning rate warmup
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

    # Training loop
    # -- We don't loop over the data, instead we sample a batch of random subsequences each time. This is not strictly
    #    better or worse as a training method, it's just a little simpler.
    #
    instances_seen = 0
    for epoch in range(arg.num_epochs):
        
        print(f'EPOCH {epoch + 1} STARTING')

        for batch in tqdm(trainBatches):
            
            opt.zero_grad() # Set gradients to z0

            batch_tensor = pad(batch) # Pad the batch with 0 values

            # Create the input and target
            inputs = batch_tensor[:, :-1]
            target = batch_tensor[:, 1:]

            # Check if over the limit, and chop off the anything after the length of NUM_TOKENS
            if inputs.size(1) > NUM_TOKENS:
                inputs = inputs[:, :NUM_TOKENS]
                target = target[:, :NUM_TOKENS]

            output = model(inputs) # Compute the output of the model via the input
            
            loss = F.nll_loss(output.transpose(2, 1), target, reduction='mean') # Compute the loss

            # Add the loss (logged) to the Tensorboard with instances seen
            instances_seen += inputs.size(0)
            tbw.add_scalar('transformer/train-loss', float(loss.item()) * LOG2E, instances_seen)

            loss.backward() # Backpropagate

            opt.step() # Do one step of Adam
            
            sch.step() # Update the learning rate

        print(f'EPOCH {epoch + 1} FINISHED. \nGENERATING SAMPLE')

        # WRITE TO FILE (For logging generated sequence per epoch)
        file = open('former/generated_seqs/05-05-2021.txt', 'a')
        file.write('---------------------------------------------------------------------------------------------\n')
        file.write(f'EPOCH {epoch + 1}:\n')
        file.close()

        # GENERATE SAMPLE
        with torch.no_grad():

            randomBatchIndex = random.randint(0, len(testBatches))
            randomBatch = testBatches[randomBatchIndex]
            t = pad(randomBatch)
            seed = t[0]

            if torch.cuda.is_available():
                seed = seed.cuda()

            sample_sequence(model, seed=seed, max_context=arg.context, verbose=True, length=arg.sample_length)

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=30, type=int)

    parser.add_argument("-N", "--num-batches",
                        dest="num_batches",
                        help="Number of batches to train on. Each batch contains randomly sampled subsequences of the data.",
                        default=50, type=int) # No longer needed

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=256, type=int)

    parser.add_argument("-D", "--data", dest="data",
                        help="Data file. Will be read as a string of 8-bit characters.",
                        default=None)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-T", "--tb-dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-C", "--context", dest="context",
                        help="Length of the sequences extracted from the corpus (and the context used during inference).",
                        default=256, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of transformer blocks)",
                        default=12, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--test-every",
                        dest="test_every",
                        help="How many batches between tests.",
                        default=1500, type=int)

    parser.add_argument("--test-subset",
                        dest="test_subset",
                        help="A subset for the validation tests.",
                        default=100000, type=int)

    parser.add_argument("--test-batchsize",
                        dest="test_batchsize",
                        help="Batch size for computing the validation loss. This can be a bit bigger than the training batch size.",
                        default=64, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=5000, type=int)

    parser.add_argument("--sample-length",
                        dest="sample_length",
                        help="Number of character to sample.",
                        default=128, type=int)

    parser.add_argument("--attention-type", dest="attention_type",
                        help="Which type of self-attention to use (default, gpt2, wide, narrow)",
                        default="default", type=str)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)

