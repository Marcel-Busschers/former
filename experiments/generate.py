from _context import former
from former import util, TransformerVAE, DecoderTransformer

from util import d, here, tic, toc

import torch
import math
import string
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist

import numpy as np
import pandas as pd

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as date
import os
import seaborn as sns
from fpdf import FPDF

import random, sys, math, gzip
from tqdm import tqdm

# NB, the enwik8 data contains tokens from 9 to 240, but well round up to the nearest
# power of two.
NUM_TOKENS = 256
START_TOKEN = 169
END_TOKEN = 174
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
        temp_array = [ord(character) for character in sentence if ord(character) >= 32 and ord(character) <= 126]
        temp_array.insert(0, START_TOKEN) # Start-of-sequence token
        temp_array.append(END_TOKEN) # End-of-sequence token
        if len(temp_array) > 3: converted_sentences.append(temp_array) # bigger than 3 because: START token, END token and New Line character

    # Helper function to sort by length
    def byLength(array):
        return(len(array))

    converted_sentences.sort(key = byLength) #sort in acsending order

    return converted_sentences

def load_financial_data(dir):
  df = pd.read_csv(dir, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], sep='\t')

  close_list = df['Close'].values.tolist()

  # close_list = [x*100_000 for x in close_list]

  chunks = [close_list[x:x+64] for x in range(0, len(close_list), 64)]

  return chunks

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
            total += len(array[end-1])#**2

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
    tokenLength = len(batch[0])
    for seq in batch:
        while len(seq) < tokenLength:
            seq.append(0)

    return torch.Tensor(batch).to(torch.float)
        
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

def sample_sequence(model, seed, max_context, fileName, log=False, length=600, temperature=0.5):
    """
    Sequentially samples a sequence from the model, token by token.

    :param model:
    :param seed: The sequence to start with.
    :param length: The total number of characters to sample.
    :param temperature: The sampling temperature.
    :param log: If true, the sampled sequence is also logged.
    :param fileName: Name of the file to log generations

    :return: The sampled sequence, including the seed.
    """

    if log: file = open(fileName, 'a')
    
    sequence = seed.detach().clone()

    if log: file.write('\nSEED:\n')
    for c in sequence:
        character = chr(c)
        print(character, end='', flush=True)
        if log: file.write(character)

    print()
    if log: file.write('\nGENERATED:\n')

    zprime = model.generate_zprime(sequence[None, :]) # Generate z'
    zprime = zprime[:, -1] # remove last character to match size of input, since there is one extra character (end-of-sequence)

    sequence = torch.tensor([START_TOKEN])

    for _ in range(length):

        # Input is the START token
        input = sequence[-max_context:]

        # Run the START token along with z' through the decoder
        output = model.decoder(input[None, :], zprime, dropout=False)

        # Sample the next token from the probabilitys at the last position of the output.
        c = sample(output[0, -1, :], temperature)

        if c == END_TOKEN: break

        character = chr(c)

        if c >= 32 and c <= 126:
            print(character, end='', flush=True)
            if log: file.write(character)

        sequence = torch.cat([sequence, c[None]], dim=0) # Append the sampled token to the sequence

    if log:
        file.write('\n')
        file.close()

    print('\n')
    return seed

def go(arg):
    path = arg.runDirectory + arg.currentDateDir
    tensorboard_path = path + arg.tb_dir
    checkpoint_path = path + arg.checkpointName

    if arg.logGenerations: 
        os.mkdir(path)
        os.mkdir(tensorboard_path)
        os.mkdir(checkpoint_path)
        tbw = SummaryWriter(log_dir=tensorboard_path) # Tensorboard logging

        pdf = FPDF()
        pngs = []

    # load the data (validation unless arg.final is true, then test)
    # data = loadCoco('former/data/coco.valannotations.txt')
    data = load_financial_data('former/data/EURUSD240.csv')
    batches = batchByTokens(data, batchSize=arg.batch_size)
    trainBatches, valBatches, testBatches = splitArray(batches)

    # We train on single batch to overfit (if set to true)
    if arg.trainOnSingleBatch:
        trainBatches = [trainBatches[0]]
        testBatches = [trainBatches[0]]

    # create the model
    model = TransformerVAE(
        emb=arg.embedding_size, 
        heads=arg.num_heads, 
        depth=arg.depth, 
        seq_length=arg.context, 
        num_tokens=NUM_TOKENS, 
        attention_type=arg.attention_type,
        dropoutProb=arg.dropoutProbability,
        latentSize=arg.latentSize,
        wordDropout=arg.implementWordDropout)
    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())

    # Linear learning rate warmup
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

    # Load Model
    if arg.modelDirectory is not None:
        checkpoint = torch.load(arg.modelDirectory)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimiser_state_dict'])
        sch.load_state_dict(checkpoint['lr_state_dict'])
        kl = checkpoint['kl_loss']
        rec = checkpoint['reconstruction_loss']
        loss = checkpoint['total_loss']

    # Training loop
    # -- We don't loop over the data, instead we sample a batch of random subsequences each time. This is not strictly
    #    better or worse as a training method, it's just a little simpler.
    #
    instances_seen = 0
    for epoch in range(arg.num_epochs):
        
        print(f'EPOCH {epoch + 1} STARTING')

        for batch in tqdm(trainBatches):

            opt.zero_grad() # Set gradients to 0

            batch_tensor = pad(batch) # Pad the batch with 0 values

            # Check if over the limit, and chop off the anything after the length of NUM_TOKENS
            if batch_tensor.size(1) > NUM_TOKENS:
                batch_tensor = batch_tensor[:, :NUM_TOKENS]

            output = model(batch_tensor) # Compute the output of the model via the input (being the batch_tensor)

            kl = model.kl_loss(arg.betaValue)[0] # Compute the Kullback–Leibler divergence for the model's loss
            rec = nn.GaussianNLLLoss()(output['mean'], batch_tensor[:,1:], output['variance']) #output, target, var
            loss = (torch.maximum(kl, torch.tensor(arg.lambdaValue)) + rec).mean() # Total loss

            loss.backward() # Backpropagate

            # Log to tensorboard
            instances_seen += batch_tensor.size(0)
            if arg.logGenerations: 
                # Record the Model Losses
                tbw.add_scalar('VAE/kl-loss', kl, instances_seen)
                tbw.add_scalar('VAE/reconstruction-loss', rec, instances_seen)
                tbw.add_scalar('VAE/total-loss', loss, instances_seen)

                # Record the Gradient Norms            
                # ENCODER
                total_norm = 0
                for p in model.encoder.parameters():
                    param_norm = p.grad.data.pow(2).sum()
                    total_norm += param_norm
                total_norm = total_norm ** (1. / 2)
                tbw.add_scalar('VAE/encoder-gradient-norm', total_norm, instances_seen)

                # DECODER
                total_norm = 0
                for p in model.decoder.parameters():
                    param_norm = p.grad.data.pow(2).sum()
                    total_norm += param_norm
                total_norm = total_norm ** (1. / 2)
                tbw.add_scalar('VAE/decoder-gradient-norm', total_norm, instances_seen)

            opt.step() # Do one step of Adam
            
            sch.step() # Update the learning rate

        # Model Checkpoint (Only save model on last epoch)
        if arg.logGenerations and epoch == range(arg.num_epochs)[-1]:
            epoch_path = checkpoint_path + f'/epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': opt.state_dict(),
                'lr_state_dict': sch.state_dict(),
                'kl_loss': kl,
                'reconstruction_loss': rec,
                'total_loss': loss,
                ########## Model Info ##########
                'emb': arg.embedding_size, 
                'heads': arg.num_heads, 
                'depth': arg.depth, 
                'seq_length': arg.context, 
                'num_tokens': NUM_TOKENS, 
                'attention_type': arg.attention_type,
                'dropoutProb': arg.dropoutProbability,
                'latentSize': arg.latentSize,
                'wordDropout': arg.implementWordDropout
            }, epoch_path)

        print(f'EPOCH {epoch + 1} FINISHED. \n')

        if arg.num_epochs <= 10: perEpoch = 1
        elif arg.num_epochs <= 100: perEpoch = 10
        elif arg.num_epochs <= 1000: perEpoch = 100
        elif arg.num_epochs <= 50000: perEpoch = 1000
        else: perEpoch = 10000

        # WRITE TO FILE (For logging reconstructed sequence per epoch)
        if arg.logGenerations and (epoch+1) % perEpoch == 0:
            # Write info to txt
            with open(f'{path}/info.txt', 'a') as file:
                if (epoch+1) == perEpoch: 
                    if arg.infoMessage is not None: file.write(f'{arg.infoMessage}\n')
                    file.write(f'\n{arg}')

            # Get random Seed
            if arg.trainOnSingleBatch: 
                randomBatch = testBatches[0]
            else:
                randomBatchIndex = random.randint(0, len(testBatches)-2)
                randomBatch = testBatches[randomBatchIndex]
            seed = pad(randomBatch)
            if torch.cuda.is_available():
                seed = seed.cuda()

            # Pass through model
            out = model(seed)
            
            # Record Reconstruction
            input_data = seed[0,1:]
            output_data = out['mean'][0,:].detach()

            pdf.add_page()

            if (epoch+1) == perEpoch and arg.infoMessage is not None:
                pdf.set_font('Arial', '', 12)
                pdf.multi_cell(0, 5, f'{arg.infoMessage}')
                pdf.set_font('Arial', 'B', 16)
                pdf.ln(5)

            # Write Epoch number and Rec Loss
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(40, 10, f'Epoch {epoch+1}')
            pdf.ln(10)
            pdf.set_font('Arial', '', 14)
            pdf.multi_cell(0, 5, f'Reconstruction Loss: {rec}')
            pdf.ln(10)

            # Plot Input
            plot = sns.lineplot(data=input_data).set_title('INPUT').figure
            png_path = f'{path}/input{epoch}.png'
            plot.savefig(png_path)
            pngs.append(png_path)
            pdf.image(png_path, w=150)
            plot.clf()

            # Plot Output
            plot = sns.lineplot(data=np.around(output_data, decimals=5)).set_title('OUTPUT').figure
            png_path = f'{path}/output{epoch}.png'
            plot.savefig(png_path)
            pngs.append(png_path)
            pdf.image(png_path, w=150)
            plot.clf()
            
    if arg.logGenerations:
        pdf.output(f"{path}/reconstructed.pdf", "F") # Save File

        # Delete left overs
        for path in pngs:
            os.remove(path)

        # GENERATE SAMPLE
        # with torch.no_grad():
        #
        #     # Get a random sequence for generation
        #     randomBatchIndex = random.randint(0, len(testBatches)-1)
        #     randomBatch = testBatches[randomBatchIndex]
        #     t = pad(randomBatch)
        #     seed = t[-1]
        #     if torch.cuda.is_available():
        #         seed = seed.cuda()
        #
        #     sample_sequence(model, seed=seed, max_context=arg.context, log=arg.logGenerations, fileName=logName, length=arg.sample_length)

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()
    currentDate = date.now().strftime('%Y-%m-%d_%H:%M')

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=80, type=int)

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
                        default=0.0003, type=float)

    parser.add_argument("-T", "--tb-dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='/tensorboard_log')

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
                        default="gpt2", type=str)

    parser.add_argument("--log", dest="logGenerations",
                        help="Log to tensorboard (also writes generations to file)",
                        default=False, type=bool)

    parser.add_argument("--date-dir", dest="currentDateDir",
                        help="The current date (used for directory naming",
                        default=f'/{currentDate}')

    parser.add_argument("--checkpoint-dir", dest="checkpointName",
                        help="the directory where the model gets saved",
                        default='/checkpoint_saves')
    
    parser.add_argument("--run-dir", dest="runDirectory",
                        help="The directory where the model runs are",
                        default='./former/runs')

    parser.add_argument("--beta", dest="betaValue",
                        help="Beta Annealing perameter used to clip the KL loss (helps Decoder Collapse)",
                        default=1.0, type=float)

    parser.add_argument("--load-model", dest="modelDirectory",
                        help="The path to the .pt model save file (To continue training on that model)",
                        default=None, type=str)

    parser.add_argument("-m", "--message", dest="infoMessage",
                        help="A message prepended to the Generated txt (To give more context)",
                        default=None, type=str)

    parser.add_argument("--word-dropout", dest="implementWordDropout",
                        help="TRUE/FALSE parameter to implement word drop out (Helps with decoder collapse)",
                        default=False, type=bool)

    parser.add_argument("--dropout-prob", dest="dropoutProbability",
                        help="Probability value for Word Dropout (see --word-dropout)",
                        default=1.0, type=float)

    parser.add_argument("--free-bits", dest="lambdaValue",
                        help="Sets a constraint on the KL Loss (Helps decoder collapse)",
                        default=0.0, type=float)

    parser.add_argument("-L", "--latent-size", dest="latentSize",
                        help="Length of the embedded Latent Vector",
                        default=28, type=int)

    parser.add_argument("--single-batch", dest="trainOnSingleBatch",
                        help="TRUE/FALSE parameter for training on single batch (to overfit)",
                        default=False, type=bool)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)

