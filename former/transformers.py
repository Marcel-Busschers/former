import torch
from torch import nn
import torch.nn.functional as F

from .modules import TransformerBlock

from .util import d

class TransformerVAE(nn.Module):
    def __init__(self, emb, heads, depth, seq_length, num_tokens, max_pool=True, attention_type='default'):
        super().__init__()

        self.emb = emb
        
        self.encoder = EncoderTransformer(emb, heads, depth, seq_length, num_tokens, max_pool)
        self.decoder = DecoderTransformer(emb, heads, depth, seq_length, num_tokens)

        self.toSampledSequence = nn.Linear(20, emb)

    def kl_loss():
        zmean = self.zmean; zsig = self.zsig
        return 0.5 * torch.sum(zsig.exp() - zsig + zmean.pow(2) - 1, dim=1)

    def sample(zmean, zsig):
        b, l = zmean.size()

        # sample epsilon from a standard normal distribution
        eps = torch.randn(b, l)

        # transform eps to a sample from the given distribution
        return zmean + eps * (zsig * 0.5).exp()

    def forward(self, x):
        z = self.encoder(x) # Encoder spits out z vector (already transformed to smaller size for mean and sigma)

        # Split z vector into zmean and zsigma
        self.zmean = z[:, :20]
        self.zsig = z[:, 20:]

        zprime = sample(self.zmean, self.zsig) # sample z' using mean and sigma

        zprime = self.toSampledSequence(zprime) # upscale z' to token length

        zprime = zprime[:, None, :] # Creates another dimension (b, 1, emb)

        b, t, e = z['batchSize'], z['timeDimension'], z['embeddingSize']

        zprime = zprime.expand(b, t, e) # Expand to add embedding vector

        output = self.decoder(x, zprime)

        return output

class EncoderTransformer(nn.Module):
    """
    Encoder for representing the sequence in (compressed) format
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, max_pool=True, attention_type='default'):
        super().__init__()

        self.max_pool = max_pool

        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, attention_type=attention_type))

        self.tblocks = nn.Sequential(*tblocks)

        self.toZ = nn.Linear(emb, 40)

    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        x = self.toZ(x)

        return {'output': x, 'batchSize': b, 'timeDimension': t, 'embeddingSize': e}

class DecoderTransformer(nn.Module):
    """
    Decoder for generating text (character by character).
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, attention_type='default'):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True, attention_type=attention_type))

        self.tblocks = nn.ModuleList(*tblocks) 

        self.toprobs = nn.Linear(emb, num_tokens)

    def forward(self, x, zprime):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions

        # Pass output from Encoder through every layer of the the Decoder.
        #  - This will help the gradients propagate to the encoder, since they don't have to pass through all layers of the decoder first.
        for block in self.tblocks:
            x = block(x + zprime)

        x = self.toprobs(x.view(b*t, e)).view(b, t, self.num_tokens)

        return F.log_softmax(x, dim=2)

class CTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0, wide=False):
        """
        :param emb: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_tokens: Number of tokens (usually words) in the vocabulary
        :param num_classes: Number of classes.
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        """
        super().__init__()

        self.num_tokens, self.max_pool = num_tokens, max_pool

        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout, wide=wide))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_classes)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions
        x = self.do(x)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)

