from argparse import ArgumentParser

from _context import former
from former import TransformerVAE
from generate import load_financial_data, batchByTokens, pad, splitArray

from sklearn.manifold import TSNE

import torch
import os
import seaborn as sns
from fpdf import FPDF

def go(arg):
    assert arg.model_path is not None, 'Need a path to the model (.pt file)'
    model_pt = torch.load(arg.model_path)

    # Create model
    model = TransformerVAE(
        emb=model_pt['emb'], 
        heads=model_pt['heads'], 
        depth=model_pt['depth'], 
        seq_length=model_pt['seq_length'], 
        num_tokens=model_pt['num_tokens'], 
        attention_type=model_pt['attention_type'],
        dropoutProb=model_pt['dropoutProb'],
        latentSize=model_pt['latentSize'],
        wordDropout=model_pt['wordDropout'])

    # Load in model
    model.load_state_dict(model_pt['model_state_dict'])

    # Load in data
    data = load_financial_data('former/data/EURUSD240.csv')
    batches = batchByTokens(data, batchSize=256)
    trainBatches, valBatches, testBatches = splitArray(batches)

    # Pass into model
    batch_index = arg.batch -1
    seq_index = arg.sequence -1

    if arg.takeFromValData:
        batch = valBatches[batch_index]
    else:
        batch = testBatches[batch_index]

    batch = pad(batch) # Pad the batch to get Tensor

    # Generate Z
    z = model.generate_zprime(batch)

    # Take certain sequence
    z = z[seq_index].detach()

    # Visualise
    tsne = TSNE()
    z_2d = tsne.fit_transform(z)

    # Make pdf with plot
    pdf = FPDF()
    pdf.add_page()
    plot = sns.scatterplot(x=z_2d[:,0], y=z_2d[:,1]).figure
    png_path = f'former/latent_representations/latent.png'
    plot.savefig(png_path)
    pdf.image(png_path, w=150)

    # Save pdf
    pdf.output(f"former/latent_representations/{arg.model_path.split('/')[2]}_batch-{arg.batch}_seq-{arg.sequence}.pdf", "F") # Save File
    os.remove(png_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("--model-dir",
                        dest="model_path",
                        help="Relative Directory Path where the .pt file is stored",
                        default=None, type=str)

    parser.add_argument("--from-batch",
                        dest="batch",
                        help="Which batch to pass in the model",
                        default=1, type=int)

    parser.add_argument("--from-sequence",
                        dest="sequence",
                        help="Which sequence of the batch to get a latent representation of",
                        default=1, type=int)

    parser.add_argument("--from-val",
                        dest="takeFromValData",
                        help="Take from Validation data (otherwise it takes from Test)",
                        default=False, type=bool)

    options = parser.parse_args()

    go(options)