from argparse import ArgumentParser

from _context import former
from former import TransformerVAE
from generate import load_financial_data, batchByTokens, pad, splitArray

from sklearn.manifold import TSNE

import torch
import os
import seaborn as sns
import pandas as pd
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
    data = batchByTokens(data, batchSize=256)

    emb = model_pt['latentSize']

    with torch.no_grad():
        outputs = []
        for index, batch in enumerate(data):
            input = pad(batch)
            output = model.encoder(input)['output']
            output = output[:, :emb] # mean
            outputs.append(output)
        z = torch.cat(outputs, dim=0)

    # Visualise
    tsne = TSNE()
    z_2d = tsne.fit_transform(z)

    # Make pdf with plot
    pdf = FPDF()
    pdf.add_page()

    d = {'x': z_2d[:,0], 'y': z_2d[:,1]}
    df = pd.DataFrame(d)
    if arg.sequence > 0: 
        seq = df.iloc[arg.sequence-1]
        seq = pd.DataFrame({'x': [seq['x']], 'y': [seq['y']]})
        df = df.drop(arg.sequence-1)

    # plot = sns.scatterplot(x=z_2d[:,0], y=z_2d[:,1]).figure
    plot = sns.scatterplot(x='x', y='y', data=df, alpha=.5).figure
    plot = sns.scatterplot(x='x', y='y', data=seq, palette='red').figure
    png_path = f'former/latent_representations/latent.png'
    plot.savefig(png_path)
    pdf.image(png_path, w=150)

    # Save pdf
    i = 0
    path = f"former/latent_representations/{arg.model_path.split('/')[2]}_sequence-{arg.sequence}_#{i}.pdf"
    while os.path.exists(path):
        i+=1
        path = f"former/latent_representations/{arg.model_path.split('/')[2]}_sequence-{arg.sequence}_#{i}.pdf"
    pdf.output(path, "F") # Save File
    os.remove(png_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("--model-dir",
                        dest="model_path",
                        help="Relative Directory Path where the .pt file is stored",
                        default=None, type=str)

    parser.add_argument("--from-sequence",
                        dest="sequence",
                        help="The sequence you want highlighted in the plot",
                        default=0, type=int)

    options = parser.parse_args()

    go(options)