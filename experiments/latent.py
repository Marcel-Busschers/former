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
    if arg.sequence is None: exit()
    assert arg.model_path is not None, 'Need a path to the model (.pt file)'
    model_pt = torch.load(arg.model_path)
    date_path = f'former/runs/{arg.model_path.split("/")[2]}/latent_representations'
    if not os.path.exists(date_path):
        os.mkdir(date_path)

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
    pngs = []

    if arg.sequence[0] == 0:
        seq = range(1,365,1)
        description = 'all'
    else:
        seq = arg.sequence
        description = arg.description
        assert description is not None, 'Please give a description using argument -d'

    for sequence in seq:
        assert sequence > 0, 'Sequence number must be greater than 0'
        pdf.add_page()

        d = {'x': z_2d[:,0], 'y': z_2d[:,1]}
        df = pd.DataFrame(d)
        seq = df.iloc[sequence-1]
        seq = pd.DataFrame({'x': [seq['x']], 'y': [seq['y']]})
        df = df.drop(sequence-1)

        # plot = sns.scatterplot(x=z_2d[:,0], y=z_2d[:,1]).figure
        plot = sns.scatterplot(x='x', y='y', data=df, alpha=.5).figure
        plot = sns.scatterplot(x='x', y='y', data=seq, palette='red').set_title(f'Sequence {sequence}').figure
        png_path = f'{date_path}/latent{sequence}.png'
        pngs.append(png_path)
        plot.savefig(png_path)
        pdf.image(png_path, w=150)
        plot.clf()

        index = 0
        s = None
        for batch in data:
            for seq in batch:
                index += 1
                if index == sequence: 
                    s = seq
                    break
        
        plot = sns.lineplot(x=range(len(s)), y=s).set_title(f'Sequence {sequence} market').figure
        png_path = f'{date_path}/market{sequence}.png'
        pngs.append(png_path)
        plot.savefig(png_path)
        pdf.image(png_path, w=150)
        plot.clf()


    # Save pdf
    i = 0
    path = f"{date_path}/{description}.pdf"
    while os.path.exists(path):
        i+=1
        path = f"{date_path}/{description}-{i}.pdf"
    pdf.output(path, "F") # Save File
    for path in pngs:
        os.remove(path)


if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("--model-dir",
                        dest="model_path",
                        help="Relative Directory Path where the .pt file is stored",
                        default=None, type=str)

    parser.add_argument("--from-sequence",
                        dest="sequence",
                        help="The sequence you want highlighted in the plot",
                        default=None, nargs='+', type=int)

    parser.add_argument("-d",
                        dest="description",
                        help="What type of sequence it is (Uptrend, etc)",
                        default=None, type=str)

    options = parser.parse_args()

    go(options)