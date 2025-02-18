import sys
import os
import joblib
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from afrcl_networks import Encoder


def load_files(args):
    mat = np.load(Path(args.output, 'mat.npy'))
    obs = np.load(Path(args.output, 'obs.npy'))
    var = np.load(Path(args.output, 'var.npy'))
    match = np.load(Path(args.output, 'match.npy'))
    cell_types = joblib.load(Path(args.model_path, args.reference, f'{args.reference}_cell_type_dict.joblib'))

    return mat, obs, var, match, cell_types


def cross_reference_genes(args, query_genes, mat, match):
    ref_genes = pd.read_csv(Path(args.model_path, args.reference, f'{args.reference}-reference-genes.csv'), header=None, index_col=False)
    ref_genes = ref_genes.values.flatten()
    order = {g: i for i, g in enumerate(ref_genes)}

    query_genes = query_genes.flatten()
    query_genes = query_genes[match]
    mat = mat[:, match] if args.transpose == 0 else mat[match, :]

    missing = np.setdiff1d(ref_genes, query_genes)
    query_genes = np.hstack((missing, query_genes))
    if args.transpose == 0:
        mat = np.hstack((np.zeros(shape=(mat.shape[0], missing.shape[0])), mat))
    else:
        mat = np.vstack((np.zeros(shape=(missing.shape[0], mat.shape[1])), mat))

    order2 = np.array([order.get(g) for g in query_genes])
    order2 = np.array([i for i, _ in sorted(zip(range(query_genes.shape[0]), order2), key=lambda g:g[1])])
    mat = mat[:, order2] if args.transpose == 0 else mat[order2, :]

    return mat


def preprocess(args, mat, epsilon=1e-6):
    mat = mat.T if args.transpose == 1 else mat
    mat = torch.tensor(mat, dtype=torch.float32)

    if args.log == 1:
        mat = torch.log(mat + 1.0 + epsilon)

    return mat


def load_encoder(args):
    enc_kwargs = {
            'enc_in_dim': args.dim, 
            'enc_dim': 1024, 
            'enc_out_dim': 512, 
            'proj_dim': 256, 
            'proj_out_dim': 128
        }
    device = torch.device('cpu')
    params = torch.load(Path(args.model_path, args.reference, f'{args.reference}_pretrained_encoder.pt'), map_location=device)
    encoder = Encoder(**enc_kwargs)
    encoder.load_state_dict(params)
    encoder.eval()

    return encoder


def get_probs(args, mat):
    clf = joblib.load(Path(args.model_path, args.reference, f'{args.reference}_svm.joblib'))
    encoder = load_encoder(args)

    with torch.no_grad():
        h, _ = encoder(mat)

    probs =  clf.predict_proba(h.detach().numpy())

    return probs


def format_results(args, probs, barcodes, cell_types):
    df = pd.DataFrame(probs, columns=cell_types)
    sum = df.sum(axis=1)
    df = df.div(sum, axis=0)
    preds = df.values.argmax(axis=1)
    mask = df.values.max(axis=1) <= args.p_value

    df['Identified as'] = [df.columns[p] for p in preds]
    df['Identified as'] = df['Identified as'].where(~mask, other='Unassigned')
    df.index = barcodes

    return df


def plot(args, mat, results):
    palette = np.array([
        '#ff0000', '#ff00ff', '#00ff00', '#ffff00', '#0000ff', '#00ffff', '#c0c0c0',
        '#800000', '#800080', '#008000', '#808000', '#000080', '#008080', '#808080'
    ])
    cell_types = results.columns.values[:-1]
    palette_ = palette[:cell_types.shape[0]]
    preds = results['Identified as'].values
    if 'Unassigned' in preds:
        cell_types = np.append(cell_types, 'Unassigned')
        palette_ = np.append(palette_, '#faebd7')

    idxs = np.hstack([np.where(cell_types == c)[0] for c in preds])
    emb = TSNE().fit_transform(mat.detach().numpy())
    plt.scatter(emb[:, 0], emb[:, 1], c=palette_[idxs], edgecolor='black')
    plt.xlabel('t-SNE 1', fontsize=16)
    plt.ylabel('t-SNE 2', fontsize=16)
    handles = []
    with open(Path(args.output, 'legend.txt'), 'w') as f:
        for cell, colour in zip(cell_types, palette_):
            handles.append(Rectangle(xy=(0, 0), width=1, height=1, edgecolor='black', facecolor=colour))
            f.write(cell + '\t' + colour + '\n')

    plt.legend(handles, cell_types, fontsize=12, handlelength=4, handleheight=3, bbox_to_anchor=(1.0, 1.02))
    plt.savefig(Path(args.output, 'tsne.svg'), dpi=600, bbox_inches='tight')
    plt.close()


def main(args):
    try:
        results = {}
        mat, obs, var, match, cell_types = load_files(args)
        query_genes = var if args.transpose == 0 else obs
        barcodes = obs if args.transpose == 0 else var
        mat = cross_reference_genes(args, query_genes, mat, match)
        mat = preprocess(args, mat)
        args.dim = mat.shape[1]

        probs = get_probs(args, mat)
        results = format_results(args, probs, barcodes, cell_types)
        results.to_csv(Path(args.output, 'probabilities.csv'))
        plot(args, mat, results)

    except:
        print(
            f'Error 140: Server error'
        )
        sys.exit(140)   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', default='.', help='the full path to the output directory')
    parser.add_argument('-r', '--reference', help='the name of the reference dataset')
    parser.add_argument('-t', '--transpose', type=int, help='transpose the input matrix if rows are genes and columns are cells')
    parser.add_argument('-mp', '--model_path', help='the full path to the reference models and the set of reference genes')
    parser.add_argument('-p', '--p_value', type=float, default=0.5, help='select a p-value cut-off for putative new cell types')
    parser.add_argument('--log', type=int, help='log transform the input matrix if it contains raw counts')

    args = parser.parse_args()
    main(args)