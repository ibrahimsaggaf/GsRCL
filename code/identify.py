import os
import csv
import joblib
import warnings
import requests
import zipfile
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from networks import Encoder


class Identify:
    def __init__(self, reference, path):
        self.reference = reference
        self.path = path
        self.zenodo_id = '10050548'
        self.zenodo_file = f'{self.reference}.zip'
        self.results_ = {}


    def _download_pretrained_encoders(self):
        '''
        Following the guidelines at https://developers.zenodo.org/
        '''
        print('Downloading pretrained encoders ...')
        url = f'https://zenodo.org/api/records/{self.zenodo_id}/files/{self.zenodo_file}/content'
        r = requests.get(url, stream=True)

        assert r.status_code == 200, r.reason

        with open(Path(self.path, self.zenodo_file), 'wb') as file:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        print('Unzipping ...')
        with zipfile.ZipFile(Path(self.path, self.zenodo_file), 'r') as zip_file:
            zip_file.extractall(self.path)

            os.remove(Path(self.path, self.zenodo_file))


    def _preprocess(self, values, epsilon=1e-6):
        values = torch.tensor(values, dtype=torch.float32)
        values = torch.log(values + 1.0 + epsilon)

        return values


    def _load_encoder(self, checkpoint):
        device = torch.device('cpu')
        params = torch.load(Path(self.path, self.reference, checkpoint), map_location=device)
        encoder = Encoder(**params['frozen_kwargs'])
        encoder.load_state_dict(params['forzen_params'])

        return encoder


    def _csv_reader(self, query=None, genes=True):
        rows = []
        if query is None:
            path = Path(self.path, self.reference, f'{self.reference}-reference-genes.csv')

        else:
            path = Path(self.path, query)

        with open(path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                rows.append(row)

        if genes:
            return np.array(rows).flatten()

        else:
            return np.array(rows, dtype=np.float32)


    def _check_genes(self, reference, query, values):
        query = query.flatten()
        assert np.unique(query).shape[0] == query.shape[0], 'The set of query genes has duplicates.'

        if len(values.shape) == 1:
            values = values.reshape(1, -1)

        print(f'Cross referencing {query.shape[0]} query genes against {reference.shape[0]} reference genes ...')
        order = {g: i for i, g in enumerate(reference)}
        drop = np.in1d(query, reference)

        assert np.logical_not(drop).sum() / query.shape[0] < 0.9, f'{(np.logical_not(drop).sum() / query.shape[0]) * 100:.2f}% of query genes are not found in reference genes. Try a different reference.'

        if np.logical_not(drop).sum() / query.shape[0] > 0.5:
            warnings.warn(f'{(np.logical_not(drop).sum() / query.shape[0]) * 100:.2f}% of query genes are not found in reference genes, this will affect the output reliability.')

        print(f'{np.logical_not(drop).sum()} out of {query.shape[0]} genes are not found in reference, hence {np.logical_not(drop).sum()} genes removed from query.')
        query = query[drop]
        values = values[:, drop]

        missing = np.setdiff1d(reference, query)
        query = np.hstack((missing, query))
        values = np.hstack((np.zeros(shape=(values.shape[0], missing.shape[0])), values))
        print(f'{missing.shape[0]} out of {reference.shape[0]} genes in reference are not found in query, hence {missing.shape[0]} genes added to query with zero values.')

        assert query.shape[0] == reference.shape[0], f'The number of genes after cross referencing is {query.shape[0]} whereas it should be {reference.shape[0]}.'

        order2 = [order.get(g) for g in query]
        order2 = np.array([i for i, _ in sorted(zip(range(query.shape[0]), order2), key=lambda g:g[1])])

        return query[order2], values[:, order2]


    def _get_probs(self, cell_type, checkpoint, values):
        if checkpoint.endswith('pt'):
            clf = joblib.load(Path(self.path, self.reference, f'{cell_type}--svm.joblib'))
            encoder = self._load_encoder(checkpoint)
            encoder.eval()

            with torch.no_grad():
                h = encoder(values)

            self.results_[cell_type] = clf.predict_proba(h.detach().numpy())[:, 1]

        else:
            clf = joblib.load(Path(self.path, self.reference, checkpoint))
            self.results_[cell_type] = clf.predict_proba(values)[:, 1]


    def save_results(self):
        df = pd.DataFrame(self.results_)
        sum = df.sum(axis=1)
        df = df.div(sum, axis=0)
        preds = df.values.argmax(axis=1)
        df['Identified as'] = [df.columns[p] for p in preds]
        df.to_csv(Path(self.path, 'probabilities.csv'), index=False)


    def predict_proba(self, query, values):
        if not os.path.isdir(Path(self.path, self.reference)):
            self._download_pretrained_encoders()

        reference = self._csv_reader()
        query = self._csv_reader(query=query)
        values = self._csv_reader(query=values, genes=False)

        query, values = self._check_genes(reference, query, values)
        values = self._preprocess(values)

        print('Obtaining probabilities for each cell-type ...')
        for checkpoint in os.listdir(Path(self.path, self.reference)):
            cell_type = checkpoint.split('--')[0]
            if cell_type in self.results_ or checkpoint.endswith('csv'):
                continue

            self._get_probs(cell_type, checkpoint, values)

