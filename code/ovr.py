import csv
import numpy as np
from pathlib import Path


class OVR:
    def __init__(self, file_y, dataset, path):
        self.file_y = file_y
        self.dataset = dataset
        self.path = path


    def one_vs_rest(self, labels):
        classes = np.unique(labels)
        for class_ in classes:
            mask = np.zeros(labels.shape[0], dtype=np.int32)
            mask[np.where(labels == class_)] = 1
            yield class_, mask


    def labels_reader(self):
        rows = []
        with open(Path(self.path, self.dataset, self.file_y), 'r') as file:
            reader = csv.reader(file)
            _ = reader.__next__()

            for row in reader:
                rows.append(row)
            
            return np.array(rows, dtype=np.int32)[:, -1]
        

    def run(self):
        labels = self.labels_reader()
        ovr = self.one_vs_rest(labels)

        for _ in range(np.unique(labels).shape[0]):
            class_, mask = ovr.__next__()
            file_y = Path(self.path, self.dataset, f"{class_.strip().replace('/', '')}_VR_Labels.csv")
            with open(file_y, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(mask)