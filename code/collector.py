from dataclasses import dataclass


@dataclass
class ResultsCollector:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


res_collector = ResultsCollector()