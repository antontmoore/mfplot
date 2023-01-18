from enum import Enum


class DatasetPart(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'