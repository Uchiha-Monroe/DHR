import argparse

from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

from dataset_proc import PassageGeneratedQueryDataset


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="experiment_outputs/")
    parser.add_argument('--backbone_model_path', type=str, default="ptm/flan-t5-small")
    parser.add_argument('--num_epoch', type=int, default=100)




trn = PassageGeneratedQueryDataset()
trnloader = DataLoader(dataset=trn, batch_size=6)



val = PassageGeneratedQueryDataset(mode="valid")

print("done")