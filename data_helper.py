# Author: Ao Zou

import os

os.environ["OMP_NUM_THREADS"] = '4'
import pickle
import argparse
from tqdm import tqdm, trange

import json
import jsonlines

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, MiniBatchKMeans

import torch
import torch.nn as nn

from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def text2emb_by_sentence_transformer(
        text: str,
        model
) -> np.array:
    return model.encode(text)


def text2emb_by_hftransformer(texts: str,
                              model: nn.Module,
                              tokenizer,
                              pooling: str = 'mean') -> torch.Tensor:
    """Convert text to embeddings, using the specified model

    :texts: text needs to be embedded
    :model: model used to embed texts, from Transformers lib by default
    :tokenizer: tokenizer used by model, usually from Transformers lib
    :pooling: pooling strategy
    """
    if not pooling in ["cls", "mean", "max", "last2mean"]:
        print("wrong pooling strategy is provided, using mean pooling strategy by default.")
        pooling = "mean"

    if pooling == "mean":
        inputs = tokenizer(texts, return_tensors="pt")
        outputs = model(**inputs)
        return outputs.last_hidden_state.detach().cpu().numpy()


def prepro_nq(save_to_path: bool = False):
    """Collect unique passages from the Natural Question dataset with
        'question-answer-context' format.

    the generated data is all_passage_id_pairs:
    
    List[
      {"passage_id": ...,
       "text": ...},
      {"passage_id": ...,
       "text": ...},
       ...
    ]
    
    """
    trn_path = "data_concerned/datasets/nq/nq-train.json"
    dev_path = "data_concerned/datasets/nq/nq-dev.json"

    all_passage_id = []
    all_passage_id_pairs = []

    def _pro_fn(raw_data):
        for data in tqdm(raw_data):
            for pos_data in data["positive_ctxs"]:
                if pos_data["passage_id"] not in set(all_passage_id):
                    all_passage_id.append(pos_data["passage_id"])
                    cur_data_pair = {"passage_id": pos_data["passage_id"], "text": pos_data["text"]}
                    all_passage_id_pairs.append(cur_data_pair)

    print('start preprocessing dev set...')
    with open(dev_path, "r") as fd:
        dev_data = json.load(fd)
    _pro_fn(dev_data)

    print('start preprocessing train set...')
    with open(trn_path, 'r') as ft:
        trn_data = json.load(ft)
    _pro_fn(trn_data)

    print(f'done. totally {len(all_passage_id_pairs)} unique passages.')

    if save_to_path:
        with open("data_concerned/datasets/nq/all_passage_id_pairs.json", "w") as f:
            json.dump(all_passage_id_pairs, f, indent=4)


def passage_embedder(
        data_path,
        embed_fn,
        model_path,
        model_name,
        save_dir='data_concerned/datasets/nq/',
        runtime_whitening=False
):
    print('reading data...')
    with open(data_path, 'r') as f:
        passages_ids_pair = json.load(f)

    all_psg_emb = []
    model = SentenceTransformer(model_path).to('cuda')

    print('start converting passages to embeddings...')
    for ind, psg_id in tqdm(enumerate(passages_ids_pair)):
        emb = embed_fn(psg_id["text"], model)
        all_psg_emb.append(emb)

        # compute mu and sigma at each step iteratively
        if runtime_whitening:
            n = len(passages_ids_pair)
            if ind == 0:
                mu = emb
                sigma = np.zeros((emb.shape, emb.shape))
            else:
                mu = (n / (n + 1)) * mu + (1 / (n + 1)) * emb
                sigma = (n / (n + 1)) * (sigma + mu.T @ mu) + (1 / (n + 1)) * (emb.T @ emb) - (mu.T @ mu)

    all_psg_emb_np = np.stack(all_psg_emb, axis=0)

    if runtime_whitening:
        all_psg_emb_np = (all_psg_emb_np + mu) @ sigma

    save_path = save_dir + 'passage_embeddings_trn_' + model_name + '.npy'
    with open(save_path, 'wb') as f:
        np.save(f, all_psg_emb_np)


def compute_kernel_bias_from_allemb(embs: np.ndarray):
    """conpute kernel and bias of whitening sentence representation
    
    :embs [num_samples, embedding_size]: the embedding representations of all passages
    
    return:
      W: kernel
      -mu: bias

      y = (x + bias).dot(kernel)
    """
    print("start SVD decomposing...")
    mu = embs.mean(axis=0, keepdims=True)
    cov = np.cov(embs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))

    return W, -mu


def matrix_whitener(matrix: np.ndarray):
    """convert a matrix to the whitened version.

    :matrix [num_samples, embedding_size]:
    """
    kernel, bias = compute_kernel_bias_from_allemb(matrix)
    return (matrix + bias) @ kernel


def extract_old_ids(
        passage_id_file_path="data_concerned/datasets/nq/all_passage_id_pairs.json",
        save2disk=False,
        trn_test_ratio=0.9
):
    """This function extracts the old passage_id with the sequence of
       *passage_id_pairs.json
    
    """
    with open(passage_id_file_path, 'r') as f:
        psg_id_data = json.load(f)

    all_old_ids = []
    for psg_id in tqdm(psg_id_data):
        all_old_ids.append(psg_id["passage_id"])

    split_divd = int(len(all_old_ids) * trn_test_ratio)
    old_ids_list_trn = all_old_ids[: split_divd]
    old_ids_list_test = all_old_ids[split_divd:]

    if save2disk:
        with open("data_concerned/datasets/nq/old_ids_list.pkl", "wb") as f:
            pickle.dump(all_old_ids, f)
        with open("data_concerned/datasets/nq/old_ids_list_trn.pkl", "wb") as f:
            pickle.dump(old_ids_list_trn, f)
        with open("data_concerned/datasets/nq/old_ids_list_test.pkl", "wb") as f:
            pickle.dump(old_ids_list_test, f)


def divide_psg_id_pairs(
        passage_id_file_path="data_concerned/datasets/nq/all_passage_id_pairs.json",
        save_dir="data_concerned/datasets/nq/",
        trn_test_ratio=0.9
):
    with open(passage_id_file_path, 'r') as f:
        psg_id_data = json.load(f)

    divd = int(len(psg_id_data) * trn_test_ratio)
    psg_id_data_trn = psg_id_data[: divd]
    psg_id_data_test = psg_id_data[divd:]

    with open(save_dir + "passage_id_pairs_train.json", "w") as f:
        json.dump(psg_id_data_trn, f)
    with open(save_dir + "passage_id_pairs_test.json", "w") as f:
        json.dump(psg_id_data_test, f)


def generate_query_from_doc(
        model,
        tokenizer,
        doc_text,
        input_max_length=300,
        output_max_length=50,
        top_p=0.95,
        num_generated_query=5,
        device=DEVICE
):
    """Generate a few querys according to the input document
    """
    input_ids = tokenizer.encode(doc_text,
                                 max_length=input_max_length,
                                 truncation=True,
                                 return_tensors="pt").to(device)
    outputs = model.generate(input_ids=input_ids,
                             max_length=output_max_length,
                             do_sample=True,
                             top_p=top_p,
                             num_return_sequences=num_generated_query)
    generated_querys = []
    for i in range(outputs.shape[0]):
        generated_querys.append(tokenizer.decode(outputs[i], skip_special_tokens=True))
    return generated_querys


def generate_querys_from_stream(
        passage_file_path="data_concerned/datasets/nq/passage_id_pairs_train.json",
        gq_model_path="ptm/doc2query-msmarco-t5-small-v1",
        device=DEVICE
):
    d2q_model = T5ForConditionalGeneration.from_pretrained(gq_model_path).to(device)
    d2q_tokenizer = T5Tokenizer.from_pretrained(gq_model_path)

    psg_id_gq_triple_list = []
    with open(passage_file_path, "r") as f_gq:
        psg_id_pairs_list = json.load(f_gq)

        for data in tqdm(psg_id_pairs_list[:300]):
            new_data = {}
            new_data["passage_id"] = data["passage_id"]
            new_data["text"] = data["text"]
            new_data["generated_querys"] = generate_query_from_doc(
                model=d2q_model,
                tokenizer=d2q_tokenizer,
                doc_text=data["text"]
            )
            psg_id_gq_triple_list.append(new_data)
        with open("data_concerned/datasets/nq/passage_id_gq_for_train.json", "w") as fw:
            json.dump(psg_id_gq_triple_list, fw, indent=4)


def convert_semanticID_to_text(hie_id):
    """Convert the generated hierarchical semantic id to text for training
       eg. [2, 5, 9, 5, 52] --> "2 5 9 5 52"
    
    Arguments:
      :hie_id - List[int]: the hierarchical semantic id for a single passage
    return:
      :str_id - str: the id with str format 
    """
    str_id_cand = []
    for sem_i in hie_id:
        str_id_cand.append(str(sem_i))
    return " ".join(str_id_cand)



if __name__ == "__main__":

    ###### test zone ######
    # divide_psg_id_pairs()
    # extract_old_ids(save2disk=True)
    # print('done')
    ###### test zone ###### 

    # step 1. preprocess the question-answer-context data into unique passages and save to disk
    # prepro_nq(True)

    # step 2. convert the passage from text to embeding
    # passages_path = 'data_concerned/datasets/nq/passage_id_pairs_train.json'
    # embedder_path = 'ptm/sentence-t5-base'
    # passage_embedder(
    #     data_path=passages_path,
    #     embed_fn=text2emb_by_sentence_transformer,
    #     model_path=embedder_path,
    #     model_name='sentence-t5-base',
    # )

    # step 3. compute semantic sturcture id for each psg
    # load passage_embedding npy file
    # psg_embedding_path = 'data_concerned/datasets/nq/passage_embeddings_trn_sentence-t5-base.npy'
    # with open(psg_embedding_path, 'rb') as f:
    #     X = np.load(f)

    # k = 10
    # c = 100
    # emb_dim = 768
    # cluster_bsz = int(1e3)

    # kmeans = KMeans(
    #     n_clusters=k,
    #     max_iter=500,
    #     n_init=100,
    #     init='k-means++',
    #     tol=1e-6
    # )

    # mini_kmeans = MiniBatchKMeans(
    #     n_clusters=k,
    #     max_iter=300,
    #     n_init=100,
    #     init='k-means++',
    #     batch_size=cluster_bsz,
    #     reassignment_ratio=0.01,
    #     max_no_improvement=50,
    #     tol=1e-7
    # )

    # semantic_id_list = []


    # def classify_recursion(x_data_pos):
    #     if x_data_pos.shape[0] <= c:
    #         if x_data_pos.shape[0] == 1:
    #             return
    #         for idx, pos in enumerate(x_data_pos):
    #             semantic_id_list[pos].append(idx)
    #         return

    #     temp_data = np.zeros((x_data_pos.shape[0], emb_dim))
    #     for idx, pos in enumerate(x_data_pos):
    #         temp_data[idx, :] = X[pos]

    #     if x_data_pos.shape[0] >= cluster_bsz:
    #         pred = mini_kmeans.fit_predict(temp_data)
    #     else:
    #         pred = kmeans.fit_predict(temp_data)

    #     for i in range(k):
    #         pos_lists = []
    #         for id_, class_ in enumerate(pred):
    #             if class_ == i:
    #                 pos_lists.append(x_data_pos[id_])
    #                 semantic_id_list[x_data_pos[id_]].append(i)
    #         classify_recursion(np.array(pos_lists))

    #     return


    # print('Start First Clustering')
    # pred = kmeans.fit_predict(X)
    # print(pred.shape)  # int 0-9 for each vector
    # print(kmeans.n_iter_)

    # for class_ in pred:
    #     semantic_id_list.append([class_])

    # print('Start Recursively Clustering...')
    # for i in range(k):
    #     print(i, "th cluster")
    #     pos_lists = []
    #     for id_, class_ in enumerate(pred):
    #         if class_ == i:
    #             pos_lists.append(id_)
    #     classify_recursion(np.array(pos_lists))

    # id_mapper = {}
    # # load old id
    # with open("data_concerned/datasets/nq/old_ids_list_trn.pkl", "rb") as f:
    #     old_id_list = pickle.load(f)
    # for i in range(len(old_id_list)):
    #     id_mapper[old_id_list[i]] = semantic_id_list[i]
    # # save to disk
    # with open(f"data_concerned/datasets/nq/ID_MAPPER_trn_st5-base_k{k}_c{c}.pkl", "wb") as f:
    #     pickle.dump(id_mapper, f)





    # step 4. Generate querys from documents
    generate_querys_from_stream()

    print('done')
