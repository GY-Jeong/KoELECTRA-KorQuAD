# import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm.auto import tqdm
import pandas as pd
import pickle
import json
import os
import numpy as np
import argparse
import os
from subprocess import Popen, PIPE, STDOUT
from scipy.special import softmax

from utils_qa import BM25

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
# from konlpy.tag import Mecab
# from rank_bm25 import BM25Okapi

import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')

class SparseRetrieval:
    def __init__(self, args, tokenize_fn, data_path="/opt/ml/input/data", context_path="wikipedia_documents.json"):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r") as f:
            wiki = json.load(f)

        self.tokenize_fn = tokenize_fn
        self.mode = args.embedding_mode
        self.contexts = list(dict.fromkeys([v['text'] for v in wiki.values()])) # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        if self.mode == 'tfidf':
            self.embedding_vector = TfidfVectorizer(
                tokenizer=tokenize_fn,
                ngram_range=(1, 2),
                # max_features=50000,
            )
        elif self.mode == 'bm25':
            self.embedding_vector = BM25(
                tokenizer=tokenize_fn,
                ngram_range=(1, 2),
                max_features=50000,
            )
        elif self.mode == 'bm25_new':
            tokenized_corpus = [self.tokenize_fn(doc) for doc in self.contexts]
            self.bm25 = BM25Okapi(tokenized_corpus)
        
        # should run get_sparse_embedding() or build_faiss() first.
        self.p_embedding = None
        self.indexer = None

    def get_sparse_embedding(self):
        # Pickle save.
        if self.mode == 'tfidf':
            embedding_bin_name = f"tfidv.bin"
            pickle_name = f"sparse_embedding_tfidv.bin"
        elif self.mode == 'bm25':
            embedding_bin_name = f"bm25.bin"
            pickle_name = f"sparse_embedding_bm25.bin"

        emd_path = os.path.join(self.data_path, pickle_name)
        embedding_bin_path = os.path.join(self.data_path, embedding_bin_name)
        if os.path.isfile(emd_path) and os.path.isfile(embedding_bin_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(embedding_bin_path, "rb") as file:
                self.embedding_vector = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.embedding_vector.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(embedding_bin_path, "wb") as file:
                pickle.dump(self.embedding_vector, file)
            print("Embedding pickle saved.")

    def build_faiss(self):
        # FAISS build
        num_clusters = 16
        niter = 5

        # 1. Clustering
        p_emb = self.p_embedding.toarray().astype(np.float32)
        emb_dim = p_emb.shape[-1]
        index_flat = faiss.IndexFlatL2(emb_dim)

        clus = faiss.Clustering(emb_dim, num_clusters)
        clus.verbose = True
        clus.niter = niter
        clus.train(p_emb, index_flat)

        centroids = faiss.vector_float_to_array(clus.centroids)
        centroids = centroids.reshape(num_clusters, emb_dim)

        quantizer = faiss.IndexFlatL2(emb_dim)
        quantizer.add(centroids)

        # 2. SQ8 + IVF indexer (IndexIVFScalarQuantizer)
        self.indexer = faiss.IndexIVFScalarQuantizer(quantizer, quantizer.d, quantizer.ntotal, faiss.METRIC_L2)
        self.indexer.train(p_emb)
        self.indexer.add(p_emb)

    def retrieve(self, query_or_dataset, what, topk=1):
        if self.mode != 'bm25_new' and self.mode != 'elastic':
            assert self.p_embedding is not None, "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()."
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            # make retrieved result as dataframe
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset['question'], k=1000, what=what)
                # print(doc_indices)
                if what == 'val':
                    for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                        # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]
                        # total_relev_doc_ids.append(relev_doc_ids)

                        # for i in range(topk):
                        #     tmp = {
                        #         "question": example["question"],
                        #         "id": example['id'] + '/' + str(i),
                        #         "context_id": doc_indices[idx][i],  # retrieved id
                        #         # "context_ids": doc_indices[idx],  # retrieved id
                        #         "context": self.contexts[doc_indices[idx][i]],   # retrieved doument
                        #         # "context": self.contexts[doc_indices[idx][0]],  # retrieved doument
                        #         # "contexts": ' '.join([self.contexts[doc_indices[idx][i]] for i in range(topk)])   # retrieved doument
                        #     }
                        #     if 'context' in example.keys() and 'answers' in example.keys():
                        #         tmp["original_context"] = example['context']  # original document
                        #         tmp["answers"] = example['answers']           # original answer
                        #     total.append(tmp)
                        candidate_contexts = [self.contexts[doc_indices[idx][i]] for i in range(100)]
                        # print(len(candidate_contexts))

                        tokenized_corpus_candidate = \
                            [self.tokenize_fn(doc) for doc in candidate_contexts]
                        bm25_candidate = BM25Okapi(tokenized_corpus_candidate)
                        candidate = bm25_candidate.get_top_n(self.tokenize_fn(example["question"]), candidate_contexts, n=topk)

                        tmp = {
                            "question": example["question"],
                            "id": example['id'],
                            "context_id": doc_indices[idx][0],  # retrieved id
                            # "context_ids": doc_indices[idx],  # retrieved id
                            # "context": ' '.join([self.contexts[doc_indices[idx][i]] for i in range(topk)]),   # retrieved doument
                            # "context": self.contexts[doc_indices[idx][0]],  # retrieved doument
                            "context": ' '.join(candidate)   # retrieved doument
                        }
                        if 'context' in example.keys() and 'answers' in example.keys():
                            tmp["original_context"] = example['context']  # original document
                            # example['answers']['answer_start'] += (len(example['title']) + 1)
                            tmp["answers"] = example['answers']           # original answer
                        total.append(tmp)

                else:
                    for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                        candidate = []
                        for i in range(topk):
                            if self.contexts[doc_indices[idx][i]] != example['context']:
                                candidate.append(self.contexts[doc_indices[idx][i]])
                            
                            if len(candidate) == 1:
                                break

                        tmp = {
                            "question": example["question"],
                            "id": example['id'],
                            "context_id": doc_indices[idx][0],  # retrieved id
                            # "context_ids": doc_indices[idx],  # retrieved id
                            # "context": ' '.join([self.contexts[doc_indices[idx][i]] for i in range(topk)]),   # retrieved doument
                            "context": example['context'] + ' ' + ' '.join(candidate),   # retrieved doument
                            # "context": self.contexts[doc_indices[idx][0]],  # retrieved doument
                            # "contexts": ' '.join([self.contexts[doc_indices[idx][i]] for i in range(topk)])   # retrieved doument
                        }
                        if 'context' in example.keys() and 'answers' in example.keys():
                            tmp["original_context"] = example['context']  # original document
                            # example['answers']['answer_start'] += (len(example['title']) + 1)
                            tmp["answers"] = example['answers']           # original answer
                        total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas


    def retrieve_elastic(self, query_or_dataset, what, es, topk=1):
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            # make retrieved result as dataframe
            total = []
            with timer("query exhaustive search"):
                if what == 'val':
                    for idx, example in enumerate(tqdm(query_or_dataset, desc="Elastic : ")):
                        # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]
                        # total_relev_doc_ids.append(relev_doc_ids)
                        
                        query = {
                            'query':{
                                'bool':{
                                    'must':[
                                            {'match':{'text':example["question"]}}
                                    ],
                                    'should':[
                                            {'match':{'text':example["question"]}}
                                    ]
                                }
                            }
                        }

                        doc = es.search(index='document_v3',body=query,size=topk)['hits']['hits']
                        topk_contexts = []
                        topk_contexts_score = []
                        for i in range(topk):
                            topk_contexts.append(doc[i]['_source']['text'])
                            topk_contexts_score.append(doc[i]['_score'])
                        
                        topk_contexts_score = softmax(topk_contexts_score)

                        tmp = {
                            "question": example["question"],
                            "id": example['id'],
                            # "context_id": doc_indices[idx][i],  # retrieved id
                            "context": '[SEP]'.join(topk_contexts),   # retrieved doument
                            "score": '[SEP]'.join(map(str, topk_contexts_score.tolist())),   # retrieved doument
                        }
                        if 'context' in example.keys() and 'answers' in example.keys():
                            tmp["original_context"] = example['context']  # original document
                            tmp["answers"] = example['answers']           # original answer
                        total.append(tmp)

                else:
                    for idx, example in enumerate(tqdm(query_or_dataset, desc="Elastic : ")):
                        candidate = []
                        for i in range(topk):
                            if self.contexts[doc_indices[idx][i]] != example['context']:
                                candidate.append(self.contexts[doc_indices[idx][i]])
                            
                            if len(candidate) == 1:
                                break

                        tmp = {
                            "question": example["question"],
                            "id": example['id'],
                            "context_id": doc_indices[idx][0],  # retrieved id
                            # "context_ids": doc_indices[idx],  # retrieved id
                            # "context": ' '.join([self.contexts[doc_indices[idx][i]] for i in range(topk)]),   # retrieved doument
                            "context": example['context'] + ' ' + ' '.join(candidate),   # retrieved doument
                            # "context": self.contexts[doc_indices[idx][0]],  # retrieved doument
                            # "contexts": ' '.join([self.contexts[doc_indices[idx][i]] for i in range(topk)])   # retrieved doument
                        }
                        if 'context' in example.keys() and 'answers' in example.keys():
                            tmp["original_context"] = example['context']  # original document
                            # example['answers']['answer_start'] += (len(example['title']) + 1)
                            tmp["answers"] = example['answers']           # original answer
                        total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas


    def get_relevant_doc(self, query, k=1):
        """
        참고: vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        if self.mode == 'bm25_new':
            result = self.bm25.get_scores(self.tokenize_fn(query))
            sorted_result = np.argsort(result.squeeze())[::-1]
            return result.squeeze()[sorted_result].tolist()[:k], sorted_result.tolist()[:k]

        with timer("transform"):
            query_vec = self.embedding_vector.transform([query])
        assert (
                np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        sorted_result = np.argsort(result.squeeze())[::-1]
        return result.squeeze()[sorted_result].tolist()[:k], sorted_result.tolist()[:k]

    def get_relevant_doc_bulk(self, queries, what, k=1):
        if self.mode == 'bm25_new':
            result = []
            if what == 'train_train':
                print("*****start bm25 embedding*****")
                # for query in tqdm(queries):
                #     result.append(self.bm25.get_scores(self.tokenize_fn(query)))

                # np.save("/opt/ml/input/data/bm25_embedding_train_train.npy", np.array(result))
                # exit()
                result = np.load("/opt/ml/input/data/bm25_embedding_train_train.npy")
                print(result.shape)
            elif what == 'train_val':
                print("*****start bm25 embedding*****")
                # for query in tqdm(queries):
                #     result.append(self.bm25.get_scores(self.tokenize_fn(query)))

                # np.save("/opt/ml/input/data/bm25_embedding_train_val.npy", np.array(result))
                # exit()
                result = np.load("/opt/ml/input/data/bm25_embedding_train_val.npy")
                print(result.shape)
            else:
                print("*****start bm25 embedding*****")
                # for query in tqdm(queries):
                #     result.append(self.bm25.get_scores(self.tokenize_fn(query)))

                # np.save("/opt/ml/input/data/bm25_embedding.npy", np.array(result))
                # exit()
                result = np.load("/opt/ml/input/data/bm25_embedding.npy")
                print(result.shape)

            # exit()
            doc_scores = []
            doc_indices = []
            print()
            for i in range(result.shape[0]):
                sorted_result = np.argsort(result[i, :])[::-1]
                doc_scores.append(result[i, :][sorted_result].tolist()[:k])
                doc_indices.append(sorted_result.tolist()[:k])
            return np.array(doc_scores), np.array(doc_indices)

        query_vec = self.embedding_vector.transform(queries)
        assert (
                np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def retrieve_faiss(self, query_or_dataset, topk=1):
        assert self.indexer is not None, "You must build faiss by self.build_faiss() before you run self.retrieve_faiss()."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            queries = query_or_dataset['question']
            # make retrieved result as dataframe
            total = []
            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(queries, k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]
                tmp = {
                    "question": example["question"],
                    "id": example['id'],  # original id
                    "context_id": doc_indices[idx][0],  # retrieved id
                    "context": self.contexts[doc_indices[idx][0]]  # retrieved doument
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"] = example['context']  # original document
                    tmp["answers"] = example['answers']           # original answer
                total.append(tmp)

 
    def get_relevant_doc_faiss(self, query, k=1):
        """
        참고: vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        query_vec = self.embedding_vector.transform([query])
        assert (
                np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)
        return D.tolist()[0], I.tolist()[0]


    def get_relevant_doc_bulk_faiss(self, queries, k=1):
        query_vecs = self.embedding_vector.transform(queries)

        assert (
                np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)
        return D.tolist(), I.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedding_mode', type=str, default='bm25')
    parser.add_argument('--topk', type=int, default=1)

    args = parser.parse_args()
    # Test sparse
    org_dataset = load_from_disk("/opt/ml/input/data/train_dataset")
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    ) # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*"*40, "query dataset", "*"*40)
    print(full_ds)

    ### Mecab 이 가장 높은 성능을 보였기에 mecab 으로 선택 했습니다 ###
    mecab = Mecab()
    def tokenize(text):
        # return text.split(" ")
        return mecab.morphs(text)

    # from transformers import AutoTokenizer
    #
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "bert-base-multilingual-cased",
    #     use_fast=True,
    # )
    ###############################################################

    wiki_path = "wikipedia_documents_pp.json"
    retriever = SparseRetrieval(
        # tokenize_fn=tokenizer.tokenize,
        tokenize_fn=tokenize,
        data_path="/opt/ml/input/data",
        context_path=wiki_path,
        args = args)

    # test single query
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.embedding_mode != 'bm25_new':
        retriever.get_sparse_embedding()
    with timer("single query by exhaustive search"):
        scores, indices = retriever.retrieve(query, topk=args.topk)
    
    # retriever.build_faiss()
    # with timer("single query by faiss"):
    #     scores, indices = retriever.retrieve_faiss(query)

    # test bulk
    with timer("bulk query by exhaustive search"):
        df = retriever.retrieve(full_ds, topk=args.topk)
        # df.to_csv('/opt/ml/code/df.csv')
        # df = pd.read_csv('/opt/ml/code/df.csv')
        df['correct'] = df['original_context'] == df['context']
        print("correct retrieval result by exhaustive search top1", df['correct'].sum() / len(df))

        correct = 0
        for i in range(len(df['original_context'])):
            correct += int(df['original_context'][i] in df['contexts'][i])
        print(f"correct retrieval result by exhaustive search top{args.topk}", correct / len(df))

    # with timer("bulk query by exhaustive search"):
    #     df = retriever.retrieve_faiss(full_ds)
    #     df['correct'] = df['original_context'] == df['context']
    #     print("correct retrieval result by faiss", df['correct'].sum() / len(df))