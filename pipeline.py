from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
import torch
import argparse
import numpy as np
from pprint import pprint

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model(args):
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        args.config_name
        if args.config_name
        else args.model_name_or_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name
        else args.model_name_or_path,
        use_fast=True,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    return config, tokenizer, model


# 해당 query에 관련된 문서
def get_relevant_doc(vectorizer, query, sp_matrix, k=1):
    """
    참고 : vocab에 없는 이상한 단어로 query하는 경우 assertion 발생 ex) 땔뙇?
    """
    query_vec = vectorizer.transform([query])
    assert np.sum(query_vec) != 0, "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
    # 모든 문서에 대한 inner product, score
    result = query_vec * sp_matrix.T
    sorted_result = np.argsort(-result.data)
    doc_scores = result.data[sorted_result]
    doc_ids = result.indices[sorted_result]
    return doc_scores[:k], doc_ids[:k]


def get_answer_from_context(context, question, model, tokenizer):
    # question, context token input
    encoded_dict = tokenizer.encode_plus(
        question,
        context,
        truncation=True,
        padding='max_length',
        max_length=512
    )
    non_padded_ids = encoded_dict["input_ids"][:encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
    full_text = tokenizer.decode(non_padded_ids)
    inputs = {
        'input_ids' : torch.tensor([encoded_dict['input_ids']], dtype=torch.long),
        'attention_mask' : torch.tensor([encoded_dict['attention_mask']], dtype=torch.long),
        'token_type_ids' : torch.tensor([encoded_dict['token_type_ids']], dtype=torch.long),
    }

    outputs = model(**inputs)
    start, end = torch.max(outputs.start_logits, axis=1).indices.item(), torch.max(outputs.end_logits, axis=1).indices.item()
    answer = tokenizer.decode(encoded_dict['input_ids'][start:end+1])
    return answer


def open_domain_qa(query, corpus, vectorizer, model, tokenizer, sp_matrix, k=1):
    # 1. Retrieve k relevant docs by using sparse matrix
    _, doc_id = get_relevant_doc(vectorizer, query, sp_matrix, k)
    context = corpus[doc_id.item()]

    # 2. Predict answer from given doc by using MRC model
    answer = get_answer_from_context(context, query, model, tokenizer)
    print("{} {} {}".format('*'*20, 'Result', '*'*20))
    print("[Search query]\n", query, "\n")
    print(f"[Relevant Doc ID(Top 1 passage)]: {doc_id.item()}")
    pprint(corpus[doc_id.item()], compact=True)
    print(f"[Answer Prediction from the model]: {answer}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', default = "/opt/ml/koelectra-korquad/output/checkpoint-7876", type=str, help='Path to pretrained model or model identifier from huggingface.co/models')
    parser.add_argument('--config_name', default = None, type=str, help='Pretrained config name or path if not the same as model_name')
    parser.add_argument('--tokenizer_name', default = None, type=str, help='Pretrained tokenizer name or path if not the same as model_name')

    args = parser.parse_args()

    return args

def main(args):
    dataset = load_dataset("squad_kor_v1")

    corpus = list(set(example['context'] for example in dataset['train']))
    corpus.extend(list(set(example['context'] for example in dataset['validation'])))
    tokenizer_func = lambda x : x.split(' ')

    # uni-gram + bi-gram
    vectorizer = TfidfVectorizer(tokenizer=tokenizer_func, ngram_range=(1,2))
    sp_matrix = vectorizer.fit_transform(corpus)

    _, tokenizer, model = load_model(args)

    model.eval()

    query = None
    while query != '그만':
        query = input("Enter any question: ")
        open_domain_qa(query, corpus, vectorizer, model, tokenizer, sp_matrix)

if __name__ == "__main__":
    args = parse_args()
    main(args)
