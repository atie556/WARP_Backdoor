import logging
import pathlib, os
import json
import torch
import sys
import transformers
import gc
import tracemalloc
import traceback
from beir import util, LoggingHandler
from beir.retrieval import models
# from beir.datasets.data_loader import GenericDataLoader # No longer needed
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.models import DPR
from transformers import RobertaTokenizerFast

from src.contriever_src.contriever import Contriever
from src.contriever_src.beir_utils import DenseEncoderModel
from src.utils import load_json
import argparse
import time   # ✅ 新增：导入 time 模块
from tqdm import tqdm  # ✅ 新增：导入 tqdm

# Setup argument parser
parser = argparse.ArgumentParser(description='test')

parser.add_argument('--model_code', type=str, default="contriever-msmarco")
parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
parser.add_argument('--top_k', type=int, default=100)
parser.add_argument('--dataset', type=str, default="msmarco_immigration", help='BEIR dataset to evaluate')
parser.add_argument('--split', type=str, default='test') # This might not be directly used now for loading
parser.add_argument('--result_output', default="results/beir_results/msmarco_immigration-contriever_msmarco.json", type=str)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument("--per_gpu_batch_size", default=256, type=int, help="Batch size per GPU/CPU for indexing.")
parser.add_argument('--max_length', type=int, default=128)
args = parser.parse_args()
from src.utils import model_code_to_cmodel_name, model_code_to_qmodel_name

# Logging setup
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logging.info("Starting the retrieval process with arguments: %s", args)

# ✅ 新增：记录开始时间
start_time = time.time()

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Memory tracking
tracemalloc.start()
logging.info("Initial Memory Usage: %s", tracemalloc.get_traced_memory())

#### Load dataset paths
dataset = args.dataset

# 修改后的 corpus 和 queries 路径结构
queries_file = f'datasets/{dataset}/queries.jsonl'
corpus_file = f'datasets/{dataset}/corpus.jsonl'

logging.info("Loading corpus from: %s", corpus_file)
logging.info("Loading queries from: %s", queries_file)
# Load corpus directly from JSONL and transform to list of dictionaries
corpus_dict = {}

try:
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                doc = json.loads(line)
                title = doc.get('title', '').strip()  # 读取 title
                text = doc.get('text', '').strip()  # 读取 text
                full_text = f"{title} {text}".strip()  # 拼接 title 和 text
                corpus_dict[doc['_id']] = full_text
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding corpus JSON: {e} in line: {line.strip()}")
            except KeyError as e:
                logging.error(f"Error: Missing key '{e}' in corpus entry: {doc}")
except FileNotFoundError:
    logging.error(f"Corpus file not found at: {corpus_file}")
    exit(1)

logging.info("Loaded corpus with %d documents", len(corpus_dict))
logging.info("Memory Usage After Corpus Load: RSS = %.2f MB, VMS = %.2f MB", *tracemalloc.get_traced_memory())

# Transform corpus dictionary to list of dictionaries
corpus = [{"_id": doc_id, "title": "", "text": text} for doc_id, text in corpus_dict.items()]

logging.info("Transformed corpus to list format with %d documents", len(corpus))
logging.info("Memory Usage After Corpus Transform: RSS = %.2f MB, VMS = %.2f MB", *tracemalloc.get_traced_memory())

queries_data = {}
try:
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                query = json.loads(line)
                queries_data[query['_id']] = query['text']
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding queries JSON: {e} in line: {line.strip()}")
            except KeyError as e:
                logging.error(f"Error: Missing key '{e}' in query entry: {query}")
except FileNotFoundError:
    logging.error(f"Queries file not found at: {queries_file}")
    exit(1)

logging.info("Loaded %d queries", len(queries_data))
logging.info("Memory Usage After Queries Load: RSS = %.2f MB, VMS = %.2f MB", *tracemalloc.get_traced_memory())

logging.info("Loading model...")
try:
    if 'contriever' in args.model_code:
        encoder = Contriever.from_pretrained(model_code_to_cmodel_name[args.model_code]).cuda()
        tokenizer = transformers.BertTokenizerFast.from_pretrained(model_code_to_cmodel_name[args.model_code])
        model = DRES(DenseEncoderModel(encoder, doc_encoder=encoder, tokenizer=tokenizer),
                     batch_size=args.per_gpu_batch_size)
    elif 'dpr' in args.model_code:
        model = DPR((model_code_to_qmodel_name[args.model_code], model_code_to_cmodel_name[args.model_code]))
        model.to(device) # Move DPR model to device
    elif 'ance' in args.model_code:
        model = DRES(models.SentenceBERT(model_code_to_cmodel_name[args.model_code]),
                     batch_size=args.per_gpu_batch_size)
    else:
        raise NotImplementedError
    logging.info(f"Model loaded: {model.model}")
except Exception as e:
    logging.error("Error loading model: %s", str(e))
    logging.error("Traceback: %s", traceback.format_exc())
    raise

logging.info("Memory Usage After Model Load: RSS = %.2f MB, VMS = %.2f MB", *tracemalloc.get_traced_memory())

logging.info("Encoding corpus...")

corpus_ids = [item['_id'] for item in corpus]
corpus_texts = [item['text'] for item in corpus]

# ✅ 使用 tqdm 分批编码 corpus
import numpy as np
corpus_embeddings_list = []
for i in tqdm(range(0, len(corpus), args.per_gpu_batch_size), desc="Encoding corpus", ncols=100):
    batch_corpus = corpus[i:i + args.per_gpu_batch_size]
    batch_embeddings = model.model.encode_corpus(batch_corpus, batch_size=args.per_gpu_batch_size)
    corpus_embeddings_list.append(batch_embeddings)

corpus_embeddings_numpy = np.concatenate(corpus_embeddings_list, axis=0)
corpus_embeddings = torch.tensor(corpus_embeddings_numpy).to(device) # 转换为 PyTorch 张量并移动到设备

logging.info("Corpus encoded")
logging.info("Memory Usage After Corpus Encoding: RSS = %.2f MB, VMS = %.2f MB", *tracemalloc.get_traced_memory())

logging.info("Starting retrieval...")

results = {}
query_ids = list(queries_data.keys())
query_texts = list(queries_data.values())

# ✅ 使用 tqdm 显示查询检索进度
for i in tqdm(range(0, len(query_texts), args.per_gpu_batch_size), desc="Retrieving queries", ncols=100):
    batch_query_ids = query_ids[i:i + args.per_gpu_batch_size]
    batch_query_texts = query_texts[i:i + args.per_gpu_batch_size]

    query_embeddings_numpy = model.model.encode_queries(batch_query_texts, batch_size=args.per_gpu_batch_size)
    query_embeddings = torch.tensor(query_embeddings_numpy).to(device)

    # Calculate similarity
    if args.score_function == 'dot':
        scores = torch.matmul(query_embeddings, corpus_embeddings.transpose(0, 1))
    elif args.score_function == 'cos_sim':
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        corpus_embeddings = torch.nn.functional.normalize(corpus_embeddings, p=2, dim=1)
        scores = torch.matmul(query_embeddings, corpus_embeddings.transpose(0, 1))
    else:
        raise ValueError("Invalid score function")

    # Get top k results
    top_k_scores, top_k_indices = torch.topk(scores, min(args.top_k, len(corpus)), dim=1)
    for query_index, query_id in enumerate(batch_query_ids):
        results[query_id] = {}
        for rank, doc_index in enumerate(top_k_indices[query_index]):
            doc_id = corpus_ids[doc_index]
            score = top_k_scores[query_index][rank].item()
            results[query_id][doc_id] = score

logging.info("Retrieval finished")
logging.info("Memory Usage After Retrieval: RSS = %.2f MB, VMS = %.2f MB", *tracemalloc.get_traced_memory())

logging.info("Writing results to %s", args.result_output)
try:
    with open(args.result_output, 'w') as f:
        json.dump(results, f, indent=4)
except Exception as e:
    logging.error("Error writing results to file: %s", str(e))
    logging.error("Traceback: %s", traceback.format_exc())
    raise

gc.collect()
logging.info("Final Memory Usage: RSS = %.2f MB, VMS = %.2f MB", *tracemalloc.get_traced_memory())
tracemalloc.stop()

# ✅ 打印总运行时间
end_time = time.time()
total_time = end_time - start_time
logging.info("Total runtime: %.2f seconds (%.2f minutes)", total_time, total_time / 60)
