import argparse
import math
import numpy as np
import socket
import importlib
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from loading_pointclouds import *
import models.PCAN as PCAN
from tensorboardX import SummaryWriter
import loss.pointnetvlad_loss

import config as cfg

cudnn.enabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    model = PCAN.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False,
                                      output_dim=cfg.FEATURE_OUTPUT_DIM, num_points=cfg.NUM_POINTS)
    model = model.to(device)

    resume_filename = cfg.LOG_DIR + cfg.MODEL_FILENAME
    print("Resuming From ", resume_filename)

    ### 加载模型 ： 通过加载模型的检查点文件恢复训练好的模型状态。
    checkpoint = torch.load(resume_filename, weights_only=False)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)

    model = nn.DataParallel(model)

    print(evaluate_model(model))


def evaluate_model(model):
    DATABASE_SETS = get_sets_dict(cfg.EVAL_DATABASE_FILE)
    QUERY_SETS = get_sets_dict(cfg.EVAL_QUERY_FILE)

    if not os.path.exists(cfg.RESULTS_FOLDER):
        os.mkdir(cfg.RESULTS_FOLDER)

    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    ### 获取数据库和查询集的特征向量 ：使用训练好的模型提取数据库和查询集的特征向量。

    DATABASE_VECTORS = []
    QUERY_VECTORS = []

    # 确保模型处于评估模式
    model.eval()

    for i in range(len(DATABASE_SETS)):
        DATABASE_VECTORS.append(get_latent_vectors(model, DATABASE_SETS[i]))

    for j in range(len(QUERY_SETS)):
        QUERY_VECTORS.append(get_latent_vectors(model, QUERY_SETS[j]))

    ### 计算召回率(Recall) ：使用KDTree查找最近邻，并计算每个查询向量与数据库向量之间的相似度。对于每对查询 - 数据库的比较，计算召回率。

    for m in range(len(QUERY_SETS)):
        for n in range(len(QUERY_SETS)):
            if (m == n):  ## DATABASE_SETS和QUERY_SETS的文件夹数量相同，文件夹下的点云文件数量不同，DATABASE_SETS中的点云文件数包含QUERY_SETS中的点云文件数
                continue  ## m不等于n意味着从DATABASE_SETS和QUERY_SETS中选择不同的文件夹
            pair_recall, pair_similarity, pair_opr = get_recall(
                m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x) ## 记录所有查询点云与数据库点云的相似度

    print()

    ### 计算平均召回率和相似度 ：最终，通过计算recall的平均值和similarity，得出模型的整体性能。

    ave_recall = recall / count
    # print(ave_recall)

    # print(similarity)
    average_similarity = np.mean(similarity)
    # print(average_similarity)

    ave_one_percent_recall = np.mean(one_percent_recall)
    # print(ave_one_percent_recall)

    ### 输出评估结果 ：评估结果会输出到指定的文件中。

    with open(cfg.OUTPUT_FILE, "w") as output:
        output.write("Average Recall @N:\n")
        output.write(str(ave_recall))
        output.write("\n\n")
        output.write("Average Similarity:\n")
        output.write(str(average_similarity))
        output.write("\n\n")
        output.write("Average Top 1% Recall:\n")
        output.write(str(ave_one_percent_recall))

    return ave_one_percent_recall


def get_latent_vectors(model, dict_to_process):

    model.eval()
    is_training = False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = cfg.EVAL_BATCH_SIZE * \
        (1 + cfg.EVAL_POSITIVES_PER_QUERY + cfg.EVAL_NEGATIVES_PER_QUERY) ### 一个批次中包含的全部点云文件数量batch_num = query数量 * （ 每个query + 对应的positive数量 + 对应的negtive数量 ）
    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)  ### 一个批次中包含的全部点云文件数量batch_num全当成queries了

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            # 处理模型返回的元组
            out_tuple = model(feed_tensor)
            vlad = out_tuple[0]  # 只取VLAD特征

        vlad = vlad.detach().cpu().numpy()
        vlad = np.squeeze(vlad)

        #out = np.vstack((o1, o2, o3, o4))
        q_output.append(vlad)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(train_file_idxs) // batch_num * batch_num
    if index_edge < len(dict_to_process.keys()):
        file_indices = train_file_idxs[index_edge:len(dict_to_process.keys())]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            # 处理模型返回的元组
            out_tuple = model(feed_tensor)
            vlad = out_tuple[0]  # 只取VLAD特征

        output = vlad.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    # print(q_output.shape)
    return q_output


def get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS):

    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    # print(len(queries_output))
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        true_neighbors = QUERY_SETS[n][i][m] ## query集合中 第n个文件夹 第i个点云文件 -> 对应的database中的 第m个文件夹 近邻点云文件索引
        if(len(true_neighbors) == 0):        ## n和m是提前固定的，也就是说：分别在query和database固定文件夹中找出真实近邻的点云文件对
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]),k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if(j == 0):
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]]) ## 相似度 = 查询点云的真实最近邻点对的特征乘积
                    top1_similarity_score.append(similarity) ##评估和记录每个查询点云的第一个最近邻相似度
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    # print(recall)
    # print(np.mean(top1_similarity_score))
    # print(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--positives_per_query', type=int, default=4,
                        help='Number of potential positives in each training tuple [default: 2]')
    parser.add_argument('--negatives_per_query', type=int, default=12,
                        help='Number of definite negatives in each training tuple [default: 20]')
    parser.add_argument('--eval_batch_size', type=int, default=12,
                        help='Batch Size during training [default: 1]')
    parser.add_argument('--dimension', type=int, default=256)
    parser.add_argument('--decay_step', type=int, default=200000,
                        help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.7,
                        help='Decay rate for lr decay [default: 0.8]')
    parser.add_argument('--results_dir', default='results/',
                        help='results dir [default: results]')
    parser.add_argument('--dataset_folder', default='/home/wzj/pan1/PointNetVlad-Pytorch/benchmark/',
                        help='PointNetVlad Dataset Folder')
    FLAGS = parser.parse_args()

    #BATCH_SIZE = FLAGS.batch_size
    #cfg.EVAL_BATCH_SIZE = FLAGS.eval_batch_size
    cfg.NUM_POINTS = 4096
    cfg.FEATURE_OUTPUT_DIM = 256
    cfg.EVAL_POSITIVES_PER_QUERY = FLAGS.positives_per_query
    cfg.EVAL_NEGATIVES_PER_QUERY = FLAGS.negatives_per_query
    cfg.DECAY_STEP = FLAGS.decay_step
    cfg.DECAY_RATE = FLAGS.decay_rate

    cfg.RESULTS_FOLDER = FLAGS.results_dir

    cfg.EVAL_DATABASE_FILE = '/home/wzj/pan1/PointNetVlad-Pytorch/generating_queries/oxford_evaluation_database.pickle'
    cfg.EVAL_QUERY_FILE = '/home/wzj/pan1/PointNetVlad-Pytorch/generating_queries/oxford_evaluation_query.pickle'

    cfg.LOG_DIR = 'log/'
    cfg.OUTPUT_FILE = cfg.RESULTS_FOLDER + 'results.txt'
    cfg.MODEL_FILENAME = "model.ckpt"

    cfg.DATASET_FOLDER = FLAGS.dataset_folder

    evaluate()
