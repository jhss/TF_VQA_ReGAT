"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.
"""
import numpy as np
import math

def bb_intersection_over_union(boxA, boxB):

    inter_x1, inter_y1 = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    inter_x2, inter_y2 = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

    A_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    B_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = inter_area / (A_area + B_area - inter_area)

    return iou

def build_graph(bbox, spatial, label_num=11):
    """ Build spatial graph

    Args:
        bbox: [num_boxes, 4]

    Returns:
        adj_matrix: [num_boxes, num_boxes, label_num]
    """

    num_box = bbox.shape[0]
    adj_matrix = np.zeros((num_box, num_box))
    xmin, ymin, xmax, ymax = np.split(bbox, 4, axis=1)
    # [num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    image_h = bbox_height[0]/spatial[0, -1]
    image_w = bbox_width[0]/spatial[0, -2]
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    image_diag = math.sqrt(image_h**2 + image_w**2)
    for i in range(num_box):
        bbA = bbox[i]
        if sum(bbA) == 0:
            continue
        adj_matrix[i, i] = 12
        for j in range(i+1, num_box):
            bbB = bbox[j]
            if sum(bbB) == 0:
                continue
            # class 1: inside (j inside i)
            if xmin[i] < xmin[j] and xmax[i] > xmax[j] and \
               ymin[i] < ymin[j] and ymax[i] > ymax[j]:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 2
            # class 2: cover (j covers i)
            elif (xmin[j] < xmin[i] and xmax[j] > xmax[i] and
                  ymin[j] < ymin[i] and ymax[j] > ymax[i]):
                adj_matrix[i, j] = 2
                adj_matrix[j, i] = 1
            else:
                ioU = bb_intersection_over_union(bbA, bbB)
                # class 3: i and j overlap
                if ioU >= 0.5:
                    adj_matrix[i, j] = 3
                    adj_matrix[j, i] = 3
                else:
                    y_diff = center_y[i] - center_y[j]
                    x_diff = center_x[i] - center_x[j]
                    diag = math.sqrt((y_diff)**2 + (x_diff)**2)
                    if diag < 0.5 * image_diag:
                        sin_ij = y_diff/diag
                        cos_ij = x_diff/diag
                        if sin_ij >= 0 and cos_ij >= 0:
                            label_i = np.arcsin(sin_ij)
                            label_j = 2*math.pi - label_i
                        elif sin_ij < 0 and cos_ij >= 0:
                            label_i = np.arcsin(sin_ij)+2*math.pi
                            label_j = label_i - math.pi
                        elif sin_ij >= 0 and cos_ij < 0:
                            label_i = np.arccos(cos_ij)
                            label_j = 2*math.pi - label_i
                        else:
                            label_i = -np.arccos(sin_ij)+2*math.pi
                            label_j = label_i - math.pi
                        adj_matrix[i, j] = int(np.ceil(label_i/(math.pi/4)))+3
                        adj_matrix[j, i] = int(np.ceil(label_j/(math.pi/4)))+3
    return adj_matrix

def tf_broadcast_adj_matrix(adj_matrix, label_num = 11):
    pass


def tf_extract_position_embedding(position_mat, feat_dim, wave_length = 1000,):

    '''feat_range = tf.range(0, feat_dim / 8)
    dim_mat    = tf.math.pow(tf.ones((1,)) * wave_length,
                             (8. / feat_dim) * feat_rnage)
    dim_mat    = tf.reshape(dim_mat, shape = (1, 1, 1, -1))

    position_mat = tf.expand_dims(100.0 * position_mat, axis = 4)

    div_mat = tf.math.divide(position_mat, dim_mat)
    sin_mat = tf.math.sin(div_mat)
    cos_mat = tf.math.cos(div_mat)

    embedding = tf.concat([sin_mat, cos_mat], axis = -1)

    embedding = tf.reshape(embedding, shape = (embedding.shape[0], embedding.shape[1],
                                               embedding.shape[2], feat_dim))'''

    feat_range = np.arange(0, feat_dim / 8)
    dim_mat    = np.power(np.ones((1,)) * wave_length,
                          (8. / feat_dim) * feat_range)
    dim_mat    = np.reshape(dim_mat, newshape = (1, 1, 1, -1))

    position_mat = np.expand_dims(100.0 * position_mat, axis = 4)

    div_mat = np.divide(position_mat, dim_mat)
    sin_mat = np.sin(div_mat)
    cos_mat = np.cos(div_mat)

    embedding = np.concatenate([sin_mat, cos_mat], axis = -1)

    embedding = np.reshape(embedding, newshape = (embedding.shape[0], embedding.shape[1],
                                               embedding.shape[2], feat_dim))

    return embedding

def tf_extract_position_matrix(bbox, nongt_dim = 36):
    #print("[DEBUG] bbox: ", bbox)
    #print("[DEBUG] bbox.shape: ", bbox.shape)
    split = np.split(bbox, 1, axis = -1)
    #print("[DEBUG] len split: ", len(split))
    #print("[DEBUG] split: ", split[0].shape)
    print("----------------")
    xmin, ymin, xmax, ymax = bbox[:,:,0:1], bbox[:,:,1:2], bbox[:,:,2:3], bbox[:,:,3:4]
    #print("xmin: ", xmin)
    #print("ymin: ", ymin)
    #print("xmax: ", xmax)
    #print("ymax: ", ymax)

    # [batch_size,num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    #print("[DEBUG] center_x.shape: ", center_x.shape)
    #print("[DEBUG] center trnaspose.shape: ", np.transpose(center_x, [0, 2, 1]).shape)
    delta_x = center_x - np.transpose(center_x, [0, 2, 1])
    delta_x = np.divide(delta_x, bbox_width)
    #print("[DEBUG] delta_x: ", delta_x)
    #print("[DEBUG] delta_x.shape: ", delta_x.shape)

    delta_x = np.abs(delta_x)
    threshold = 1e-3
    delta_x[delta_x < threshold] = threshold
    delta_x = np.log(delta_x)
    delta_y = center_y-np.transpose(center_y, [0, 2, 1])
    delta_y = np.divide(delta_y, bbox_height)
    delta_y = np.abs(delta_y)
    delta_y[delta_y < threshold] = threshold
    delta_y = np.log(delta_y)
    delta_width = np.divide(bbox_width, np.transpose(bbox_width, [0, 2, 1]))
    delta_width = np.log(delta_width)
    delta_height = np.divide(bbox_height, np.transpose(bbox_height, [0, 2, 1]))
    delta_height = np.log(delta_height)
    concat_list = [delta_x, delta_y, delta_width, delta_height]

    for idx, sym in enumerate(concat_list):
        sym = sym[:, :nongt_dim]
        concat_list[idx] = np.expand_dims(sym, axis = 3)

    position_matrix = np.concatenate(concat_list, 3)

    return position_matrix

def prepare_graph_variables(relation_Type, bb, sem_adj_matrix, spa_adj_matrix,
                            num_objects, nongt_dim, pos_emb_dim, spa_label_num,
                            sem_label_num):

    #print("[DEBUG] prepare bb.shape: ", bb.shape)
    pos_mat = tf_extract_position_matrix(bb, nongt_dim = nongt_dim)
    #print("[DEBUG] pos_mat.shape: ", pos_mat.shape)
    pos_emb = tf_extract_position_embedding(pos_mat, feat_dim = pos_emb_dim)
    #print("[DEBUG] pos_emb.shape: ", pos_emb.shape)

    return pos_emb, None, None
