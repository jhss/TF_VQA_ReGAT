'''
This code is modified by Juhong from Linjie Li's repository.
(Original: PyTorch -> Modified: TensorFlow2.0)
https://github.com/linjieli222/VQA_ReGAT
Lincensed under the MIT license.
'''

import os
import gc
import time
import sys
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers.experimental import Adamax

import utils
from model.position_emb import prepare_graph_variables

def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert len(logits.shape) == 2

    loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels = labels, logits = logits)

    return loss

def compute_score_with_logits(logits, labels):
    logits = logits.numpy()
    labels = labels.numpy()

    logits = np.argmax(logits, axis = 1)
    logits = np.expand_dims(logits, axis = 1)

    one_hots = np.zeros(shape = labels.shape)
    np.put_along_axis(one_hots, logits, 1.0, axis = 1)

    scores = tf.reduce_sum(one_hots * labels)
    return scores

def train(model, train_loader, eval_loader, args):
    N = train_loader.data_loader_len
    num_total_data = train_loader.num_total_data
    lr_default      = args.base_lr
    num_epochs      = args.epochs
    
    # Set optimizer
    optimizer = Adamax(learning_rate = lr_default, beta_1 = 0.9, beta_2 = 0.999,
                       epsilon = 1e-8)

    model.compile(loss = instance_bce_with_logits, optimizer = optimizer)

    # Set LR scheduling
    gradual_warmup_steps = [lr_default, lr_default, 1.2 * lr_default, 1.3 * lr_default, 1.4 * lr_default]
    lr_decay_epochs = range(5, num_epochs, args.lr_decay_step)

    logger = utils.Logger(os.path.join(args.output, 'log.txt'))

    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f,'
                 % (lr_default, args.lr_decay_step,
                    args.lr_decay_rate) + 'grad_clip=%.2f' % args.grad_clip)

    relation_type = args.relation_type

    # Start epoch
    for epoch in range(0, num_epochs):

        train_score = 0
        losses = utils.AverageMeter()
        start = end = time.time()

        # Increase learning rate at the first four epochs (gradual increase)
        if epoch < len(gradual_warmup_steps):
            old_lr = model.optimizer.lr.read_value()
            new_lr = gradual_warmup_steps[epoch]
            model.optimizer.lr.assign(new_lr)
            logger.write(f"\nEpoch: {epoch}. Reducing Learning Rate from {old_lr} to {new_lr}")
        # Reduce learning rate after 5 epochs (gradual decrease)
        elif epoch in lr_decay_epochs:
            old_lr = model.optimizer.lr.read_value()
            new_lr = old_lr * args.lr_decay_rate
            model.optimizer.lr.assign(new_lr)
            logger.write(f"\nEpoch: {epoch}. Reducing Learning Rate from {old_lr} to {new_lr}")
            
        
        logger.write("--"*50)
        logger.write(f"[DEBUG] epoch {epoch}, number of steps: {train_loader.data_loader_len}")
        logger.write("--"*50)

        # Start iterations
        for i, (visual_feature, norm_bb, question, target, bb,
                spa_adj_matrix, sem_adj_matrix) in enumerate(train_loader.generator()):

            batch_size, num_objects = visual_feature.shape[0], visual_feature.shape[1]

            # Get position embedding from bounding boxes
            pos_emb, sem_adj_mat, spa_adj_mat = prepare_graph_variables(
                relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
                args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
                args.sem_label_num)

            # Forward data and calculate loss value
            with tf.GradientTape() as g:
                pred, att = model(visual_feature,
                                  norm_bb, question, pos_emb,
                                  sem_adj_mat,
                                  spa_adj_mat)

                loss = instance_bce_with_logits(pred, target)
                loss_avg = tf.reduce_mean(loss) * tf.cast(tf.shape(target)[1], tf.float32)

            # Estimate gradients and update the parameters
            gradients = g.gradient(loss_avg, model.trainable_variables)             
            gradients = [(tf.clip_by_norm(grad, args.grad_clip)) for grad in gradients]
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Calculate train score
            batch_score = compute_score_with_logits(pred, target)
            train_score += batch_score

            losses.update(loss_avg.numpy().item(), batch_size)

            if (i+1) % args.print_freq == 0:
                elapsed = utils.timeSince(start, float(i+1)/N)
                logger.write(f"Epoch [{epoch+1}][{i}/{N}] Elapsed {elapsed} Loss: {losses.val:.5f}({losses.avg:.5f})")

        
        train_score /= num_total_data

        # Start evaluation
        eval_score = evaluate(model, eval_loader, epoch, args, logger)
        eval_score = eval_score.numpy().item()
        gc.collect()
        logger.write(f"[DEBUG] train_score: {train_score:.4f} eval_score: {eval_score:.4f}")


def evaluate(model, eval_loader, epoch, args, logger):

    logger.write("[DEBUG] Evaluation Start")
    score = 0.0
    num_total_data = eval_loader.num_total_data
    N = eval_loader.data_loader_len
    logger.write(f"[DEBUG] total eval data len: {num_total_data}")
    logger.write(f"[DEBUG] eval data loader len: {N}")
    
    relation_type = eval_loader.relation_type

    losses = utils.AverageMeter()
    start = end = time.time()

    # Start evaluation
    for i, (visual_feature, norm_bb, question, target, _, _, bb,
            spa_adj_matrix, sem_adj_matrix) in enumerate(eval_loader.generator()):

        batch_size, num_objects = visual_feature.shape[0], visual_feature.shape[1]

        pos_emb, sem_adj_mat, spa_adj_mat = prepare_graph_variables(
            relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
            args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
            args.sem_label_num)

        pred, att = model(visual_feature,
                          norm_bb, question, pos_emb,
                          sem_adj_mat,
                          spa_adj_mat, target)

        loss = instance_bce_with_logits(pred, target)
        loss_avg = tf.reduce_mean(loss) * tf.cast(tf.shape(target)[1], tf.float32)
        losses.update(loss_avg.numpy().item(), batch_size)

        batch_score = compute_score_with_logits(pred, target)

        score += batch_score

        if (i+1) % args.print_freq == 0:
            elapsed = utils.timeSince(start, float(i+1)/N)
            logger.write(f"Epoch [{epoch+1}][{i}/{N}] Elapsed {elapsed} Loss: {losses.val:.5f}({losses.avg:.5f})")
    score = score / num_total_data
    return score
