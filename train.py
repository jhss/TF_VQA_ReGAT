import os
import time
import sys
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers.experimental import Adamax

import utils
from model.position_emb import prepare_graph_variables

def instance_bce_with_logits(labels, logits):
    assert len(logits.shape) == 2
    #print("[DEBUG] labels.dtype: ", labels.dtype, " labels.shape: ", labels.shape, " logits.dtype: ", logits.dtype)
    #print("[DEBUG} label type: ", type(labels), " logits type: ", type(logits))
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels = labels, logits = logits)

    return loss
    # 바교실험 해봐야함

def compute_score_with_logits(logits, labels):
    # argmax
    logits = logits.numpy()
    labels = labels.numpy()

    logits = np.argmax(logits, axis = 1)
    logits = np.expand_dims(logits, axis = 1)

    one_hots = np.zeros(shape = labels.shape)
    np.put_along_axis(one_hots, logits, 1.0, axis = 1)
    #indices, updates, shape, name=None)

    scores = tf.reduce_sum(one_hots * labels)
    return scores

def train(model, train_loader, eval_loader, args):
    N = train_loader.data_len # [Check] 수정
    lr_default      = args.base_lr
    num_epohcs      = args.epochs

    print("[DEBUG] weight decay off")
    optimizer = Adamax(learning_rate = lr_default, beta_1 = 0.9, beta_2 = 0.999,
                       epsilon = 1e-8)

    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    best_eval_score = 0

    model.compile(loss = instance_bce_with_logits, optimizer = optimizer)
    gradual_warmup_steps = [lr_default, lr_default, 1.2 * lr_default, 1.3 * lr_default, 1.4 * lr_default]
    lr_decay_epochs = range(5, num_epochs, args.lr_decay_step)
    #utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f,'
                 % (lr_default, args.lr_decay_step,
                    args.lr_decay_rate) + 'grad_clip=%.2f' % args.grad_clip)
    #logger.write('LR decay epochs: '+','.join(
    #                                    [str(i) for i in lr_decay_epochs]))
    last_eval_score, eval_score = 0, 0

    relation_type = 'implicit' # train_loader.dataset.relation_type

    for epoch in range(0, num_epohcs):

        total_norm, count_norm = 0, 0
        total_loss, train_score = 0, 0
        count, average_loss, att_entropy = 0, 0, 0

        losses = utils.AverageMeter()
        start = end = time.time()

        if epoch < len(gradual_warmup_steps):
            old_lr = model.optimizer.lr.read_value()
            new_lr = gradual_warmup_steps[epoch]
            model.optimizer.lr.assign(new_lr)
            logger.write(f"\nEpoch: {epoch}. Reducing Learning Rate from {old_lr} to {new_lr}")
        elif epoch in lr_decay_epochs:
            old_lr = model.optimizer.lr.read_value()
            new_lr = old_lr * 0.75
            model.optimizer.lr.assign(new_lr)
            logger.write(f"\nEpoch: {epoch}. Reducing Learning Rate from {old_lr} to {new_lr}")

        logger.write("--"*50)
        logger.write(f"[DEBUG] epoch {epoch}, number of steps: {train_loader.data_len}")
        logger.write("--"*50)
        for i, (visual_feature, norm_bb, question, target, _, _, bb,
                spa_adj_matrix, sem_adj_matrix) in enumerate(train_loader.generator()):

            #print("[DEBUG] dtype: ", type(visual_feature), type(norm_bb), type(question), type(target), type(bb), type(spa_adj_matrix), type(sem_adj_matrix))
            #print("[DEBUG] question.shape: ", question.shape)
            #print("[DEBUG] loop visual.shape: ", visual_feature.shape) # [9, 30, 2048]
            batch_size, num_objects = visual_feature.shape[0], visual_feature.shape[1]

            pos_emb, sem_adj_mat, spa_adj_mat = prepare_graph_variables(
                relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
                args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
                args.sem_label_num)

            with tf.GradientTape() as g:
                pred, att = model(visual_feature,
                                  norm_bb, question, pos_emb,
                                  sem_adj_mat,
                                  spa_adj_mat, target)

                loss = instance_bce_with_logits(target, pred)

            gradients = g.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            loss_avg = tf.reduce_mean(loss)
            losses.update(loss_avg.numpy().item(), batch_size)

            if (i+1) % args.print_freq == 0:
                elapsed = utils.timeSince(start, float(i+1)/N)
                logger.write(f"Epoch [{epoch+1}][{i}/{N}] Elapsed {elapsed} Loss: {losses.val:.5f}({losses.avg:.5f})")

			#print("[DEBUG] loss: ", loss_avg)
			#print("[DEBUG] score: ", compute_score_with_logits(pred, target))
            #pbar.update(1)

        eval_score = evaluate(model, train_loader, epoch, args)
        eval_score = eval_score.numpy().item()
        logger.write(f"[DEBUG] eval_score: {eval_score:.4f}")

def train_fit(model, train_loader, eval_loader, args):
    optimizer = Adamax(learning_rate = lr_default, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
    model.compile(optimizer, loss = instance_bce_with_logits)
    model.fit_generator(train_loader, batch_size = 32,
              epochs = args.num_epochs, workers = 8, use_multiprocessing=True
              validation_data = eval_loader, validation_batch_size = 64)

def evaluate(model, eval_loader, epoch, args):

    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write("[DEBUG] Evaluation Start")
    score, upper_bound, num_data = 0.0, 0.0, 0.0
    eval_len = eval_loader.data_len
    entropy = None
    N = eval_loader.data_len
    relation_type = eval_loader.relation_type

    losses = utils.AverageMeter()
    start = end = time.time()

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
        loss_avg = tf.reduce_mean(loss)
        losses.update(loss_avg.numpy().item(), batch_size)

        batch_score = compute_score_with_logits(pred, target)

        score += batch_score

        if (i+1) % args.print_freq == 0:
            elapsed = utils.timeSince(start, float(i+1)/N)
            logger.write(f"Epoch [{epoch+1}][{i}/{N}] Elapsed {elapsed} Loss: {losses.val:.5f}({losses.avg:.5f})")
    score = score / N
    return score
