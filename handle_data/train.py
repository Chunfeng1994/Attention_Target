import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler
from handle_data.batch_iter import *
import shutil
import random

torch.manual_seed(233)
random.seed(233)





def train(model, train_data, dev_data, test_data, vocab_srcs, vocab_tgts,config):


    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)



    steps = 0
    best_acc= 0
    best_test = 0
    model.train()
    for epoch in range(1, config.epochs+1):
        #adjust_lr_decay(optimizer,epoch,args)
        for batch in create_batch_iter(train_data, config.batch_size, shuffle=True):

            feature, target, feature_lengths,start,end = pair_data_variable(batch, vocab_srcs, vocab_tgts, config)

            optimizer.zero_grad()
            model.zero_grad()
            logit = model(feature,start,end,feature_lengths)
            loss = F.cross_entropy(logit, target)
            loss_value = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), config.clip_norm,norm_type=2)
            optimizer.step()

            steps += 1
            if steps % config.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = float(corrects)/float(len(batch))
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss_value,
                                                                                 accuracy,
                                                                             corrects,
                                                                             len(batch)))
            if steps % config.test_interval == 0:
                dev_acc = eval(model, dev_data,  vocab_srcs, vocab_tgts, config)
                if(dev_acc > best_acc):
                    best_acc = dev_acc
                print("\nThe DEV Current best result is {} ".format(best_acc))
            if steps % config.save_interval == 0:
                if not os.path.isdir(config.save_dir):
                    os.makedirs(config.save_dir)
                save_prefix = os.path.join(config.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)
                test_model = torch.load(save_path)

                test_acc = test_eval(model, test_data, vocab_srcs, vocab_tgts, config)
                if test_acc > best_test:
                   best_test = test_acc
                print("\nThe TEST Current best result is {} ".format(best_test))

# def adjust_lr_decay(optimizer, epoch,args):
#     if epoch % 50 == 0:
#         lr = args.lr * (0.1 ** (epoch // 30))
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = param_group['lr'] * 0.99


def eval(model,data_iter,vocab_srcs, vocab_tgts,config):
    model.eval()
    corrects, avg_loss,size = 0, 0, 0
    for batch in create_batch_iter(data_iter, config.batch_size):

        feature, target, feature_lengths, start, end = pair_data_variable(batch, vocab_srcs, vocab_tgts, config)

        logit = model(feature, start, end, feature_lengths)
        loss = F.cross_entropy(logit, target, size_average=True)
        loss_value = loss.item()
        avg_loss += loss_value
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        size += len(batch)

    avg_loss = avg_loss/size

    accuracy = float(corrects)/float(size)
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))

    return accuracy


def test_eval(model,data_test,vocab_srcs,vocab_tgts, config):
    model.eval()
    corrects, avg_loss,size = 0, 0, 0
    for batch in create_batch_iter(data_test, config.batch_size):
        feature, target, feature_lengths, start, end = pair_data_variable(batch, vocab_srcs, vocab_tgts, config)

        logit = model(feature, start, end, feature_lengths)
        loss = F.cross_entropy(logit, target, size_average=True)
        loss_value = loss.item()
        avg_loss += loss_value
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        size += len(batch)

    avg_loss = avg_loss/size
    accuracy =  float(corrects)/float(size)
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                      size))

    return accuracy
    # # test result
    # if os.path.exists("./Test_Result.txt"):
    #     file = open("./Test_Result.txt", "a")
    # else:
    #     file = open("./Test_Result.txt", "w")
    # file.write("models " + save_path + "\n")
    # file.write("Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n".format(avg_loss, accuracy, corrects, size))
    # file.write("\n")
    # file.close()
    # # calculate the best score in current file
    # resultlist = []
    # if os.path.exists("./Test_Result.txt"):
    #     file = open("./Test_Result.txt")
    #     for line in file.readlines():
    #         if line[:10] == "Evaluation":
    #             resultlist.append(float(line[34:41]))
    #     result = sorted(resultlist)
    #     file.close()
    #     file = open("./Test_Result.txt", "a")
    #     file.write("\nThe Current Best Result is : " + str(result[len(result) - 1]))
    #     file.write("\n\n")
    #     file.close()
    # shutil.copy("./Test_Result.txt", "./snapshot/" + args.mulu + "/Test_Result.txt")
    # # whether to delete the models after test acc so that to save space
    # if os.path.isfile(save_path) and args.rm_model is True:
    #     os.remove(save_path)
