import sys, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


def neg_sample(packed_sess):
    """
    input:
        packed_sess: torch packed_sequence containing one hot encoded items in the batch of sessions
    return 
        packed: mask, same size as input, containing negative samples 
    
    """
    padded_sess, lengths = pad_packed_sequence(packed_sess)

    padded_samples = torch.zeros_like(padded_sess, dtype=torch.uint8)
    # padded_sess = padded_sess.cpu().numpy()
    item_in_batches = torch.sum(padded_sess, dim=0) # sum over length dimension: items in each sessions in batch
    item_total = torch.sum(item_in_batches, dim=0) # sum over batch dimension: total items in batch
    
    for i, length in enumerate(lengths):
        item_this = item_in_batches[i, :]
        
        assert(item_this.shape == item_total.shape)
        item_other = np.where((item_total != 0) * (item_this == 0))[0] # items absent only in this session
        sample_indices = np.random.choice(item_other, size=(length, ))
        
        padded_samples[np.arange(length), i, sample_indices] = 1

    return pack_padded_sequence(padded_samples, lengths)


def bpr_loss(y_input, y_target, n_samp=4):
    # getting tensors, batch size from packed sequences
    score, batch_sizes = y_input

    target, _ = y_target
    batch_size = batch_sizes[0]

    loss_batch = torch.zeros((batch_size, 1), dtype=torch.float32)
    for i in range(batch_size):
        pos_score = torch.masked_select(score, target.byte())
        for j in range(n_samp):
            samples = neg_sample(y_target)
            neg_score = torch.masked_select(score, samples[0])
            loss_sample = - torch.sum(F.logsigmoid(pos_score - neg_score)) / n_samp
            loss_batch[i] += loss_sample
            
    return torch.sum(loss_batch) / batch_size.float()

    
def top1_loss(y_input, y_target, n_samp=4):
    # getting tensors, batch size from packed sequences
    score, batch_sizes = y_input
    target, _ = y_target
    batch_size = batch_sizes[0]

    loss_batch = torch.zeros((batch_size, 1), dtype=torch.float32)
    for i in range(batch_size):
        pos_score = torch.masked_select(score, target.byte())
        for j in range(n_samp):
            samples = neg_sample(y_target)
            neg_score = torch.masked_select(score, samples[0])
            
            regularization = torch.sum(neg_score * neg_score)
            loss_sample = torch.sum(F.sigmoid(neg_score - pos_score)) 

            loss_batch[i] += (loss_sample + regularization) / n_samp
        
        return torch.sum(loss_batch) / batch_size.float()



sys.path.append('./')
from data_loader import RSC15DataLoader


if __name__ == "__main__":
    dl = RSC15DataLoader('datasets/data', batch_size=5, shuffle=True)
    for batch in dl:
        print('batches')
        print(batch[0].shape) # data
        print(batch[1]) # batch sizes

        # padded = pad_packed_sequence(batch)
        # print("\npadded")
        # print(padded[0].shape)
        # print(padded[1].shape)

        print("\nnegative sampling")
        sample = neg_sample(batch)
        print(sample[0].shape)
        print(sample[1])

        dummy_score = (torch.rand_like(batch[0]) + 3 * batch[0]) / 4
        # possitive_score = torch.masked_select(dummy_score, batch[0].byte())
        # sampled_score = torch.masked_select(dummy_score, sample[0].byte())
        # paired_score = possitive_score - sampled_score
        # print(possitive_score)
        # print(max_score)
        # print(possitive_score.shape)
        # print(sampled_score)
        # print(sampled_score.shape)
        # print(paired_score)
        # print(paired_score.shape)
        max_score, _ = torch.max(dummy_score, 1)
        # items_in_batch(batch)
        BPR_loss = bpr_loss((dummy_score, batch[1]), batch)
        TOP1_loss = top1_loss((dummy_score, batch[1]), batch)
        print('BPR_loss: ', BPR_loss)
        print('Top1_loss: ', TOP1_loss)

        break

