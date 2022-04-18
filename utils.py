import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def label_cross_entropy(preds, labels):
    return F.binary_cross_entropy_with_logits(preds, labels)


def label_accuracy(preds, target):
    _, preds = preds.max(-1)
    _, target = target.max(-1)
    correct = preds.int().data.eq(target.int()).sum()
    count = correct.tolist()
    return correct.float() / (target.size(0)), count


def label_accuracy2(preds, target):
    count = 0
    for i in range(len(target)):
        if preds[i] == target[i]:
            count += 1
    return count / len(target), count


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {
        c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
    }
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def edge_emb(num_region):
    off_diag = np.ones([num_region, num_region]) - np.eye(num_region)
    receive_mat = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    send_mat = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    receive_mat = torch.FloatTensor(receive_mat)
    send_mat = torch.FloatTensor(send_mat)
    return receive_mat, send_mat


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, temp=1, eps=1e-10, dim=-1):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.to(device)
    y = logits + gumbel_noise
    return F.softmax(y / temp, dim=dim)


def gumbel_softmax(logits, temp=1, eps=1e-10, dim=-1):
    y = gumbel_softmax_sample(logits, temp=temp, eps=eps, dim=dim)

    return y
