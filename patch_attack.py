import argparse

import random
import torch
from torch.autograd import Variable
import numpy as np
from utils.data_list import ImageList
import utils.pre_process as prep
from utils.patch_utils import patch_initialization,  clamp_patch, un_normalize
from torchvision.utils import save_image as t_save_img
import os
from tqdm import tqdm
from network import ResNet, AlexNet



from utils.tools import CalcTopMap


def target_adv_loss(batch_output, real_output, target_hash):
    weights = torch.where((real_output - target_hash).absolute() < 0.3, 0, 1)
    k = weights.sum()
    if k == 0:
        return - float('inf')
    products = weights * batch_output @ target_hash.t()
    loss = - products.sum() / k

    return loss
# 以点积之负为损失函数


def mask_generation(patch,  img_size=(3, 224, 224)):
    # masks = applied_patchs.copy()
    applied_patch = np.zeros(img_size)
    x_location = 200 - patch.shape[1]
    y_location = 200 - patch.shape[2]
    # y_location = 50
    applied_patch[:, x_location: x_location + patch.shape[1], y_location: y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0

    return mask, applied_patch ,x_location, y_location


def load_data(args, train_list):

    train = ImageList(open(train_list).readlines(),
                         transform=prep.image_test(resize_size=255, crop_size=224))
    train_loader = torch.utils.data.DataLoader(train, batch_size=args['batch_size'], shuffle=True,
                                                  num_workers=args['num_workers'])
    return train_loader


def get_args():
    parser = argparse.ArgumentParser(description='single_patch_on_one_class')
    parser.add_argument('--gpu_id', type=int, default=0, help='id of which gpu')
    parser.add_argument('--num_iteration', type=int, default=1000, help='number of iteration')
    parser.add_argument('--noise_percentage', type=float, default=0.05, help='patch of the percentage')
    parser.add_argument('--hash_bit', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.01, help='the alpha of tanh')
    parser.add_argument('--source_txt', type=str, default='./attack/source0.txt', help='the training pics from source')
    parser.add_argument('--test_txt', type=str, default='./attack/test0.txt', help='the testing from source')
    parser.add_argument('--product_threshold', type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_arg', type=int, default=2)
    parser.add_argument('--attack_framework', type=str, default='CSQ')
    parser.add_argument('--model_type', type=str, default='ResNet50')
    parser.add_argument('--attack_which', type=int, default=1)
    config = parser.parse_args()
    args = vars(config)
    return args


def load_model(args):
    if 'ResNet' in args['model_type']:
        model = ResNet(args['hash_bit'], res_model=args['model_type'])
    else:
        model = AlexNet(args['hash_bit'])
    model.load_state_dict(torch.load(args['model_path'],  map_location='cuda:0'))
    model.eval()
    return model



def load_model_and_hashcenter(args):
    hashcenters = np.load(args['hashcenters_path']).astype('float32')
    target_hash = torch.from_numpy(hashcenters[args['attack_which']]).unsqueeze(0).cuda()
    model = load_model(args).cuda()
    return model, target_hash


def program_count():
    coun = 0
    with open('./attack/' + args['attack_framework'] + '/count.txt') as f:
        coun = int(f.readline()) + 1
    with open('./attack/' + args['attack_framework'] + '/count.txt', 'w') as f:
        f.write(str(coun))
    return coun


def save_imgs(perturbated_images, new_patch, product, epoch, count, products, idx):
    now = "epoch_" + str(epoch)
    path = './attack/' + args['attack_framework'] + '/experiment_' + str(count)
    for i in range(perturbated_images.shape[0]):
        if i == 0:
            t_save_img(un_normalize(perturbated_images[i]), path + '/pert_' + now + '_' + str(idx)+ '_' + str(products[0][i].item()) + '.JPEG')
    t_save_img(un_normalize(new_patch.squeeze(0)), path+'/patch_' + now + '_' + str(idx)+'_' + str(product.item())+'.JPEG')


def get_factor(product):
    if product /args['hash_bit'] > 0.9:
        return 1
    return 0.01


def grad_aggregation(patch_grads, all_product):
    # hall_product)
    aggregated_grad = torch.zeros_like(patch_grads[0])
    hamdiss = torch.zeros_like(all_product) + abs(min(all_product))+ 2
    # hamdiss = torch.zeros_like(all_product) + args['hash_bit']
    for i in range(patch_grads.shape[0]):
        hamdiss[i] =  hamdiss[i] + all_product[i]
        aggregated_grad += ( hamdiss[i] * patch_grads[i])
    return aggregated_grad / hamdiss.sum()


def cal_map(args, per_codes, attack_labels):
    if args['attack_framework'] == 'CSQ':
        database_binary = np.load('./save/CSQ/0.8846248014569444/imagenet0.8846248014569444-database_binary.npy')
        database_label = np.load('./save/CSQ/0.8846248014569444/imagenet0.8846248014569444-database_label.npy')
    elif args['attack_framework'] == 'HashNet':
        database_binary = np.load('./save/HashNet/0.6546396906195787/imagenet0.6546396906195787-database_binary.npy')
        database_label = np.load('./save/HashNet/0.6546396906195787/imagenet0.6546396906195787-database_label.npy')
    elif args['attack_framework'] == 'DSHSD':
        database_binary = np.load('./save/DSHSD/imagenet0.860870342547366-trn_binary.npy')
        database_label = np.load('./save/HashNet/0.6546396906195787/imagenet0.6546396906195787-database_label.npy')
    mAP = CalcTopMap(database_binary, per_codes, database_label, attack_labels, topk=1000)
    return mAP


def batch_patch_attack(images, applied_patch, mask, target_hash, model, product_threshold, alpha=0.01, sigma=1/255, max_iterations=2000):
    new_shape = list(mask.shape)
    new_shape.insert(0, images.shape[0] )
    perturbated_images = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul(1 - mask.expand(new_shape ).type(torch.FloatTensor),  images.type(torch.FloatTensor))
    sigma = 1 / 255

    avg_product, count = 0, 0
    while avg_product < product_threshold and count < max_iterations:

        alpha = get_factor(avg_product)
        count += 1
        perturbated_images = Variable(perturbated_images.data, requires_grad=True)
        per_image = perturbated_images
        per_image = per_image.cuda()
        output = model.adv_forward(per_image, alpha)
        real_output = model.adv_forward(per_image)
        model.zero_grad()
        loss = target_adv_loss(output, real_output, target_hash)
        if loss == - float('inf'):
            break
        print(loss)
        loss.backward()
        patch_grad = perturbated_images.grad.clone().cpu()
        perturbated_images.grad.data.zero_()
        if args['mean'] == 1:
            patch_grad = patch_grad.mean(dim=0)
        else:
            patch_grad = grad_aggregation(patch_grads=patch_grad, all_product=all_product)

        applied_patch = applied_patch.type(torch.FloatTensor) - torch.sign(patch_grad) * sigma
        applied_patch = clamp_patch(applied_patch)

        perturbated_images = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul(
            1 - mask.expand(new_shape).type(torch.FloatTensor), images.type(torch.FloatTensor))
        perturbated_images = perturbated_images.cuda()
        output = model.adv_forward(perturbated_images)
        print(output)
        tem = output.detach()
        all_product = torch.mm(torch.sign(tem), target_hash.view(-1, 1 ) )
        avg_product = all_product.sum() / tem.shape[0]
        print(all_product)

        print(avg_product.item())
        if float(avg_product.item()) > product_threshold:
            # wandb.log({'loss': loss.item(), 'avg_product': avg_product.item()})
            break

        wandb.log({'loss': loss.item(), 'avg_product': avg_product.item()})
    perturbated_images = perturbated_images
    applied_patch = applied_patch
    return perturbated_images, applied_patch, avg_product, all_product.reshape(1, -1)


def compute_result(dataloader, applied_patch, mask, net, device):
    bs, bs_2, clses = [], [], []
    net.eval()
    for img, cls in tqdm(dataloader):
        perturbated_images = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul(
            1 - mask.type(torch.FloatTensor), img.type(torch.FloatTensor))
        bs.append((net(perturbated_images.to(device))).data.cpu())
        bs_2.append((net(img.to(device))).data.cpu() )
        clses.append(cls)
    return torch.cat(bs).sign(), torch.cat(bs_2).sign(), torch.cat(clses)


def get_sigma(mask):
    sigma = torch.zeros_like(mask)
    sigma[:, :, :] = 1 / 255
    std = torch.FloatTensor([[0.229], [0.224], [0.225]])
    std = std.unsqueeze(-1).expand(mask.shape)
    sigma.div(std)
    return sigma


def args_setting(args):
    if args['hash_bit'] == 64:
        # hashcenter_path
        if args['attack_framework'] == 'CSQ':
            if args['model_type'] =='ResNet50':
                args['hashcenters_path'] = './save/CSQ/hashcenters.npy'
                args['model_path'] = './save/CSQ/imagenet-0.8857249423099585-model.pt'
            #
        elif args['attack_framework'] == 'HashNet':
            if args['model_type'] == 'ResNet50':
                args['hashcenters_path'] = './save/HashNet/0.6546396906195787/hashcenters.npy'
                args['model_path'] = './save/HashNet/0.6546396906195787/imagenet-0.6546396906195787-model.pt'
            #
        elif args['attack_framework'] == 'DSHSD':
            if args['model_type'] == 'ResNet50':
                args['hashcenters_path'] = './save/DSHSD/hashcenters.npy'
                args['model_path'] = './save/DSHSD/imagenet-0.860870342547366-model.pt'
            #

    return args


def white_box_attack(mAP_loader, applied_patch, mask, model):
    # white-box attack
    per_codes, org_codes, org_labels = compute_result(mAP_loader, applied_patch, mask, model, device="cuda:0")
    np.save('./attack/' + args['attack_framework'] + '/experiment_' + str(count) + "/" + str(epoch) + '_' + str(
        idx) + "per_codes.npy", per_codes.numpy())
    np.save('./attack/' + args['attack_framework'] + '/experiment_' + str(count) + "/" + str(epoch) + '_' + str(
        idx) + "org_codes.npy", org_codes.numpy())
    np.save('./attack/' + args['attack_framework'] + '/experiment_' + str(count) + "/" + str(epoch) + '_' + str(
        idx) + "org_labels.npy", org_labels.numpy())
    attack_labels = np.zeros([num, 100])
    attack_labels[:, args['attack_which']] = 1

    np.save('./attack/' + args['attack_framework'] + '/experiment_' + str(count) + "/" + str(epoch) + '_' + str(
        idx) + "_labels.npy", attack_labels)
    np.save('./attack/' + args['attack_framework'] + '/experiment_' + str(count) + "/" + str(epoch) + '_' + str(
        idx) + "_patch.npy", patch.numpy())
    mAP = cal_map(args, per_codes, attack_labels)
    print("white-box-mAP: ", mAP)
    # wandb.log({'mAP': mAP})
    return mAP


GLOBAL_SEED = 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


GLOBAL_WORKER_ID = None


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


if __name__ == '__main__':
    args = get_args()
    args = args_setting(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu_id'])
    print("attack model :", args['model_path'])
    model, target_hash = load_model_and_hashcenter(args)

    mAP_loader = load_data(args,args['test_txt'])
    loader = load_data(args,  args['source_txt'])
    num = len(open(args['test_txt']).readlines())
    patch = patch_initialization(noise_percentage=args['noise_percentage'])
    mask, applied_patch, x, y = mask_generation(patch)

    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)

    count = program_count()
    if not os.path.exists('./attack/' + args['attack_framework']+'/experiment_'+str(count)):
        os.mkdir('./attack/' + args['attack_framework']+'/experiment_'+str(count))

    target_hash = target_hash.cuda()
    # import wandb
    import re

    source = re.findall(r"\d+\.?\d*", args['source_txt'])[0]

    # wandb.init(project="DeepHashPatchAttack", config=args,
               # name="source_" + str(source) + "attack_" + str(args['attack_which']))

    batch_patchs = applied_patch.unsqueeze(0)
    product = 0
    sigma = get_sigma(mask)
    mAP = 0
    for epoch in range(args['epochs']):
        print(epoch)
        # wandb.log({"epoch": epoch})

        for idx, (image, label) in enumerate(loader):

            if batch_patchs.shape[0] >= 2:
                applied_patch = batch_patchs.mean(dim=0)
                # applied_patch = args['last_patch_percent'] * batch_patchs[-1] + (1 -args['last_patch_percent']) * batch_patchs[-2]
            else:
                applied_patch = applied_patch
            perturbated_images, this_patch, product, all_product = batch_patch_attack( image, applied_patch, mask, target_hash,
                                                                     model, args['product_threshold'], args['alpha'], sigma,
                                                                     max_iterations=args['num_iteration'])
            patch = this_patch[:, x: x + patch.shape[1], y: y + patch.shape[2]]
            batch_patchs = torch.cat([batch_patchs, this_patch.unsqueeze(0)])
            if batch_patchs.shape[0] >= args['num_arg']:
                batch_patchs = batch_patchs[- args['num_arg']:, :, :, :]
            else:
                batch_patchs = batch_patchs[-2:, :, :, :]
            save_imgs(perturbated_images, patch, product, epoch, count, all_product,idx)

        if product >= 0.8 * args['hash_bit']:
            mAP = white_box_attack(mAP_loader, applied_patch, mask, model)

        if product >= (args['hash_bit'] - 0.8)  and mAP > 0.8:
            break

