import torch
import logging
import numpy as np
import os
import argparse

flickr25k_img_path = "./imgdata/MIRFLICKR-25K/images"
nuswide_img_path = "./imgdata/NUS-WIDE/images"
coco_img_path = "./imgdata/MSCOCO/MSCOCO-all-images/coco_images"
flickr25k_class_name_list = ['animals', 'baby', 'bird', 'car', 'clouds', 'dog', 'female', 'flower', 'food', 'indoor', 'lake', 'male', 'night', 'people', 'plant_life', 'portrait', 'river', 'sea', 'sky', 'structures', 'sunset', 'transport', 'tree', 'water']
nuswide_class_name_list = ['animal', 'beach', 'buildings', 'clouds', 'flowers', 'grass', 'lake', 'mountain', 'ocean', 'person', 'plants', 'reflection', 'road', 'rocks', 'sky', 'snow', 'sunset', 'tree', 'vehicle', 'water', 'window']
coco_class_name_list = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-bits-list", type=str, default="16,32,64", help="length of multi-bit hash codes.")
    parser.add_argument("--auxiliary-bit-dim", type=int, default=128, help="length of auxiliary hash codes.")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--noise_ratio", type=float, default=0.4)
    parser.add_argument('--modal_missing_ratio', type=float, default=0.0,help='Probability to mask image or text modality during training/inference')
    parser.add_argument("--res-mlp-layers", type=int, default=2, help="the number of ResMLP blocks.")
    parser.add_argument("--valid-freq", type=int, default=1, help="To valid every valid-freq epochs.")
    parser.add_argument("--rank", type=int, default=0, help="GPU rank")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clip-lr", type=float, default=0.000001, help="learning rate for CLIP in PromptHash.")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate for other modules in PromptHash.")
    parser.add_argument("--is-train", action="store_true")
    parser.add_argument("--is-freeze-clip", default=True)
    parser.add_argument("--tao_global", type=float, default=0.07, help="")
    parser.add_argument("--tao_local", type=float, default=0.07, help="")
    parser.add_argument("--concept-num", type=int, default=196)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--hyper_recon", type=float, default=0.005, help="weight of the recon loss.")
    parser.add_argument("--hyper_global", type=float, default=5, help="weight of the global prompt alignment loss.")
    parser.add_argument("--hyper_local", type=float, default=5, help="weight of the local prompt alignment loss.")
    parser.add_argument("--mu", type=float, default=20, help="")
    parser.add_argument("--hyper_cls_sum", type=float, default=5.005, help="weight of the auxiliary hash code loss.")
    parser.add_argument("--dataset", type=str, default="flickr", help="choose from [coco, flickr25k, nuswide]")
    parser.add_argument("--query-num", type=int, default=2000)
    parser.add_argument("--train-num", type=int, default=10000)
    parser.add_argument("--pretrained", type=str, default="", help="pretrained model path.")
    parser.add_argument("--index-file", type=str, default="index.mat")
    parser.add_argument("--caption-file", type=str, default="caption.mat")
    parser.add_argument("--label-file", type=str, default="label.mat")
    parser.add_argument("--max-words", type=int, default=77)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--result-name", type=str, default="result", help="result dir name.")
    parser.add_argument("--seed", type=int, default=1814)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--image_normalize_factor", type=float, default=1.0)
    parser.add_argument("--text_normalize_factor", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=10000, help="Number of steps to warmup for.")
    parser.add_argument("--warmup-proportion", type=float, default=0.05, help="Proportion of training to perform learning rate warmup.")
    args = parser.parse_args()

    import datetime
    _time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not args.is_train:
        _time += "_test"

    # k_bits = args.k_bits
    k_bits_list = list(map(int, args.k_bits_list.split(",")))  # str -> list

    parser.add_argument("--save-dir", type=str, default=f"./{args.result_name}/{args.dataset}_{k_bits_list}/{_time}")
    args = parser.parse_args()

    return args

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.random.manual_seed(seed)
    np.random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger

def calc_neighbor(train_labels_all: torch.Tensor, query_batch_labels: torch.Tensor) -> torch.Tensor:
    similarity_scores = torch.matmul(query_batch_labels, train_labels_all.t())
    label_similarity_matrix = (similarity_scores > 0).float() 
    return label_similarity_matrix

def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map
