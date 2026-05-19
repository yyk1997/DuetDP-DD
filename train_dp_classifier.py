import os
import sys
import argparse
import datetime
import numpy as np

from models.model_loader import load_model
from data.dataset_loader import load_data
from utils.utils import initialize_environment, run, parse_config
from evaluation.evaluator import Evaluator

os.environ['MKL_NUM_THREADS'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(PROJECT_ROOT, "models", "PE", "guided-diffusion"))


def main(config):
    # 初始化训练环境
    initialize_environment(config)

    # 在model的初始化阶段可通过ckpt对训练模型进行加载
    model, config = load_model(config)

    sensitive_train_loader, sensitive_val_loader, sensitive_test_loader, public_train_loader, config = load_data(config)
    semantic_public_dataset = None

    print("load Synthetic dataset")
    gen = np.load(gen_path)
    syn_data = gen['x']
    syn_labels = gen['y']


    acc_models = ["convnet-3"]
    evaluator = Evaluator(config)

    evaluator.train_classifier(syn_data, syn_labels, sensitive_train_loader, sensitive_val_loader, sensitive_test_loader, acc_models, semantic_public_dataset)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', default="configs")
    parser.add_argument('--method', '-m', default="PrivImage")
    parser.add_argument('--epsilon', '-e', default="1.0")
    parser.add_argument('--data_name', '-dn', default="cifar10_32")
    parser.add_argument('--exp_description', '-ed', default="")
    parser.add_argument('--resume_exp', '-re', default=None)
    parser.add_argument('--config_suffix', '-cs', default="")
    opt, unknown = parser.parse_known_args()

    config = parse_config(opt, unknown)

    gen_path = './exp/privimage/cifar10_32_eps1.0_synthetic_dataset/gen/gen.npz'

    classifier_save_path = './exp/privimage/cifar10_32_eps1.0_synthetic_dataset'

    run(main, config)