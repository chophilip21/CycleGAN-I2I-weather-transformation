import os
import glob
import argparse
import shutil
import numpy as np

from tqdm import tqdm
from argparse import Namespace
import yaml
from weathergan.mseg_semantic.tool.universal_demo import run_universal_demo

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="the dataset to create the mseg segmetnations for",
                        nargs='?', default="/data/cloudy", const="/data/cloudy")
    parser.add_argument("--model_path", default="model/mseg-3m.pth",type=str, help="the path to the model to use")
    parser.add_argument("--config_path", default="src/weathergan/mseg_semantic/config/test/default_config_360_ms.yaml", type=str, help="the path to the config to use")
    parser.add_argument("--rank", type=int, help="the rank of the current proccess (default: 0)",
                    nargs='?', default=0, const=0)
    parser.add_argument("--skip_if_exists", type=bool, help="skipps the creation of segmentations if they already exist (default: 1)",
                    nargs='?', default=True, const=True)
    parser.add_argument("--num_gpus", type=int, help="the number of gpus to devide the process (default: 1)",
                    nargs='?', default=1, const=1)
    parser.add_argument("--prefix", type=str, help="the prefix for the split eg. 'daytime/' for the daytime folder in BDD100K (default: "")",
                    nargs='?', default="", const="")
    args = parser.parse_args()
    return args

def createMSegSegmentations(dataset_path, rank, num_gpus, skip_if_exists, prefix):
    model_path = args.model_path
    config = args.config_path
    # load full MSeg config and build args namespace with all required fields
    with open(config) as f:
        mcfg = yaml.safe_load(f)
    train_cfg = mcfg['TRAIN']
    test_cfg = mcfg['TEST']
    args_ns = Namespace(
        # universal_demo flags
        config=config, file_save='default', opts=[],
        # model & input/output
        model_name=model_name, model_path=model_path,
        # train section
        arch=train_cfg['arch'], network_name=train_cfg.get('network_name',''),
        layers=train_cfg['layers'], zoom_factor=train_cfg['zoom_factor'],
        ignore_label=train_cfg['ignore_label'], workers=train_cfg['workers'],
        # test section
        vis_freq=test_cfg['vis_freq'], img_name_unique=test_cfg['img_name_unique'],
        batch_size_val=test_cfg['batch_size_val'], split=test_cfg['split'],
        small=test_cfg['small'], base_size=test_cfg['base_size'],
        test_h=test_cfg['test_h'], test_w=test_cfg['test_w'],
        scales=test_cfg['scales'], has_prediction=test_cfg.get('has_prediction', False),
        index_start=test_cfg.get('index_start',0), index_step=test_cfg.get('index_step',0),
        test_gpu=[rank],
        # placeholders overwritten per image
        input_file=None, save_folder=None, dataset=None
    )
    splits = [f"{prefix}train", f"{prefix}val", f"{prefix}test"]
    for split in splits:
        print(f"{dataset_path}/sequences/{split}")
        if not os.path.exists(f"{dataset_path}/sequences/{split}"):
            continue     
        file_paths = [x for x in sorted(glob.glob(f"{dataset_path}/sequences/{split}/frames/*"))]
        file_paths = np.array_split(file_paths, num_gpus)

        for file_path in tqdm(file_paths[rank], total=len(file_paths[rank]), desc="Creating semantic segmentations"):  
            dir_path = f"{dataset_path}/sequences/{split}/{rank}/gray"
            seq_id = file_path.split("/")[-1]
            save_dir = f"{dataset_path}/sequences/{split}/segmentations_mseg/{seq_id}"
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
            if skip_if_exists and os.path.isdir(save_dir):
                # Get input and output files
                input_files = set(os.path.basename(f) for f in glob.glob(f"{file_path}/*"))
                output_files = set(os.path.basename(f) for f in glob.glob(f"{save_dir}/*"))
                
                # Check if all input files have corresponding segmentations
                if input_files.issubset(output_files):
                    print(f"Skipping seq {seq_id}. All {len(input_files)} input files have corresponding segmentations.")
                    continue
                
            # run MSeg inference in-process
            args_ns.input_file = file_path
            args_ns.dataset = os.path.basename(file_path)
            args_ns.save_folder = dir_path
            run_universal_demo(args_ns, use_gpu=True)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)  
            for file_name in os.listdir(dir_path):
                shutil.move(f"{dir_path}/{file_name}", save_dir)
   
if __name__ == "__main__":
    args = parseArguments()
    print('is anything running')
    dataset_path = os.path.dirname(os.path.abspath("feamgan")) + args.dataset_path
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist")
    else:
        print(f"Dataset path {dataset_path} exists")
        rank = args.rank
        num_gpus = args.num_gpus
        skip_if_exists = args.skip_if_exists
        prefix = args.prefix
        createMSegSegmentations(dataset_path, rank, num_gpus, skip_if_exists, prefix)