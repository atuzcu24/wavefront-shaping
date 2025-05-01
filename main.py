from wavemo import *
import argparse

import torch

print('Visible devices:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'Device {i}:', torch.cuda.get_device_name(i))

parser = argparse.ArgumentParser(description='Train WaveMo')
parser.add_argument('--save_folder', type=str, default='Default_Experiment', help='Save folder')
parser.add_argument('--training_data_dir', type=str, default='/fs/vulcan-datasets/mit_places/data_large', help='Training data directory')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--use_wandb', action='store_true')
args = parser.parse_args()
print('use_wandb', args.use_wandb)

wavemo(
      sim=True, 
      use_modulation=True, 
      #slm_mode='random',
      learn_slm=True, 
      hidden_dim=16,
      nframe=16, 
      batch_size=args.batch_size, #Changed here
      save_folder=args.save_folder, 
      training_data_dir=args.training_data_dir,
      use_wandb=args.use_wandb,
      #resume_ckpt_path="/projectnb/ec522/projects/Group8/output_dir/28k_Sim_Test__LearnSLM_SLM_mlp_1_16_ZernAbe_5_6_N16/checkpoints/pth/Trained_8_hours_epoch31_iter7984_PSNR_31.09.pth"
)  

'''
Example Usage:
source myvenv38/bin/activate # Virtual env
cd wavemo/
python main.py --save_folder /projectnb/ec522/projects/Group8/output_dir/final_test --training_data_dir /projectnb/ec522/projects/Group8/data --batch_size 8 --use_wandb

'''


