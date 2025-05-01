# evaluate_val_checkpoints.py
import torch
import os
import pandas as pd
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader
from network.UNetAttMulti import AttU_Net_Multi
from modulate import gen_masks, generate_zernike_basis
from data import FFN_Sim
from wavemo import evaluate_model
import torch
import torch.nn as nn

# === CONFIG ===
checkpoint_path = "/projectnb/ec522/projects/Group8/output_dir/28k_Sim_Test__LearnSLM_SLM_mlp_1_16_ZernAbe_5_6_N16/checkpoints/pth/Trained_8_hours_epoch31_iter7984_PSNR_31.09.pth"
slm_ckpt_path = checkpoint_path.replace(".pth", "_slm.pth")
val_data_dir = "/projectnb/ec522/projects/Group8/datasets/places365"
save_dir = "output_dir/val_eval_output"
img_size = 256
batch_size = 2
hidden_dim = 16
mlp_hidden_layers = 1
in_dim = 28 
nframe = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset Setup ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(128),
    transforms.Resize((img_size, img_size))
])
val_dataset = FFN_Sim(data_folder=val_data_dir, input_transforms=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === Model Setup ===
model = AttU_Net_Multi(img_ch=1 + nframe, output_ch=1, residual=True).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# === Zernike Setup ===
zernike_basis = generate_zernike_basis(img_size).to(device)
zern_net_basis = zernike_basis.permute(1, 2, 0).unsqueeze(0).repeat(1, 1, 1, 1).to(device)
#zern_net = torch.load(checkpoint_path, map_location=device)['zern_mlp']
"""layers = []
layers.append(nn.Linear(in_dim, hidden_dim))
for _ in range(mlp_hidden_layers):
    #layers.append(nn.LeakyReLU())
    layers.append(nn.Linear(hidden_dim, hidden_dim))
layers.append(nn.Linear(hidden_dim, nframe))
zern_net = nn.Sequential(*layers).to(device)"""

zern_net = nn.Sequential(
    nn.Linear(in_dim, hidden_dim),  # 0
    nn.LeakyReLU(),                 # 1 (activation, no weights)
    nn.Linear(hidden_dim, hidden_dim),  # 2
    nn.LeakyReLU(),                     # 3
    nn.Linear(hidden_dim, nframe)      # 4
).to(device)

zern_net.load_state_dict(torch.load(checkpoint_path, map_location=device)['zern_mlp'])
zern_net.eval()
slm_alphas = zern_net(zern_net_basis).permute(0, 3, 1, 2)

mask_batch = gen_masks(width=img_size, grid_size=1, DEVICE=device, vis=False)

# === Evaluate ===
loss, psnr_val, ssim_val = evaluate_model(
    model=model,
    test_loader=val_loader,
    crop_fn=transforms.CenterCrop(128),
    device=device,
    sim=True,
    use_modulation=True,
    dataset='places',
    slm_mode='mlp',
    slm_alphas=slm_alphas,
    zern_net=zern_net,
    zern_net_basis=zern_net_basis,
    zernike_basis=zernike_basis,
    mask_batch=mask_batch,
    abe_std_low=5.0,
    abe_std_high=6.0,
    grid_size=1,
    nframe=nframe,
    zern_order=7
)

print(f"\nEvaluation Completed. PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.3f}, Loss: {loss:.4f}\n")

# Save results for the first few examples
model.eval()
os.makedirs(save_dir, exist_ok=True)

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if i > 3:
            break
        inputs = batch.to(device)

        # Re-run modulation to match eval setup
        from modulate import generate_zern_patterns, gen_psf, conv_psf
        abe_std = torch.FloatTensor(1).uniform_(5.0, 6.0).to(device)
        abe_alphas = abe_std * torch.rand(1, (7 * 8) // 2).to(device)
        abe_patterns = generate_zern_patterns(abe_alphas, zernike_basis, device=device)
        abe_psfs = gen_psf(abe_patterns)
        y_zero = conv_psf(inputs, abe_psfs, mask=mask_batch)

        slm_alphas = zern_net(zern_net_basis).permute(0, 3, 1, 2)
        slm_patterns = torch.exp(1j * slm_alphas)
        offset = (img_size - int(img_size / 1920 * 1080)) // 2
        slm_patterns = torch.nn.functional.pad(slm_patterns[..., offset:-offset, :], (0, 0, offset, offset), "constant", 0)
        mod_psfs = gen_psf(slm_patterns.permute(1, 0, 2, 3) * abe_patterns)
        y_mod = conv_psf(inputs, mod_psfs, mask=mask_batch)
        sample = torch.cat((y_zero, y_mod), dim=1)

        recon = model(sample)

        folder = os.path.join(save_dir, f"val_idx_{i}")
        os.makedirs(folder, exist_ok=True)
        save_image(inputs[0], f"{folder}/GT.png", normalize=True)
        save_image(sample[0, 0], f"{folder}/Measurement.png", normalize=True)
        save_image(recon[0], f"{folder}/Reconstruction.png", normalize=True)

        pd.DataFrame({"Loss": [loss], "PSNR": [psnr_val], "SSIM": [ssim_val]}).to_csv(f"{folder}/metrics.csv", index=False)

print(f"Saved reconstruction results to: {save_dir}")