import os
import torch
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
import sys

# 注册插件
sys.path.insert(0, '/home/data/OpenOccupancy')
from projects.occ_plugin.occupancy.detectors.occnet import OccNet
import projects.occ_plugin.occupancy

# ====================== 配置 ======================
CONFIG_PATH = "projects/configs/Cascade-Occupancy-Network/without_head.py"
CHECKPOINT_PATH = "work_dirs/without_head/latest.pth"
SAVE_DIR = "nusc_fc_features"
DEVICE = "cuda:0"
# ===================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    cfg = mmcv.Config.fromfile(CONFIG_PATH)
    cfg.data.test.test_mode = True

    if "pretrained" in cfg.model.img_backbone:
        cfg.model.img_backbone.pretrained = None

    model = build_model(cfg.model)
    load_checkpoint(model, CHECKPOINT_PATH, map_location=DEVICE)
    model = model.to(DEVICE).eval()

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, shuffle=False, dist=False
    )

    print("开始提取FC层前特征\n")

    with torch.no_grad():
        for idx, data in enumerate(data_loader):

            # 读取数据
            points = data['points'].data[0]
            points = [p.to(DEVICE) for p in points]

            img_metas = data['img_metas'].data[0]
            img_inputs = data['img_inputs']

            new_img = []
            for item in img_inputs:
                try:
                    tensor = item.data[0]
                except:
                    tensor = item
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.unsqueeze(0).to(DEVICE)
                new_img.append(tensor)

            # 特征提取
            voxel_feats, img_feats, pts_feats, depth = model.extract_feat(
                points=points,
                img=new_img,
                img_metas=img_metas
            )

            feat_3d = voxel_feats[0].cpu().numpy()

            # 特征保存
            np.save(f"{SAVE_DIR}/feature_{idx:04d}.npy", feat_3d)

            print(f"✅ [{idx+1}] 保存成功 | shape: {feat_3d.shape}")

            # 测试只跑5帧
            if idx >= 4:
                break

    print("\n特征提取完成！！！")

if __name__ == '__main__':
    main()
