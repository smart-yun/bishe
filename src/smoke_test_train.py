# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # 语义分割假设：输入 [B,3,H,W]，输出 [B,C,H,W]
    B, C, H, W = 2, 19, 256, 256
    x = torch.randn(B, 3, H, W, device=device)
    y = torch.randint(0, C, (B, H, W), device=device)

    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, C, 1)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for step in range(5):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        print(f"step {step} loss={loss.item():.4f}")

if __name__ == "__main__":
    main()