import matplotlib.pyplot as plt

steps = [0,1000,2000,3000,4000,5000]
train_losses = [4.2876,1.2700,1.0512,0.8651, 0.7048,0.6183]
val_losses =  [4.2826,1.5110,1.4641,1.5208,1.6242,1.7050]

plt.figure(figsize=(8, 5))
plt.plot(steps, train_losses, 'b-o', label='Train Loss')
plt.plot(steps, val_losses, 'r-s', label='Val Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('nanoGPT Training on Shakespeare')
plt.legend()
plt.grid(True)


plt.savefig('assets/loss_curve.png', dpi=150, bbox_inches='tight')
print("图片已保存到 assets/loss_curve.png")