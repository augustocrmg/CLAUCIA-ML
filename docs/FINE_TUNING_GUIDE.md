## Fine-tuning epochs and techniques for image classification (EfficientNet and similar)

### TL;DR
- **How many epochs?**
  - Train the new head with base frozen: **3–5 epochs**.
  - Unfreeze and fine-tune: **5–10 epochs** more.
  - Use **EarlyStopping (patience 3–5)** and stop when validation loss stops improving. In practice this often lands in the ranges above even if max epochs are higher.
- **Best techniques beyond “basic transfer learning”**:
  - Gradual unfreezing; discriminative learning rates; smaller LR for fine-tuning.
  - One-cycle or cosine LR schedules; ReduceLROnPlateau.
  - Keep BatchNorm frozen during fine-tuning; apply weight decay and dropout.
  - Strong augmentation (MixUp, CutMix, RandAugment), label smoothing.

Your current scripts already do several good practices: freeze base, use a low LR for FT (1e-5), keep BatchNorm frozen, and use EarlyStopping. With `EPOCHS_INITIAL=100` and `patience=5`, you will typically converge in far fewer than 100 epochs; `EPOCHS_FINE_TUNING=20` is also within recommended ranges.

---

### Why those epoch ranges?
Pretrained backbones already encode general visual features. Fine-tuning mostly adapts to your domain and classes, which converges faster than training from scratch. Typical practice is:
- Head-only phase: 3–5 epochs to fit the classifier.
- Fine-tuning phase: unfreeze some or all layers, train another 5–10 epochs with a much lower LR and EarlyStopping.

These are starting points; rely on validation curves and EarlyStopping rather than a fixed number.

---

### Techniques that reliably help beyond basic transfer learning
1) Gradual unfreezing
- Unfreeze the model in stages (e.g., last N blocks → more blocks → full), training a few epochs per stage with lower LR.
- Helps avoid catastrophic forgetting and stabilizes training.

2) Discriminative learning rates
- Use higher LR for the newly added head and progressively smaller LRs for earlier layers (earlier layers change least). This often improves convergence quality vs. a single LR.

3) LR schedules that ease fine-tuning
- One-Cycle policy or cosine decay with warmup are effective for quick, high-quality convergence.
- Alternatively, add `ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2–3)` in the FT phase.

4) Keep BatchNorm frozen during FT
- BatchNorm stats from ImageNet are strong priors; updating them on small/shifted datasets can hurt. Keep BN layers non-trainable when unfreezing.

5) Regularization and augmentation
- Weight decay (L2), dropout in the head, and label smoothing (e.g., 0.05–0.1) reduce overfitting.
- Strong policies like MixUp, CutMix, and RandAugment improve generalization, especially with limited or imbalanced data.

6) Monitoring and early stopping
- Monitor `val_loss`; use EarlyStopping with `restore_best_weights=True`. Combine with model checkpointing.

---

### Practical guidance tailored to your scripts
- You already have:
  - Head training → fine-tuning flow, BN frozen, low LR (1e-5) in FT, EarlyStopping.
  - Suggested minor tweaks:
    - Add `ReduceLROnPlateau` in both phases (more important for FT).
    - Consider staging unfreezing by EfficientNet blocks instead of unfreezing everything at once.
    - Consider One-Cycle or cosine schedule if you prefer schedule-based control.
    - Add label smoothing in the loss: `tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)` (or equivalent for sparse labels).
- Epoch caps:
  - Keep high caps (`EPOCHS_INITIAL` 30–50, `EPOCHS_FINE_TUNING` 10–20) and let EarlyStopping decide. Your current caps (100 / 20) are fine given patience=5; you can lower the initial cap without losing quality.

---

### Answering the question directly
- How many epochs to fine-tune? Start with 3–5 for the head, then 5–10 for FT, with EarlyStopping. Adjust based on validation trends and dataset size.
- Best techniques beyond transfer learning? Gradual unfreezing, discriminative LRs, one-cycle/cosine schedules or ReduceLROnPlateau, keep BN frozen, weight decay + dropout, label smoothing, and strong augmentations like MixUp/CutMix/RandAugment.

---

### References and further reading
- Keras: Transfer learning & fine-tuning (BN guidance, `training=False` for base during FT)
  - https://keras.io/guides/transfer_learning/
- TensorFlow: Transfer learning and fine-tuning tutorials
  - https://www.tensorflow.org/tutorials/images/transfer_learning
- fastai: Fine-tune API, gradual unfreezing, discriminative LRs, one-cycle
  - https://docs.fast.ai/learner.html#fine_tune
  - https://docs.fast.ai/callback.schedule.html#fit_one_cycle
- One-Cycle Policy
  - Smith, L.N. (2018). A disciplined approach to neural network hyper-parameters. https://arxiv.org/abs/1803.09820
- MixUp (data augmentation)
  - Zhang et al. (2018). mixup: Beyond Empirical Risk Minimization. https://arxiv.org/abs/1710.09412
- CutMix (data augmentation)
  - Yun et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. https://arxiv.org/abs/1905.04899
- RandAugment (data augmentation)
  - Cubuk et al. (2020). RandAugment: Practical automated data augmentation. https://arxiv.org/abs/1909.13719

Note: Epoch counts are guidelines; prefer validation-driven stopping. Stronger augmentations usually warrant a few more epochs (and/or a schedule) to converge.


