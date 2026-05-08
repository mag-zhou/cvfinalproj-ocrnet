# Mod 4 (CBL-lite on OCR features) at 50% of ADE20K.
_base_ = ['./ocrnet_r50_mod4_cbl.py']

# Linear LR scaling consistent with 50pct baseline (0.002 -> 0.004).
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0005))

train_dataloader = dict(dataset=dict(indices=10105))
