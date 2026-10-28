from pathlib import Path
from cellpose import io
from cellpose import models
'''训练自己的cellpose'''
train_img_dir = Path("dataset/train/images")
train_mask_dir = Path("dataset/train/masks")

val_img_dir = Path("dataset/val/images")
val_mask_dir = Path("dataset/val/masks")

train_images = sorted(list(train_img_dir.glob("*")))
train_masks = sorted(list(train_mask_dir.glob("*_masks.*")))

val_images = sorted(list(val_img_dir.glob("*")))
val_masks = sorted(list(val_mask_dir.glob("*_masks.*")))

train_data = [io.imread(str(f)) for f in train_images]
train_labels = [io.imread(str(f)) for f in train_masks]

val_data = [io.imread(str(f)) for f in val_images]
val_labels = [io.imread(str(f)) for f in val_masks]

print(f"Train images: {len(train_data)}")
print(f"Validation images: {len(val_data)}")


model = models.CellposeModel(
    gpu=True,
    pretrained_model="/home/hyw/chendawei/cellpose-weight/cpsam_v2"
)

save_path = "./my_models"

model.train(
    train_data=train_data,
    train_labels=train_labels,

    test_data=val_data,
    test_labels=val_labels,

    channels=[0, 0],

    learning_rate=5e-5,
    weight_decay=0.1,

    n_epochs=200,
    batch_size=8,

    model_name="my_cellpose_sam",
    save_path=save_path
)