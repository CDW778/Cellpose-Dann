# Cellpose + DANN 有丝分裂检测 — Agent 说明

## 本仓库是什么
在 Cellpose-SAM 分割管线上接入 **DANN 有丝分裂二分类器**
（`lillianlam/multi-stage-dann-model-mitotic-figures`：ResNet50 + 域对抗，
训练于 MIDOG++）。**零样本直接加载权重，二阶段不重训**。

流程：`cpsam_v2` 现场分割 → 每个 mask 裁 224×224 原始 RGB 窗口 →
DANN 打分 → 质心检测点 → GT 5µm 一对一贪心匹配 → F1。

完整方案与资源链接见 `DANN_plan.txt`（注意其权重路径已过期，见下）。
DANN 参考源码已在仓库内：`Biomedical-Data-Design-MIDOG-main/Phase 2/`
（`final_model.py`、`pre_processing/224_patch_around_bbox.py`），无需再联网下载。

## 待创建文件（尚未实现）
- `dann_model.py`：从 `final_model.py` **只抽** `GradientReversalFunction` /
  `GradientReversalLayer` / `DANNModel`。**不能直接 import 整个文件**：
  其顶部 `import umap, seaborn`（本环境未安装）。**勿引入 umap/seaborn/histomicstk 等重依赖。**
- `eval_2012_dann_live.py`：复用 `eval_2012_live.py` 的评估逻辑与
  `gen_dataset_2012.py` 的 `load_gt_regions` / `find_csv` / `label_bboxes`，
  把分类器换成 DANN、输入换成 224 原始窗口。

## DANN 关键规格（已从源码核对）
- 构造（必须用此 dict 才能 load state_dict；值须与 checkpoint 一致）：
  ```python
  num_domain_classes = {'Tumor': 7, 'Species': 2, 'Origin': 4, 'Scanner': 4}
  model = DANNModel(num_classes=2, num_domain_classes=num_domain_classes, lambda_val=0.0)
  model.load_state_dict(torch.load(WEIGHT, map_location="cpu"))
  model.eval()
  ```
  注意 `DANNModel.__init__` 内部调用 `models.resnet50(pretrained=True)`，本机无
  torch hub 缓存会联网下载；抽类时建议改 `pretrained=False`（权重随后被覆盖）。
- 权重（已下载，**无 `dann/` 子目录**）：
  `/home/hyw/chendawei/classification-weight/best_dann_model.pth`
- 推理：`logits = model.predict_only(x)`；
  `p_mitosis = torch.softmax(logits, dim=1)[:, 1]`。标签 0=non-mitotic, 1=mitotic
  （源码 `cat_to_label={1:1, 2:0}`）。
- 输入：224×224 **原始 RGB 原图窗口（不是白板！）**，ImageNet 归一化：
  `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`。
- 裁剪（严格照 `224_patch_around_bbox.py`）：以 bbox 中心为中心、**固定像素窗口**
  （不按细胞大小缩放）。越界时**向内平移使窗口保持在图内**（贴右/下边）：
  `if right>W: right=W; left=right-224`；`if bottom>H: bottom=H; top=bottom-224`。
  注：A 扫描仪 0.2456µm/px 与 MIDOG 量级接近，尺度大致可比。

## 环境与命令
- Python：`/home/hyw/miniforge3/envs/cellpose/bin/python`（3.8.5, torch 2.4.1+cu121,
  torchvision 0.19.1, mmpretrain 1.2.0 已装为包；skimage/cv2 有；**umap/seaborn 缺失**）。
- 必须 `KMP_DUPLICATE_LIB_OK=TRUE`。
- 仅需现有 conda 环境 + torchvision + PIL；**无需 mmdetection / Caffe / 新环境**。
- 评估（现场 Cellpose，只读）：`python eval_2012_dann_live.py ...`。
- `eval_2012_live.py` 为二阶段 ConvNeXt 版本，**仅作参考**：其 MacenkoNorm 依赖的
  `my_transforms` 随 `mmpretrain-main/` 已删除，不能直接跑。
- **长任务**：不要自己轮询。给出
  `setsid env KMP_DUPLICATE_LIB_OK=TRUE <python> <脚本> </dev/null > <log> 2>&1 &`
  以及 `tail -f <log>`，由用户手动运行/监控。仅 <~1 分钟的小检查可直接跑。

## 关键坑（已核实）
- 仓库目录是 `cellpose-DANN-opencode`，但旧脚本/文档常硬编码
  `/home/hyw/chendawei/cellpose-4.2.1.1-opencode`（该目录仍存在）。直接照抄会踩坑，
  新代码请用本目录相对/绝对路径。
- 数据/权重**在仓库外**，不要混进代码目录：
  - test 原图：`/home/hyw/dataset/chendawei/train-classification/ICPR-2012/train-val-test/test-original-img/*.bmp`
  - Cellpose 缓存（RGB）：`/home/hyw/dataset/chendawei/train-classification/ICPR-2012/cellpose_cache/test_masks_cp3/*_mask.npy`
  - GT CSV：`/home/hyw/dataset/chendawei/ICPR2012/{A,H}0?_v2/{base}.csv`
  - Cellpose 权重：`/home/hyw/chendawei/cellpose-weight/cpsam_v2`
- 分辨率：A=0.2456 µm/px，H=(0.2273+0.22753)/2 µm/px。
- `cache_cellpose_2012.py` 可重生成 mask 缓存；`gen_dataset_2012.py` 的工具函数可复用。
- 旧的 ConvNeXt 二阶段框架（`mmpretrain-main/`、其 configs/脚本/HANDOVER 文档）已被删除，
  不要假设这些文件存在。

## 评估口径（唯一判据）
- **唯一判据：test slide04 的端到端 5µm F1。patch 级指标不作数。**
- 阈值扫描 0.05–0.95，报告 oracle 最优。
- 只在我们 train（00-02）上训练/微调；test（04）仅评估（无泄漏）。
- 域偏移风险：MIDOG++ → ICPR2012 零样本效果未知，必须实测；
  若差可在 train 上微调（头/LoRA/全量）或冻结 backbone 只训新头。
