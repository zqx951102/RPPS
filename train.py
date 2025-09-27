import argparse
import datetime
import os.path as osp
import time
import torch
import torch.utils.data
import torch.nn.functional as F
from datasets import PersonSearchUDADataMoudle   #这里是加载 数据prw和cuhk的数据
from defaults import get_default_cfg
from engine import evaluate_performance, train_one_epoch_da
from models.seqnet_da import SeqNetDa  #引入SeqNetDa
from models.cpm import ClusterProxyMemory  #在这里引包



from utils import (
    mkdir,
    resume_from_ckpt,  #断点训练的 ckpt
    save_on_master,
    set_random_seed,
    generate_pseudo_labels,
    generate_cluster_features,
    generate_class_features,
)
from apex import amp
from spcl.models.dsbn import convert_dsbn  #DSBN 是为了让源域和目标域分别拥有独立的归一化路径，从而缓解跨域特征分布偏移的问题。
from spcl.utils.faiss_rerank import compute_jaccard_distance
from spcl.evaluators import extract_dy_features
from sklearn.cluster import DBSCAN  #聚类 DBSCAN 是无监督密度聚类方法，用于目标域伪标签生成


def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()  #cfg.freeze() 锁定配置，防止后续意外更改
    print(cfg)
    device = torch.device(cfg.DEVICE)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)  #设置随机种子以确保实验可复现

    print("Creating model and convert dsbn")
    model = SeqNetDa(cfg)  #实例化模型
    """用于在模型中切换批归一化和域自适应归一化层之间"""
    convert_dsbn(model.roi_heads.reid_head)  # 将 reid head 的普通 BN 转为 DSBN 以适配源/目标域
    model.to(device)   #参数放在GPU上
    print(model)
    # build dataset module
    DataMoudle = PersonSearchUDADataMoudle(cfg)  #加载跨域行人搜索的数据模块，统一管理数据加载逻辑

    if args.eval:  #如果是评估模式 就执行 如果提供 --eval 参数，则加载模型并对测试集执行评估流程
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(args.ckpt, model)
        DataMoudle.setup(stage="test")
        gallery_loader, query_loader = DataMoudle.test_dataloader()
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
        )
        exit(0)

    # build source predict dataloader  设置数据模块为 predict 模式
    DataMoudle.setup(stage="predict")
    predict_loader_source = DataMoudle.predict_dataloader(is_source=True)
    # init source domian identity level centroid
    print("==> Initialize source-domain class centroids in the hybrid memory")
    sour_fea_dict = extract_dy_features(cfg, model, predict_loader_source, device, is_source=True) #提取源域特征
    source_centers = generate_class_features(sour_fea_dict)  #聚合成每类一个向量表示（类中心）
    print(f"source_centers length: {len(source_centers)}")
    print("the last one is the feature of 5555, remember don't use it")

    # build cluster memory 构建 ClusterProxyMemory
    source_classes = DataMoudle.dataset_source_train.num_train_pids

    memory = ClusterProxyMemory(    #初始化 Hybrid Memory     设置初始为源域类中心（包括虚拟 ID 5555）  特征维度是 256
        256, source_classes, source_classes, temp=0.05, momentum=cfg.MODEL.UPDATE_FACTOR.ONLINE
    ).to(device)

    memory.features = source_centers.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(  #用 SGD 初始化优化器
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    model.roi_heads.memory = memory
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O1") ##cuhk时候选择1
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")    ##Prw时候选择1   amp.initialize 启用混合精度
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1)   #学习率衰减策略使用 Milestone

    start_epoch = 0
    if args.resume:  #如果设置了 resume，恢复训练状态（模型、优化器、学习率调度器）
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, model, optimizer, lr_scheduler) + 1

    print("Creating output folder") #日志与 TensorBoard 设置
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir) #创建输出目录；
    path = osp.join(output_dir, "config.yaml")
    target_start_epoch = cfg.TARGET_REID_START
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path) # 如果设置了 TensorBoard 则初始化其目录。
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training")
    start_time = time.time()
    #开始训练主循环
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        if epoch == target_start_epoch:  #到达指定 epoch 后，开始目标域伪标签生成
            # DBSCAN cluster
            eps = 0.5  #eps=0.5 是 DBSCAN 中的半径阈值
            print(f"Clustering criterion: eps: {eps:.3f}")
            cluster = DBSCAN(eps=eps, min_samples=4, metric="precomputed", n_jobs=-1)  #创建 DBSCAN 聚类器，用于后续伪标签生成。 metric="precomputed" 表示我们后面会提供距离矩阵而不是原始数据

        if epoch >= target_start_epoch:  #如果当前训练轮次超过或等于 target_start_epoch，就开始对目标域样本进行特征提取、伪标签生成和聚类
            # init target domain instance level features
            # we can't use target domain GT detection box feature to init, this is only for measuring the upper bound of cluster performance
            # for dynamic clustering method, we use the proposal after several epoches for first init, moreover, we'll update the memory with proposal before each epoch
            print("==> Initialize target-domain instance features in the hybrid memory")
            DataMoudle.setup(stage="predict") #设置数据加载器进入 “预测” 模式，加载目标域数据
            tgt_cluster_loader = DataMoudle.predict_dataloader(is_source=False)   #tgt_cluster_loader 用于之后的目标域特征提取
            if epoch == target_start_epoch:  #如果是第一次初始化，就用模型从目标域提取实例级别的特征（包括图像、候选框、负样本、正样本特征）
                target_features, img_proposal_boxes, negative_fea, positive_fea = extract_dy_features(
                    cfg, model, tgt_cluster_loader, device, is_source=False
                )
            else:
                if args.resume and epoch == start_epoch:  #如果不是第一次聚类，则使用之前提取过的目标域特征和候选框，使用 momentum 对其更新（带动量的特征更新）
                    target_features, img_proposal_boxes, negative_fea, positive_fea = extract_dy_features(
                        cfg, model, tgt_cluster_loader, device, is_source=False
                    )
                else:  #这是动态聚类设计的一部分，防止每次都重新提取特征而导致不稳定
                    target_features, img_proposal_boxes, negative_fea, positive_fea = extract_dy_features(
                        cfg,
                        model,
                        tgt_cluster_loader,
                        device,
                        is_source=False,
                        memory_proposal_boxes=img_proposal_boxes,
                        memory_target_features=target_features,
                        momentum=cfg.MODEL.UPDATE_FACTOR.OFFLINE,
                    )
            sorted_keys = sorted(target_features.keys())  #将目标域特征按照 key 排序后拼接
            print("target_features instances :" + str(len(sorted_keys)))
            target_features = torch.cat([target_features[name] for name in sorted_keys], 0)
            target_features = F.normalize(target_features, dim=1).to(device)  #并将特征进行 L2 归一化后传入 GPU
            #处理 hard negative 样本（目标域中容易与其他样本混淆的特征）
            negative_fea = torch.cat([negative_fea[name] for name in sorted(negative_fea.keys())], 0)
            negative_fea = F.normalize(negative_fea, dim=1).to(device)  #归一化并移入设备
            print("hard negative instances :" + str(len(negative_fea)))

            # Calculate distance  计算基于 Jaccard 距离的重排序距离矩阵（适用于跨域 ReID 伪标签生成）
            rerank_dist = compute_jaccard_distance(target_features, k1=30, k2=6, search_option=3, use_float16=True)
            pseudo_labels = cluster.fit_predict(rerank_dist)  #使用 DBSCAN 聚类器对距离矩阵进行聚类，生成伪标签（其中 -1 表示离群点）
            num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)  #计算聚类后得到的有效类数量
            print(f"pseudo_labels length :{len(pseudo_labels)}")
            # merge source dataset and target dataset, set pseudo_labels after source domain
            pseudo_labels = generate_pseudo_labels(pseudo_labels, source_classes, num_ids)  #生成目标域新伪标签，并加上源域类数偏移，避免伪标签编号冲突
            print("==> Modifying labels in target domain to build new training set")
            DataMoudle.reset_dataset_with_pseudo_labels(img_proposal_boxes, sorted_keys, pseudo_labels)  #将目标域的新伪标签赋值到数据集中，形成新的训练数据用于下一个 epoch
            # re-intalizating features memory   生成每个聚类的中心（cluster 特征）并更新 memory

            cluster_features = generate_cluster_features(pseudo_labels, target_features, source_classes)
            print(f"cluster_features length: {len(cluster_features)}")
            source_centers = memory.features[0:source_classes].clone()
            memory.features = torch.cat((source_centers, cluster_features), dim=0).to(device)
            memory.features = torch.cat((memory.features, negative_fea), dim=0).to(device)
            memory.num_samples = memory.features.shape[0]  #memory 中前 source_classes 是源域特征，后面是目标域聚类中心，最后是 hard negatives

            print(f"total features length: {len(memory.features)}")
            target_features = target_features.cpu().clone()
        else: #如果还未开始聚类，memory 中只有源域特征
            memory.num_samples = source_classes



        DataMoudle.setup(stage="train")  #设置为训练阶段，获取源域和目标域的训练数据加载器
        train_loader_s, train_loader_t = DataMoudle.train_dataloader()

        train_one_epoch_da(cfg, model, optimizer, train_loader_s, train_loader_t, device, epoch, tfboard) #执行一轮训练，调用训练函数 train_one_epoch_da
        lr_scheduler.step() #更新学习率（step LR scheduler）

        if (epoch + 1) % cfg.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:  #如果到达评估周期或最后一轮，进行模型评估 使用 gallery 和 query 计算指标（如 mAP, top1）
            DataMoudle.setup(stage="test")
            gallery_loader, query_loader = DataMoudle.test_dataloader()
            evaluate_performance(
                model,
                gallery_loader,
                query_loader,
                device,
                use_gt=cfg.EVAL_USE_GT,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
            )

        if (epoch + 1) % cfg.CKPT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:  #	到达保存周期或最后一轮时保存模型、优化器、AMP等信息为 checkpoint
            save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "amp": amp.state_dict(),
                },
                osp.join(output_dir, f"epoch_{epoch}.pth"),
            )

    if tfboard: #若启用 TensorBoard，关闭文件句柄
        tfboard.close()
    total_time = time.time() - start_time  #打印整个训练过程所花费的时间
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")


if __name__ == "__main__":    #使用 argparse 解析命令行参数（配置文件路径、是否评估、是否 resume、权重路径等）
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to configuration file.",
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate the performance of a given checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Resume from the specified checkpoint.")
    parser.add_argument(
        "--ckpt",
        help="Path to checkpoint to resume or evaluate.",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line")
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()

    main(args)  #调用 main(args) 开始训练
