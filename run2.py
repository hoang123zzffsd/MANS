import torch
import mmns
from mmns.config import Trainer, Tester
from mmns.module.model import MMRotatE
from mmns.module.loss import MarginLoss
from mmns.module.strategy import NegativeSampling
from mmns.data import TrainDataLoader, TestDataLoader
from args import get_args

if __name__ == "__main__":
    args = get_args()
    print(args)
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="/content/MANS/benchmarks/" + args.dataset + '/',
        nbatches=args.num_batch,
        threads=8,
        # 当dismult的时候是cross
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=args.neg_num,
        neg_rel=0
    )

    # dataloader for test
    test_dataloader = TestDataLoader("/content/MANS/benchmarks/" + args.dataset + '/', "link")
    img_emb = torch.load('/content/MANS/visual/' + args.dataset + '-visual.pth')
    if args.kernel == 'rotate':
        # define the model
        rotate = MMRotatE(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim=128,
            img_dim=args.img_dim,
            img_emb=img_emb,
            test_mode=args.test_mode,
            beta=args.beta
        )
        print(rotate)
        # define the loss function
        model = NegativeSampling(
            model=rotate,
            loss=MarginLoss(margin=args.margin),
            batch_size=train_dataloader.get_batch_size(),
            neg_mode=args.neg_mode
        )

        # train the model
        trainer = Trainer(
            model=model, 
            data_loader=train_dataloader,
            train_times=args.epoch, 
            alpha=1.0, 
            use_gpu=True,
            opt_method='Adam', 
            train_mode=args.train_mode
        )
        trainer.run()
        rotate.save_checkpoint(args.save)
        # test the model
        rotate.load_checkpoint(args.save)
        tester = Tester(model=rotate, data_loader=test_dataloader, use_gpu=True)
        # link prediction task
        tester.run_link_prediction(type_constrain=False)
        # triple classification task
        acc, p, r, f, _ = tester.run_triple_classification_four_metrics()
        print(acc, p, r, f)
    else:
        raise NotImplementedError
