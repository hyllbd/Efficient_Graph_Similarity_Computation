import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run SimGNN.")

    parser.add_argument("--dataset",
                        nargs="?",
                        default="AIDS700nef",
                        help="Dataset name. Default is AIDS700nef")

    parser.add_argument("--gnn-operator",
                        nargs="?",
                        default="gin",
                        help="Type of GNN-Operator. Default is gcn")

    parser.add_argument("--gnn-operator-fix",
                        nargs="?",
                        default="gin",
                        help="Type of GNN-Operator-fix. Default is gin")

    parser.add_argument("--epochs",
                        type=int,
                        default=350,
                        help="Number of training epochs. Default is 350.")

    parser.add_argument("--filters-1",
                        type=int,
                        default=64,
                        help="Filters (neurons) in 1st convolution. Default is 64.")

    parser.add_argument("--filters-2",
                        type=int,
                        default=32,
                        help="Filters (neurons) in 2nd convolution. Default is 32.")

    parser.add_argument("--filters-3",
                        type=int,
                        default=16,
                        help="Filters (neurons) in 3rd convolution. Default is 16.")
    parser.add_argument("--filters-4",
                        type=int,
                        default=8,
                        help="Filters (neurons) in 3rd convolution. Default is 16.")

    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
                        help="Neurons in tensor network layer. Default is 16.")

    parser.add_argument("--bottle-neck-neurons",
                        type=int,
                        default=16,
                        help="Bottle neck layer neurons. Default is 16.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--bins",
                        type=int,
                        default=16,
                        help="Similarity score bins. Default is 16.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0,
                        help="Dropout probability. Default is 0.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5 * 10 ** -4,
                        help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--histogram",
                        dest="histogram",
                        action="store_true")

    parser.add_argument("--diffpool",
                        dest="diffpool",
                        action="store_true",
                        help="Enable differentiable pooling.")

    parser.add_argument("--plot",
                        dest="plot",
                        action="store_true")

    parser.add_argument("--load_pretrain",
                        dest="load",
                        action="store_true")

    parser.add_argument("--synth",
                        dest="synth",
                        action="store_true")

    parser.add_argument("--measure-time",
                        action="store_true",
                        help="Measure average calculation time for one graph pair. Learning is not performed.")

    parser.add_argument("--notify",
                        dest="notify",
                        action="store_true",
                        help="Send notification message when the code is finished (only Linux & Mac OS support).")

    parser.add_argument("--mode",
                            default="l1",
                            help="rkd_dis, l1, both")
    parser.add_argument("--kd_joint_mode",
                            default="l1",
                            help="rkd_dis, l1, both")

    parser.add_argument("--kd_layer",
                            default="c0",
                            help="c0, c1")

    parser.add_argument("--d-critic",
                            type=int,
                            default=1,
                            help="discriminator critic")

    parser.add_argument("--g-critic",
                            type=int,
                            default=1,
                            help="generator critic")
    
    parser.add_argument("--use-adversarial",
                            type=int,
                            default=1,
                            help="use adversarial")

    parser.add_argument("--wandb",
                            type=int,
                            default=1,
                            help="use wandb")

    # parser.add_argument("--stu_loss_type",
    #                         default="all",
    #                         help="loss type")

    parser.add_argument("--adversarial_ouput_class",
                            type=int,
                            default=16,
                            help="prediction output class in adversarial network")

    parser.add_argument("--cuda-id",
                            type=int,
                            default=0,
                            help="cuda device id")

    parser.add_argument("--light",
                            type=int,
                            default=0,
                            help="use light model")
    
    parser.add_argument("--test-interval",
                            type=int,
                            default=50,
                            help="test interval")
    
    parser.add_argument("--checkpoint",
                            type=str,
                            default=None,
                            help="checkpint path for knowledge distillation")

    parser.set_defaults(histogram=False)
    parser.set_defaults(diffpool=False)
    parser.set_defaults(plot=False)
    parser.set_defaults(load_pretrain=True)
    parser.set_defaults(measure_time=False)
    parser.set_defaults(notify=False)
    parser.set_defaults(synth=False)

    return parser.parse_args()
