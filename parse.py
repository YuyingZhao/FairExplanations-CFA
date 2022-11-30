import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--no-cuda', action='store_true', default=False,
	                    help='Disables CUDA training.')
	parser.add_argument('--cuda_device', type=int, default=0,
	                    help='cuda device running on.')
	parser.add_argument('--dataset', type=str, default='bail',
	                    help='a dataset from credit, german and bail.')
	parser.add_argument('--epochs', type=int, default=1000,
	                    help='Number of epochs to train.')
	parser.add_argument('--lr', type=float, default=1e-3,
	                    help='Initial learning rate.')
	parser.add_argument('--weight_decay', type=float, default=1e-5,
	                    help='Weight decay (L2 loss on parameters).')
	parser.add_argument('--hidden', type=int, default=8,
	                    help='Number of hidden units.')
	parser.add_argument('--dropout', type=float, default=0.3,
	                    help='Dropout rate (1 - keep probability).')
	parser.add_argument('--seed', type=int, default=0,
	                    help='seed.')
	parser.add_argument('--model', type=str, default='mlp',
	                    help='mlp model.')
	parser.add_argument('--topK', type=int, default=1,
	                    help='features to be masked when computing fidelity.')
	parser.add_argument('--lambda_', type=float, default=1.0,
	                    help='lambda_: coefficient for fairness loss')
	parser.add_argument('--top_ratio', type=float, default=0.2,
	                    help='top_ratio.')
	parser.add_argument('--opt_start_epoch', type=int, default=400,
                    	help='the epoch we start optimization')
	return parser.parse_args()