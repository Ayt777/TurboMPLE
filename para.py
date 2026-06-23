import argparse


class Parameter:
    def __init__(self):
        self.args = self.set_args()

    def set_args(self):
        self.parser = argparse.ArgumentParser(description='TurboMPLE')

        # Global parameters
        self.parser.add_argument('--seed', type=int, default=40, help='random seed')
        self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
        self.parser.add_argument('--mode', type=str, default='formal', help='continue of formal')
        self.parser.add_argument('--model_name', type=str, default='TurboMPLE')

        # Data parameters
        self.parser.add_argument('--frame_length', type=int, default=30, help='length of one sequence')
        self.parser.add_argument('--save_dir', type=str, default='./model/experiment/', help='directory to save the models')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='directory to save the results')
        self.parser.add_argument('--data_root', type=str, default='data_new', help='the path of dataset')

        # Training parameters
        self.parser.add_argument('--start_epoch', type=int, default=1, help='first epoch number')
        self.parser.add_argument('--end_epoch', type=int, default=100, help='last epoch number')

        args = self.parser.parse_args()

        return args
