import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Options():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--ModelName', help='Model Name', default='RGDNbasic')
        parser.add_argument('--UseCUDA', help='Use CUDA?', type=str2bool, nargs='?', const=True, default=True)

        parser.add_argument('--UseGradAdj',
                            help='Use grad adj module?',
                            type=str2bool,
                            nargs='?',
                            const=True,
                            default=True)
        parser.add_argument('--UseReg',
                            help='Use Reg?',
                            type=str2bool,
                            nargs='?',
                            const=True,
                            default=True)
        parser.add_argument('--UseGradScaler',
                            help='Add the grad scaler?',
                            type=str2bool,
                            nargs='?',
                            const=True,
                            default=True)

        parser.add_argument('--StepNum',
                            help='maximum number of steps',
                            type=int,
                            nargs='?',
                            const=True,
                            default=40)
        parser.add_argument('--StopEpsilon',
                            help='stopping condition',
                            type=float,
                            # default=1e-7)
                            default=float("inf"))

        # CropSize =0 when no padding applied on y in advance; -1 for padding with kernel size in advance.
        parser.add_argument('--CropSize', help='crop boundaies of results', type=int, default=-1)
        parser.add_argument('--ImgPad', help='pad image before processing', type=str2bool, default=False)
        parser.add_argument('--DataPath', help='DataPath', type=str, default='../rgdn_dataset/')
        parser.add_argument('--OutPath', help='Path for output', type=str, default='../rgdn_results/')
        parser.add_argument('--TrainedModelPath', help='path of trained model', type=str, default='./rgdn.tr')
        parser.add_argument('--Suffix', type=str, help='Manually set suffix', default='Debug')
        self.initialized = True
        self.parser = parser
        return parser

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        self.message = message

    def parse(self, is_print):
        parser = self.initialize()
        opt = parser.parse_args()
        if(is_print):
            self.print_options(opt)
        self.opt = opt
        return self.opt

