from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='epoch_8_result_original/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test_all/', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='1', help='which epoch to load? set to latest to use latest cached model')#latest
        parser.add_argument('--how_many', type=int, default=1000, help='how many test images to run')
        self.isTrain = False
        return parser
