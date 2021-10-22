import torch

class HParams():
    def __init__(self):
        self.encoder_len = 500
        self.decoder_len = 70
        self.max_vocab_size = 30000
        self.batch_size = 16 if torch.cuda.is_available() else 8
        self.num_layers = 6
        self.d_model = 512
        self.dff = 2048
        self.num_heads = 8
        self.dropout_rate = 0.1
        self.epochs = 100
        self.learning_rate = 1e-4
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
