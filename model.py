import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ETC_LSTM(nn.Module):
    """
    ETC-LSTM Net, The input of this net is matrix of shape (B X S X F)
    Where B is batch size, S is length of sequence (video) and F is size of features.
    Each item in the sequence representing one second.  
    """

    def __init__(
        self,
        input_size,
        hidden_dim=128,
        num_lstm_layers=1,
        dropout_prob=0.5,
        bidirectional_lstm=False,
    ):
        """
        input size: The size of features vector for each second.
        hidden_dim: The dim of the linear layer after lstm.
        num_lstm_layers: Number of lstm layers.
        dropout_prob: The probability of the dropout layer between lstm and linear layers. 
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size,
            hidden_dim,
            bidirectional=bidirectional_lstm,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout_prob)

        self.output_layer = nn.Linear(
            hidden_dim if not bidirectional_lstm else hidden_dim * 2, 2
        )

    def forward(self, padded_sentence):
        batched_samples, lengths = padded_sentence

        psq = pack_padded_sequence(
            batched_samples, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        lstm_out, _ = self.lstm(psq)
        s, _ = pad_packed_sequence(lstm_out, batch_first=True)
        s = self.dropout(s)

        return torch.sigmoid(self.output_layer(s))


class ETCFormer(nn.Module):
    """
    ETCFormer Net, The input of this net is matrix of shape (B X S X F)
    Where B is batch size, S is length of sequence (video) and F is size of features.
    Each item in the sequence representing one second.  
    """
    def __init__(
        self,
        input_size,
        transformer_layers_count=3,
        nhead=2,
        dim_feedforward=512,
        dropout=0.5,
    ):
        """
        input size: The size of features vector for each second
        dim_feedforward: The dim of the feedforward in the transformer.
        transformer_layers_count: Number transformer layers
        nhead: Number of transformer heads.
        dropout: The probability of the dropout layer between transformer and linear layers. 
        """
        super().__init__()

        self._encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size - 1,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(
            self._encoder_layer, transformer_layers_count
        )

        self.etc_progress_head = nn.Linear(input_size, 2)

    def look_ahead_mask(self, tgt_len: int, src_len: int) -> torch.FloatTensor:
        """
        This will be applied before sigmoid function, so '-inf' for proper positions needed.
        look-ahead masking is used for decoder in transformer,
        which prevents future target label affecting past-step target labels.
        """
        mask = (torch.triu(torch.ones(src_len, tgt_len)) == 1).transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, padded_sentence):
        batched_samples, lengths = padded_sentence

        src_padding_mask = torch.zeros(
            (batched_samples.shape[0], batched_samples.shape[1]),
            device=batched_samples.device,
        ).bool()

        for i in range(len(lengths)):
            src_padding_mask[i, lengths[i].int():] = True

        mask = self.look_ahead_mask(
            batched_samples.shape[1], batched_samples.shape[1]
        ).to(batched_samples.device)

        x = self.encoder(
            src=batched_samples[:, :, :-1],
            mask=mask,
            src_key_padding_mask=src_padding_mask,
        )

        s = torch.concat([x, batched_samples[:, :, -1].unsqueeze(dim=2)], dim=2)

        return torch.sigmoid(self.etc_progress_head(s))


class ETCouple(nn.Module):
    """
    The model gets "two points" in time - one at time x, and the other at x-1 (minute*).
    outputs are ETC values and the Progress values (progress act as an aux task).
    """

    def __init__(self, input_size, hidden_dim, num_lstm_layers=1, dropout_prob=0.0):
        """
        input size: The size of features vector for each second.
        hidden_dim: The dim of the linear layer after lstm.
        num_lstm_layers: Number of lstm layers.
        dropout_prob: The probability of the dropout layer between lstm and linear layers. 
        """

        super(ETCouple, self).__init__()
        self.hidden_dim = hidden_dim

        self.bi_lstm = nn.LSTM(
            input_size,
            hidden_dim,
            bidirectional=True,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout_prob)
        self.fc_prog_etc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, padded_sentence):
        batched_tuple_samples, lengths = padded_sentence
        batched_tuple_samples2 = torch.reshape(
            batched_tuple_samples,
            (
                batched_tuple_samples.shape[0] * batched_tuple_samples.shape[1],
                batched_tuple_samples.shape[2],
                batched_tuple_samples.shape[3],
            ),
        )
        lengths2 = torch.reshape(lengths, (lengths.shape[0] * lengths.shape[1],))

        psq = pack_padded_sequence(
            batched_tuple_samples2,
            lengths2.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        bi_lstm_out, _ = self.bi_lstm(psq)
        s, _ = pad_packed_sequence(bi_lstm_out, batch_first=True)
        s = self.dropout(s)
        prog_etc_output = self.fc_prog_etc(s)

        batch_indices = torch.arange(prog_etc_output[:, :, 0].shape[0]).type_as(lengths2)
        prog_output = prog_etc_output[batch_indices, lengths2 - 1, 0].squeeze()
        prog_pred = torch.sigmoid(prog_output)

        etc_output = prog_etc_output[batch_indices, lengths2 - 1, 1].squeeze()
        etc_pred = torch.sigmoid(etc_output)

        return (prog_output, etc_output), (prog_pred, etc_pred)
