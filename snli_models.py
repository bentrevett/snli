import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

class StandardRNN(nn.Module):
    """
    Standard bidirectional LSTM
    """
    def __init__(self, vocab_size, embedding_dim, translation_dim, rnn_type, hidden_dim, n_layers, bidirectional, dropout, output_dim):
        super(StandardRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.translation = TimeDistributed(nn.Linear(embedding_dim, translation_dim))

        #need to handling lstm output in the forward method
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout)
        else:
            raise ValueError(f'rnn_type must be GRU or LSTM! Got {rnn_type}')

        n_directions = 2 if bidirectional else 1

        self.fc1 = nn.Linear(hidden_dim*n_directions*2, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*2)

        self.dropout = nn.Dropout(dropout) #every time dropout is used in the forward method it gets a new mask

        self.bn1 = nn.BatchNorm1d(hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.bn3 = nn.BatchNorm1d(hidden_dim*2)
        self.bn4 = nn.BatchNorm1d(hidden_dim*2)
        self.bn5 = nn.BatchNorm1d(hidden_dim*2)

        self.out = nn.Linear(hidden_dim*2, output_dim)

        #batchnorm before or after relu? nobody knows!
        #https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/

        #http://forums.fast.ai/t/questions-about-batch-normalization/230
        #relu -> bn -> dropout

        #keras implementation has 15.7m params, if we have lots more, try doing affine=False for the batchnorm layer
        #or try layernorm

        #https://discuss.pytorch.org/t/example-on-how-to-use-batch-norm/216/24
        #IF USING RELU, DO DROPOUT THEN RELU, NOT OTHER WAY AROUND!

        #https://ikhlestov.github.io/pages/machine-learning/pytorch-notes/#weights-initialization

        """#set these as attributes as we need in the forward method
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim

        #linear input size is num_directions * hidden_dim
        fc_inp = 2 * hidden_dim if bidirectional else hidden_dim 
        self.fc = nn.Linear(fc_inp, output_dim)

        #layer normalization
        self.ln = LayerNorm(hidden_dim)

        self.do = nn.Dropout(dropout)"""

    def forward(self, sent1, sent2):

        #print("sent1",sent1.shape)

        #all the inputs are [seq. len, bsz], i'm pretty sure embedding layer wants them [bsz, seq. len]
        sent1 = sent1.permute(1, 0)
        sent2 = sent2.permute(1, 0)

        #print("sent1",sent1.shape)

        emb_sent1 = self.embedding(sent1)
        emb_sent2 = self.embedding(sent2)

        #print("emb_sent1",emb_sent1.shape)

        translated_sent1 = self.translation(emb_sent1)
        translated_sent2 = self.translation(emb_sent2)

        #print("translated_sent1",translated_sent1.shape)

        translated_sent1 = translated_sent1.permute(1, 0, 2)
        translated_sent2 = translated_sent2.permute(1, 0, 2)

        if self.rnn_type == 'LSTM':
            o, (rnn_sent1, _) = self.rnn(translated_sent1)
            _, (rnn_sent2, _) = self.rnn(translated_sent2)
        else:
            _, rnn_sent1 = self.rnn(translated_sent1)
            _, rnn_sent2 = self.rnn(translated_sent2)

        #print("rnn_sent1",rnn_sent1.shape)

        #print(rnn_sent1[0,0,:]) #fwd
        #print(rnn_sent1[1,0,:]) #bkd

        rnn_sent1 = self.bn1(torch.cat((rnn_sent1[-2,:,:], rnn_sent1[-1,:,:]),dim=1))

        rnn_sent2 = self.bn2(torch.cat((rnn_sent2[-2,:,:], rnn_sent2[-1,:,:]),dim=1))

        sents = self.dropout(torch.cat((rnn_sent1, rnn_sent2), dim=1))

        sents = self.dropout(self.bn3(F.relu(self.fc1(sents))))

        sents = self.dropout(self.bn4(F.relu(self.fc2(sents))))

        sents = self.dropout(self.bn4(F.relu(self.fc3(sents))))

        output = self.out(sents)

        """
        #either concat or do it w/ permute and view
        rnn_sent1 = rnn_sent1.permute(1, 0, 2).contiguous()
        
        rnn_sent1 = rnn_sent1.view(rnn_sent1.shape[0], -1)

        print("rnn_sent1",rnn_sent1.shape)

        print("***")
        print(rnn_sent1[0,:300]) #fwd
        print(_x[0,:300]) #SAME AS ABOVE
        print("***")
        print(rnn_sent1[0,300:]) #bkd
        print(_x[0,300:]) #SAME AS ABOVE"""

        """
        #concat them all together
        emb = torch.cat((emb_s, emb_e1, emb_e2), dim=1)

        #reshape as need to be [seq. len, bsz, emb. dim] for the rnn
        emb = self.do(emb.permute(1, 0, 2))

        if self.rnn_type == 'LSTM':
            o, (h, _) = self.rnn(emb)
        else:
            o, h = self.rnn(emb)

        #   h is [num dir * num layer, bsz]
        #   the first dim of h goes [layer1 forward, layer1 backward, layer2 forward, layer2 backward]
        #   top layer forward == h[-2,:,:], top layer backward == h[-1,:,:]
        #   so to get the final forward+backward hidden, use h[-2:,:,:]

        #   o is [seq len, bsz, hid_dim * bum dir]
        #   last dim of o is cat(forward, backward), so o[-1,:,:hid dim] is the final forward hidden state, equal to h[-2,:,:]
        #   to get the final backward state, you need to get the first element of the seq. len and the last hid dim of the final dimension
        #   i.e. o[0, :, hid dim:], which equals h[-1,:,:]

        #assert torch.equal(o[0, :, self.hidden_dim:], h[-1, :, :])
        #assert torch.equal(o[-1, :, :self.hidden_dim], h[-2, :, :])

        h = self.do(self.ln(h[-2:,:,:].permute(1, 0, 2).contiguous()))
        h = h.view(h.shape[0], -1)

        output = self.fc(h)"""

        return output


"""
STOLEN FROM ALLENNLP
"""
class TimeDistributed(torch.nn.Module):
    """
    Given an input shaped like ``(batch_size, time_steps, [rest])`` and a ``Module`` that takes
    inputs like ``(batch_size, [rest])``, ``TimeDistributed`` reshapes the input to be
    ``(batch_size * time_steps, [rest])``, applies the contained ``Module``, then reshapes it back.
    Note that while the above gives shapes with ``batch_size`` first, this ``Module`` also works if
    ``batch_size`` is second - we always just combine the first two dimensions, then split them.
    """
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self._module = module

    def forward(self, *inputs):  # pylint: disable=arguments-differ
        reshaped_inputs = []
        for input_tensor in inputs:
            input_size = input_tensor.size()
            if len(input_size) <= 2:
                raise RuntimeError("No dimension to distribute: " + str(input_size))

            # Squash batch_size and time_steps into a single axis; result has shape
            # (batch_size * time_steps, input_size).
            squashed_shape = [-1] + [x for x in input_size[2:]]
            reshaped_inputs.append(input_tensor.contiguous().view(*squashed_shape))

        reshaped_outputs = self._module(*reshaped_inputs)

        # Now get the output back into the right shape.
        # (batch_size, time_steps, [hidden_size])
        new_shape = [input_size[0], input_size[1]] + [x for x in reshaped_outputs.size()[1:]]
        outputs = reshaped_outputs.contiguous().view(*new_shape)

        return outputs

    
