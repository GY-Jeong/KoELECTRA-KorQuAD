import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import numpy as np

from transformers import (
    BertModel,
    BertPreTrainedModel,
    ElectraModel,
    ElectraPreTrainedModel,
)

# from transformers.models.bert.modeling_bert import BertPooler
from transformers.activations import get_activation
from transformers.modeling_outputs import QuestionAnsweringModelOutput


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Returns BaseModelOutputWithPoolingAndCrossAttentions
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        # outputs[1] is pooler_output of shape (batch_size, hidden_size):
        # Last layer hidden-state of the first token of the sequence (classification token) further processed by a
        # Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
        # prediction (classification) objective during pretraining.
        pooled_output = outputs[1]
        return pooled_output


class ElectraPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, features, **kwargs):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # take <s> token (equiv. to [CLS]) hidden_states[:, 0]
        first_token_tensor = features[:, 0, :]
        first_token_tensor = self.dropout(first_token_tensor)
        first_token_tensor = self.dense(first_token_tensor)
        first_token_tensor = get_activation("gelu")(first_token_tensor)
        return first_token_tensor


class ElectraEncoder(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.pooler = ElectraPooler(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Returns BaseModelOutputWithPastAndCrossAttentions
        encoder_outputs = self.electra(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return pooled_output


class ElectraForQuestionAnsweringWeightedLoss(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)

        # TODO: Use a more complex layer
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.activation = get_activation("gelu")
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.rnn = nn.LSTM(
            input_size=config.hidden_size,  # + 1,
            hidden_size=config.hidden_size,  # // 2,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # sequence_output = discriminator_hidden_states[0]  # torch.Size([32, 440, 768])
        # logits = self.qa_outputs(sequence_output)

        # sequence_output = self.dropout(sequence_output)
        # logits = self.dense(sequence_output)
        # logits = self.activation(logits)
        # logits = self.dropout(logits)
        # logits = self.qa_outputs(logits)

        encode_layers = discriminator_hidden_states[0]
        lstm_out, (h_n, c_n) = self.rnn(
            encode_layers.permute(1, 0, 2)
        )  # h_n shape: 2*direction, batch_size, electra_hidden_size

        seq_len, batch, hs = lstm_out.shape
        lstm_out = lstm_out.view(seq_len, batch, 2, hs // 2)
        lstm_out_forward = lstm_out[:, :, 0, :]
        lstm_out_backward = lstm_out[:, :, 1, :]
        tr = (lstm_out_forward + lstm_out_backward).permute(
            1, 0, 2
        )  # 32, 440, 2*electra_hidden_size

        # h_out = torch.cat((h_n[0], h_n[1]), dim=1)  # both directions
        logits = self.qa_outputs(tr)

        start_logits, end_logits = logits.split(1, dim=-1)
        # print(start_logits.shape)  # (bs, seq_len, 1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            # Start position loss is weighted twice as end loss
            # TODO: only weigh when there is answer
            nonzero_start = torch.nonzero(start_positions, as_tuple=False).cpu()
            nonzero_end = torch.nonzero(end_positions, as_tuple=False).cpu()
            has_answer = np.intersect1d(nonzero_start, nonzero_end)

            loss_fct = CrossEntropyLossDynamic(
                ignore_index=ignored_index, answer_weight_idx=has_answer
            )
            start_loss = loss_fct(start_logits, start_positions, weigh=True)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + discriminator_hidden_states[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class CrossEntropyLossDynamic(CrossEntropyLoss):
    def __init__(
        self,
        weight=None,
        answer_weight_idx=None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.answer_weight_idx = answer_weight_idx

    def forward(self, input, target, weigh=False):
        weights = torch.ones(input.size(0)).to(device="cuda")
        if weigh:
            weights[self.answer_weight_idx] *= 0.5
        loss = F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        loss = loss * weights
        return loss.sum() / weights.sum()