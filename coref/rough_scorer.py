""" Describes RoughScorer, a simple bilinear module to calculate rough
anaphoricity scores.
"""

from typing import Tuple

import torch

from coref.config import Config


class RoughScorer(torch.nn.Module):
    """
    Is needed to give a roughly estimate of the anaphoricity of two candidates,
    only top scoring candidates are considered on later steps to reduce
    computational complexity.
    """
    def __init__(self, features: int, config: Config):
        super().__init__()
        self.dropout = torch.nn.Dropout(config.dropout_rate)
        self.bilinear = torch.nn.Linear(features, features)

        self.k = config.rough_k

    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                mentions: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns rough anaphoricity scores for candidates, which consist of
        the bilinear output of the current model summed with mention scores.
        """
        # [n_mentions, n_mentions]
        pair_mask = torch.arange(mentions.shape[0])
        pair_mask = pair_mask.unsqueeze(1) - pair_mask.unsqueeze(0)
        pair_mask = torch.log((pair_mask > 0).to(torch.float))
        pair_mask = pair_mask.to(mentions.device)

        bilinear_scores = self.dropout(self.bilinear(mentions)).mm(mentions.T)

        rough_scores = pair_mask + bilinear_scores

        return self._prune(rough_scores)

    def _prune(self,
               rough_scores: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects top-k rough antecedent scores for each mention.

        Args:
            rough_scores: tensor of shape [n_mentions, n_mentions], containing
                rough antecedent scores of each mention-antecedent pair.

        Returns:
            FloatTensor of shape [n_mentions, k], top rough scores
            LongTensor of shape [n_mentions, k], top indices
        """
        top_scores, indices = torch.topk(rough_scores,
                                         k=min(self.k, len(rough_scores)),
                                         dim=1, sorted=False)
        return top_scores, indices


class IncrementalRoughScorer(torch.nn.Module):
    """
    Is needed to give a roughly estimate of the anaphoricity of two candidates,
    only top scoring candidates are considered on later steps to reduce
    computational complexity.
    """
    def __init__(self, features: int, config: Config):
        super().__init__()
        self.dropout = torch.nn.Dropout(config.dropout_rate)
        self.bilinear = torch.nn.Linear(features, features)

        self.k = config.rough_k

    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                mentions: torch.Tensor,
                first: bool
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns rough anaphoricity scores for candidates, which consist of
        the bilinear output of the current model summed with mention scores.
        """
        # [n_mentions, n_mentions]
        if first:
            pair_mask = torch.arange(mentions.shape[0])
            pair_mask = pair_mask.unsqueeze(1) - pair_mask.unsqueeze(0)
            pair_mask = torch.log((pair_mask > 0).to(torch.float))
            pair_mask = pair_mask.to(mentions.device)

            self.window_scores = self.dropout(self.bilinear(mentions))

            bilinear_scores = self.window_scores.mm(mentions.T)
            self.bilinear_scores = bilinear_scores

            rough_scores = pair_mask + bilinear_scores

            return self._prune(rough_scores)
        else:
            pair_mask = torch.arange(mentions.shape[0])
            pair_mask = pair_mask.unsqueeze(1) - pair_mask.unsqueeze(0)
            pair_mask = torch.log((pair_mask > 0).to(torch.float))
            pair_mask = pair_mask.to(mentions.device)

            # [1, features]
            new_mention_scores = self.dropout(self.bilinear(mentions[-1, :]))

            # [window_size, features]
            self.window_scores = torch.vstack((self.window_scores[1:, :], new_mention_scores))

            # [1, window_size - 1]
            new_bilinear_scores = self.window_scores[1:, :].matmul(new_mention_scores.T)

            # [1, window_size]
            new_bilinear_scores = torch.cat(
                (
                    new_bilinear_scores,
                    torch.Tensor(-torch.inf)
                )
            )

            # [window_size, window_size]
            self.bilinear_scores = torch.vstack((self.bilinear_scores[1:, :], new_bilinear_scores))

            rough_scores = pair_mask + self.bilinear_scores

            return self._prune(rough_scores)

    def _prune(self,
               rough_scores: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects top-k rough antecedent scores for each mention.

        Args:
            rough_scores: tensor of shape [n_mentions, n_mentions], containing
                rough antecedent scores of each mention-antecedent pair.

        Returns:
            FloatTensor of shape [n_mentions, k], top rough scores
            LongTensor of shape [n_mentions, k], top indices
        """
        top_scores, indices = torch.topk(rough_scores,
                                         k=min(self.k, len(rough_scores)),
                                         dim=1, sorted=False)
        return top_scores, indices
