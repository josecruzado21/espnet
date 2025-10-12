import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked
from torch import Tensor
import pdb 
import editdistance
from itertools import groupby
import time
import Levenshtein

class DROCTCLoss(torch.nn.Module):
    def __init__(self, blank=0, reduction='mean', zero_infinity=False, dro_group_count=0, dro_step_size=0.01, dro_q_epsilon=1e-10,
    accumulation=False, smoothing=0, agg="sum", normalize_grad=True):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.dro_group_count = dro_group_count
        self.dro_step_size = dro_step_size

        self.dro_q = torch.ones(self.dro_group_count) * 1.0/self.dro_group_count
        self.dro_q_epsilon = dro_q_epsilon
        # self.cers = torch.ones(self.dro_group_count) # JC TO DO: Try different initialization
        self.group_id_to_ix = {}
        self.agg = agg
        self.normalize_grad = normalize_grad

        self.accumulation = accumulation
        self.smoothing = smoothing

    def init_weights(self, train_file, valid_file):
        self.utt2category = {}
        with open(str(train_file) + '/utt2category', 'r') as f:
            for line in f:
                line = line.strip().split()
                self.utt2category[line[0]] = line[1]

        # Also load mappings for test and dev
        with open(str(valid_file) + '/utt2category', 'r') as f:
            for line in f:
                line = line.strip().split()
                self.utt2category[line[0]] = line[1]

        if self.accumulation:
            # Get unique categories directly from utt2category
            unique_categories = set(self.utt2category.values())
            
            # We can also pre-populate group_id_to_ix for deterministic indexing
            for i, category in enumerate(sorted(unique_categories)):
                self.group_id_to_ix[category] = i
            
            # Initialize group_losses with correct number of groups
            self.group_losses = {}
            for i in range(len(unique_categories)):
                self.group_losses[i] = []

    def forward(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, utt_id: List[str], valid: bool = True) -> Tensor:
        log_probs = torch.transpose(log_probs, 0, 1)

        batch_lang_ids = [self.utt2category[_] for _ in utt_id]

        batch_lang_q_indices = []
        for lang_id in batch_lang_ids:
            if lang_id not in self.group_id_to_ix:
                self.group_id_to_ix[lang_id] = len(self.group_id_to_ix)
            batch_lang_q_indices.append(self.group_id_to_ix[lang_id])
        ix_to_group_id = {index: lang_name for lang_name, index in self.group_id_to_ix.items()}
        losses = F.ctc_loss(
            log_probs, 
            targets, input_lengths, target_lengths, 
            self.blank, reduction='none',
            zero_infinity=self.zero_infinity
        )

        # Beginning of code for CER calculation
        beginning_time_cer = time.time()
        ys_hat = log_probs.argmax(dim=-1).transpose(0, 1)

        per_utt_cer_stats = []
        per_utt_hyp_ref = []

        for i in range(len(losses)):
            y_hat = ys_hat[i][:input_lengths[i]]
            y_true = targets[i][:target_lengths[i]]
            
            y_hat_collapsed = [x[0] for x in groupby(y_hat.cpu().numpy())]
            y_true_numpy = y_true.cpu().numpy()
            
            seq_hat, seq_true = [], []
            
            for idx in y_hat_collapsed:
                idx = int(idx)
                if idx != -1 and idx != self.blank:
                    seq_hat.append(idx)

            for idx in y_true_numpy:
                idx = int(idx)
                if idx != -1 and idx != self.blank:
                    seq_true.append(idx)
            
            hyp_chars = seq_hat
            ref_chars = seq_true
            
            per_utt_hyp_ref.append((hyp_chars, ref_chars))
            
            if len(ref_chars) > 0:
                ops = Levenshtein.opcodes(hyp_chars, ref_chars)
                insertions = sum((j2 - j1) for tag, i1, i2, j1, j2 in ops if tag == 'insert')
                deletions = sum((i2 - i1) for tag, i1, i2, j1, j2 in ops if tag == 'delete')
                substitutions = sum((i2 - i1) for tag, i1, i2, j1, j2 in ops if tag == 'replace')
                total = len(ref_chars)
            else:
                insertions = len(hyp_chars)
                deletions = 0
                substitutions = 0
                total = 0
            
            stats_dict = {
                # 'utt_id': utt_id[i],
                'insertions': insertions,
                'deletions': deletions,
                'substitutions': substitutions,
                'total': total,
            }
            per_utt_cer_stats.append(stats_dict)
        end_time_cer = time.time()
        print("Time calculating CER:", end_time_cer - beginning_time_cer, "seconds")
        # End for code for CER calculation


        # print stuff
        for i in range(len(losses)):
            lang_id = batch_lang_ids[i]
            filename = utt_id[i]
            loss_value = losses[i]
            input_length = input_lengths[i]
            target_length = target_lengths[i]
            cer_stats = per_utt_cer_stats[i]
            hyp_chars, ref_chars = per_utt_hyp_ref[i]
            if valid:
                print(f"Validation Sample {i}: Language = {lang_id}, Filename = {filename}, Loss = {loss_value}, Input Length = {input_length}, Target Length = {target_length}, (hyp, ref) = ({hyp_chars, ref_chars}), Target-ref equal length = {len(ref_chars) == target_length}, (I, D, S, T) = ({cer_stats['insertions']}, {cer_stats['deletions']}, {cer_stats['substitutions']}, {cer_stats['total']})")
            else:
                print(f"Training Sample {i}: Language = {lang_id}, Filename = {filename}, Loss = {loss_value}, Input Length = {input_length}, Target Length = {target_length}, (hyp, ref) = ({hyp_chars, ref_chars}), Target-ref equal length = {len(ref_chars) == target_length}, (I, D, S, T) = ({cer_stats['insertions']}, {cer_stats['deletions']}, {cer_stats['substitutions']}, {cer_stats['total']})")

        step_size = self.dro_step_size

        if not valid:
            for q_ix in set(batch_lang_q_indices):
                group_losses = torch.tensor([
                    losses[i]
                    for i in range(losses.shape[0])
                    if batch_lang_q_indices[i] == q_ix
                ])

                # Calculate CER for this group
                group_cer_stats = [
                    per_utt_cer_stats[i]
                    for i in range(len(per_utt_cer_stats))
                    if batch_lang_q_indices[i] == q_ix
                ]
                
                # Aggregate CER statistics for the group
                total_insertions = sum(stats['insertions'] for stats in group_cer_stats)
                total_deletions = sum(stats['deletions'] for stats in group_cer_stats)
                total_substitutions = sum(stats['substitutions'] for stats in group_cer_stats)
                total_characters = sum(stats['total'] for stats in group_cer_stats)
                
                # Calculate group CER
                if total_characters > 0:
                    numerator_cer = total_insertions + total_deletions + total_substitutions
                    denominator_cer = total_characters
                    group_cer = (total_insertions + total_deletions + total_substitutions) / total_characters
                else:
                    numerator_cer = 0
                    denominator_cer = 0
                    group_cer = 0.0

                numerator_cer_tensor = torch.tensor(numerator_cer, device=self.dro_q.device)
                denominator_cer_tensor = torch.tensor(denominator_cer, device=self.dro_q.device)
                group_cer_tensor = torch.tensor(group_cer, device=self.dro_q.device)
                print(f"Group {q_ix, ix_to_group_id[q_ix]} CER: {group_cer:.4f} (I={total_insertions}, D={total_deletions}, S={total_substitutions}, T={total_characters})")

                if (self.agg == "sum"):
                    group_mean_loss = torch.sum(group_losses)
                else:
                    group_mean_loss = torch.mean(group_losses)

                if not self.accumulation:
                    if self.smoothing > 0:
                        # add the smoothing hyperparameter
                        # self.dro_q[q_ix] *= torch.exp((group_mean_loss * step_size) / (self.dro_q[q_ix] + self.smoothing))
                        self.dro_q[q_ix] *= torch.exp((group_cer_tensor * step_size) / (self.dro_q[q_ix] + self.smoothing))
                        # print("Update Magnitude", torch.exp((group_mean_loss * step_size) / (self.dro_q[q_ix] + self.smoothing)))
                        print("Update Magnitude with CER", torch.exp((group_cer_tensor * step_size) / (self.dro_q[q_ix] + self.smoothing)))
                    else:
                        # self.dro_q[q_ix] *= torch.exp(group_mean_loss * step_size) 
                        self.dro_q[q_ix] *= torch.exp(group_cer_tensor * step_size) 
                        # print("Update Magnitude", torch.exp(group_mean_loss * step_size))
                        print("Update Magnitude with CER", torch.exp(group_cer_tensor * step_size))
                else:
                    print("Loss Stored")
                    self.group_losses[q_ix].append((numerator_cer_tensor, denominator_cer_tensor))

            if self.accumulation:
                check = True
                for _ in self.group_losses:
                    if len(self.group_losses[_]) == 0:
                        check = False
                        break 

                if check:
                    for _ in self.group_losses:
                        update_term_numerator_cer = sum(t[0] for t in self.group_losses[_])
                        update_term_denominator_cer = sum(t[1] for t in self.group_losses[_])
                        if update_term_denominator_cer > 0:
                            update_term_group_cer = update_term_numerator_cer / update_term_denominator_cer
                            print(f"Group {_} Accumulated CER: {update_term_group_cer:.4f}")
                        else:
                            update_term_group_cer = 0.0
                            print(f"Group {_} Accumulated CER: {update_term_group_cer:.4f}")
                        if self.smoothing > 0:
                            self.dro_q[_] *= torch.exp((update_term_group_cer * step_size)/(self.dro_q[_] + self.smoothing))
                            print("Update Magnitude using CER and accumulation", torch.exp((update_term_group_cer * step_size)/(self.dro_q[_] + self.smoothing)))
                        else:
                            self.dro_q[_] *= torch.exp(update_term_group_cer * step_size)
                            print("Update Magnitude using CER and accumulation", torch.exp(update_term_group_cer * step_size))

                    self.normalize_dro_q()
                    for _ in self.group_losses:
                        self.group_losses[_] = []
            else:
                self.normalize_dro_q()
        
        if self.normalize_grad:
            # multiply loss by number of groups
            dro_losses = torch.stack([
                losses[ix] * self.dro_q[batch_lang_q_indices[ix]] 
                * self.dro_group_count
                for ix in range(losses.shape[0])
            ])
        else:
            print("Not normalizing gradient")
            dro_losses = torch.stack([
                losses[ix] * self.dro_q[batch_lang_q_indices[ix]] 
                for ix in range(losses.shape[0])
            ])
        
        if not valid:
            return dro_losses
        else:
            return losses

    def normalize_dro_q(self):
        self.dro_q += self.dro_q_epsilon # to prevent zero weights
        self.dro_q = self.dro_q / self.dro_q.sum()
        print("normalized dro_q:")
        for group_id, group_ix in self.group_id_to_ix.items():
            print(f"q[group#{group_id}]= {self.dro_q[group_ix].item()}")
