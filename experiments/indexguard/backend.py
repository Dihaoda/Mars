"""Paper Table 8 optimizer schedule, sharing Mars' tokenization and adapter I/O."""
import math
import random
import time
import torch
from mars.backend import Backend
from mars.utils import seed_all


class IndexBackend(Backend):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg['model'].get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            self.model.enable_input_require_grads()

    def train(self, state, rows, seed, max_steps=None):
        self.load(state)
        seed_all(seed)
        self.model.train()
        tr = self.cfg['train']
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=tr['learning_rate'], betas=tuple(tr['betas']),
                                      weight_decay=tr['weight_decay'])
        window_size = tr['batch_size'] * tr['gradient_accumulation']
        full_steps = math.ceil(len(rows)/window_size) * tr['local_epochs']
        steps_limit = full_steps if max_steps is None else min(full_steps, max_steps)
        # The warm-up sampling run uses one complete optimizer step at the nominal LR.
        # Main local training uses the full Table 8 schedule, reset for every client/round.
        warmup = math.ceil(full_steps * tr['warmup_ratio']) if max_steps is None else 0
        total_loss = total_tokens = steps = 0
        start = time.perf_counter()
        for epoch in range(tr['local_epochs']):
            shuffled = list(rows)
            random.Random(seed+epoch).shuffle(shuffled)
            for pos in range(0, len(shuffled), window_size):
                window = shuffled[pos:pos+window_size]
                token_count = sum(len(self.answers[r['label']]) for r in window)
                step_number = steps + 1
                if max_steps is not None:
                    factor = 1.0
                elif step_number <= warmup:
                    factor = step_number / max(warmup, 1)
                else:
                    factor = (full_steps-step_number+1) / max(full_steps-warmup, 1)
                for group in optimizer.param_groups:
                    group['lr'] = tr['learning_rate'] * factor
                optimizer.zero_grad(set_to_none=True)
                for j in range(0, len(window), tr['batch_size']):
                    losses, counts = self._losses(window[j:j+tr['batch_size']])
                    if not torch.isfinite(losses).all():
                        raise FloatingPointError('Nonfinite local training loss')
                    (losses.sum()/token_count).backward()
                    total_loss += float(losses.detach().sum())
                    total_tokens += int(counts.sum())
                torch.nn.utils.clip_grad_norm_(params, tr['max_grad_norm'], error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                if steps == steps_limit:
                    break
            if steps == steps_limit:
                break
        result = self.state()
        if not all(torch.isfinite(v).all() for v in result.values()):
            raise FloatingPointError('Nonfinite adapter')
        self.model.zero_grad(set_to_none=True)
        return result, {'loss': total_loss/max(total_tokens, 1), 'optimizer_steps': steps,
                        'answer_tokens': total_tokens, 'seconds': time.perf_counter()-start}
