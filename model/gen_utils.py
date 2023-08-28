from pkg_utils import (
is_numpy_available,
is_torch_available,
is_torch_cuda_available,
PackageNotFoundError
)
from typing import Dict, Tuple, List, Optional, Callable, Iterable
if is_torch_available():
    import torch
    import torch.nn.functional as F
    import torch.utils.data as tud
else:
    raise PackageNotFoundError('torch')
from outputs_utils import GreedySearchDecoderOnlyOutput, BeamSearchDecoderOnlyOutput, SampleDecoderOnlyOutput


class TextGeneratorBase:
    """
    This TextGeneratorBase class is a base class for generating text using various techniques.
    Subclasses can implement specific generation strategies, such as sampling, beam search, etc.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer()
        # self.tokenizer.encode_ = tokenizer.encode
        # self.tokenizer.decode_ = tokenizer.decode

    def set_seed(self, seed):
        if seed is not None:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            else:
                torch.manual_seed(seed)

    def generate_text(
        self,
        prompt: str,
        conditioned : str = None,
        question: str = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = None,
        top_p: float = None,
        num_return_sequences: int = 1,
        early_stopping: bool = False,
        no_repeat_ngram_size: int = None,
        beam_width: int = None,
        seed: int = None,
        device : str = 'cpu',
        repetition_penalty : float = None,
        diversion_penalty : float = None,
        **kwargs
        ):
        self.set_seed(seed)
        input_ids = torch.tensor(self.tokenizer.encode_(text=prompt), device=device, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1)
        generated_sequences = []
        input_ids = self.model.prepare_inputs_for_generation(input_ids)
        input_ids = input_ids['input_ids']

        with torch.no_grad():
                sequence = input_ids.clone()
                for _ in range(max_length):
                    logits = self.model(sequence).logits

                    scaled_logits = self._set_temp(logits, temperature)

                    if top_k is not None or top_p is not None:
                        logits = self._filter_logits(scaled_logits, top_k, top_p)

                    selected_token = self._sample_token(self._repetition_penalty(scaled_logits, sequence[:, -1], repetition_penalty))

                    if early_stopping:
                        break

                    if diversion_penalty is not None:
                        scaled_logits = self._diversion_penalty(scaled_logits, selected_token, diversion_penalty)

                    sequence = torch.cat([sequence, selected_token], dim=-1)

                generated_sequences.append(sequence.tolist())
        generated_texts = self.decode_generated_text(generated_sequences, max_length)
        return generated_texts

    def generate_text_beam_search(self, prompt, max_length=100, num_return_sequences=1, beam_width=None, batch_size=None, device='cpu', seed=None, **kwargs):
        if beam_width is None:
            beam_width = self.beam_width

        self.set_seed(seed)
        input_ids = torch.tensor(self.tokenizer.encode_(text=prompt), device=device, dtype=torch.long).unsqueeze(0).repeat(beam_width, 1)
        generated_sequences = []

        beam_search = BeamSearch(self.model, beam_width=beam_width, batch_size=batch_size)
        sequences, probabilities = beam_search.beam_search(input_ids, predictions=max_length)
        return sequences, probabilities

    def _set_temp(self, logits: torch.FloatTensor, temperature: float) -> torch.FloatTensor:
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        scaled_logits = logits[:,-1, :] / temperature
        return scaled_logits

    def _sample_token(self, logits):
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        selected_token = torch.multinomial(probabilities, num_samples=1, replacement=True)
        return selected_token

    def _filter_logits(self, logits, top_k=None, top_p=None):
        if top_k is not None:
            logits = self.top_k_filter(logits, top_k)
        if top_p is not None:
            logits = self._top_p_logits(logits, top_p)
        return logits

    def top_k_filter(self, logits, k):
        values, indices = torch.topk(logits, k)
        min_values = values[:, -1].unsqueeze(-1)
        filtered_logits = torch.where(logits < min_values, logits.new_full(logits.shape, float('-inf')), logits)
        return filtered_logits

    def _top_p_logits(self, logits : torch.LongTensor, top_p : float):
        logits_sort, logits_ids = torch.sort(logits, dim=-1, descending=True)
        logits_sort = torch.nn.functional.softmax(logits_sort, dim=-1)
        logits_sum = torch.cumsum(logits_sort, dim=-1)
        mask = logits_sum - logits_sort > top_p
        logits_sort[mask] = 0.0
        logits_sort.div_(logits_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(logits_sort, num_samples=1)
        next_token = torch.gather(logits_ids, -1, next_token)
        return next_token
    
    def _repetition_penalty(self, logits, selected_token, penalty : float):
        penalty_logits = logits.clone()
        penalty_logits[:, selected_token] /= penalty
        return penalty_logits

    def _get_ngrams(self,ngram_size, prev_input_ids, num_hypos):
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram.setdefault(prev_ngram_tuple, []).append(ngram[-1])
        return generated_ngrams

    def _get_generated_ngrams(self,banned_ngrams, prev_input_ids, ngram_size, cur_len):
        start_idx = cur_len + 1 - ngram_size
        ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
        return banned_ngrams.get(ngram_idx, [])

    def _calc_banned_ngram_tokens(self,ngram_size, prev_input_ids, num_hypos, cur_len):
        if cur_len + 1 < ngram_size:
            return [[] for _ in range(num_hypos)]
        
        generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
        
        banned_tokens = [
            _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
            for hypo_idx in range(num_hypos)
        ]
        return banned_tokens

    def decode_generated_text(self, generated_sequences, max_length):
            generated_texts = []

            for seq in generated_sequences:
                for txt in seq:
                    generated_text = self.tokenizer.decode_(txt[0:max_length:])
                    generated_texts.append({'output':generated_text})

            return generated_texts

class RandomSampling(TextGeneratorBase):
    """
    This class generates text using random sampling without top-k or top-p filtering.
    """

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        device: str = 'cpu',
        seed: int = None,
        **kwargs
    ):
        return super().generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            device=device,
            seed=seed,
            top_k=None,
            top_p=None,
            **kwargs
        )

class Sampling(TextGeneratorBase):
    """
    This class generates text using advanced sampling techniques with various features.
    """

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.90,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        early_stopping: bool = False,
        seed: int = None,
        device : str = 'cpu',
        **kwargs
    ):
        return super().generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            early_stopping=early_stopping,
            seed=seed,
            device=device,
            **kwargs
        )

class Beam(TextGeneratorBase):
    def generate_text_beam(self, prompt, max_length=100, num_return_sequences=1, beam_width=None, device='cpu', seed=None, **kwargs):
        return self.generate_text_beam_search(
            prompt=prompt,
            max_length=max_length,
         num_return_sequences=num_return_sequences, 
         beam_width=beam_width,
          device=device, 
          seed=seed,
        **kwargs)

class BeamSearch:
    def __init__(self, model, beam_width=5, batch_size=128):
        self.model = model
        self.beam_width = beam_width
        self.batch_size = batch_size

    def beam_search(self, X, predictions=20):
        with torch.no_grad():
            next_probabilities = self.model(X).logits
            vocabulary_size = next_probabilities.shape[-1]
            probabilities, idx = next_probabilities.squeeze().log_softmax(-1) \
                .topk(k=self.beam_width, axis=-1)
            X = X.repeat((self.beam_width, 1, 1)).transpose(0, 1) \
                .flatten(end_dim=-2)
            next_chars = idx.reshape(-1, 1)
            X = torch.cat((X, next_chars), axis=-1)

            for i in range(predictions - 1):
                dataset = tud.TensorDataset(X)
                loader = tud.DataLoader(dataset, batch_size=self.batch_size)
                next_probabilities = []
                for (x,) in loader:
                    next_probabilities.append(
                        self.model(x).logits.log_softmax(-1)
                    )
                next_probabilities = torch.cat(next_probabilities, axis=0)
                next_probabilities = next_probabilities.reshape(
                    (-1, self.beam_width, next_probabilities.shape[-1])
                )
                probabilities = probabilities.unsqueeze(-1) + next_probabilities
                probabilities = probabilities.flatten(start_dim=1)
                probabilities, idx = probabilities.topk(
                    k=self.beam_width,
                    axis=-1
                )
                next_chars = torch.remainder(idx, vocabulary_size).flatten() \
                    .unsqueeze(-1)
                best_candidates = (idx / vocabulary_size).long()
                best_candidates += torch.arange(
                    X.shape[0] // self.beam_width,
                    device=X.device
                ).unsqueeze(-1) * self.beam_width
                X = X[best_candidates].flatten(end_dim=-2)
                X = torch.cat((X, next_chars), axis=1)
            return X.reshape(-1, self.beam_width, X.shape[-1]), probabilities