from pkg_utils import is_torch_available,PackageNotFoundError
from typing import Any, Tuple, Optional, List, Union
from collections import OrderedDict
from dataclasses import dataclass, fields
from process_ import Tokenizer

is_tensor = Tokenizer.is_tensor

if is_torch_available():
    import torch
else:
    raise PackageNotFoundError('torch')

class cached_property(property):
    """
    Mimics @property but caches output in a member variable.
    """

    def __init__(self, func):
        self.func = func
        self.attr_name = f"__cached_{func.__name__}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        cached_value = getattr(obj, self.attr_name, None)
        if cached_value is None:
            cached_value = self.func(obj)
            setattr(obj, self.attr_name, cached_value)
        return cached_value

    def __set__(self, obj, value):
        setattr(obj, self.attr_name, value)

    def __delete__(self, obj):
        if hasattr(obj, self.attr_name):
            delattr(obj, self.attr_name)

class ModelOutput(OrderedDict):
    """
    Base class for model outputs as dataclass. Supports dictionary-like access with string keys and
    tuple-like access with integer indices for non-None attributes.
    """
    
    def __post_init__(self):
        class_fields = fields(self)
        
        # Safety and consistency checks
        if not class_fields:
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if any(field.default is not None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should have at most one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])
        
        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
            else:
                try:
                    iterator = iter(first_field)
                except TypeError:
                    iterator = None

            if iterator is not None:
                for key, value in iterator:
                    setattr(self, key, value)
        else:
            for field in class_fields:
                value = getattr(self, field.name)
                if value is not None:
                    self[field.name] = value

    def __getitem__(self, key):
        if isinstance(key, str):
            inner_dict = dict(self.items())
            return inner_dict[key]
        else:
            return self.to_tuple()[key]

    def __setattr__(self, name, value):
        if name in self and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert non-None attributes into a tuple.
        """
        return tuple(self[k] for k in self if self[k] is not None)

@dataclass
class BaseModelOutput(ModelOutput):
    
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BaseModelOutputWithNoAttention(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class BaseModelOutputWithPooling(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BaseModelOutputWithPoolingAndNoAttention(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BaseModelOutputWithPast(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BaseModelOutputWithCrossAttentions(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class MoECausalLMOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    z_loss: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class MoEModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    router_probs: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class MoEModelOutputWithPastAndCrossAttentions(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    router_probs: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class Seq2SeqModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class Seq2SeqMoEModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_router_logits: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_router_logits: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class CausalLMOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class CausalLMOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    
    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mc_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class SequenceClassifierOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MaskedLMOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Seq2SeqLMOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Seq2SeqMoEOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    encoder_z_loss: torch.FloatTensor = None
    decoder_z_loss: torch.FloatTensor = None
    encoder_aux_loss: torch.FloatTensor = None
    decoder_aux_loss: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_router_logits: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_router_logits: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class NextSentencePredictorOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SequenceClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Seq2SeqSequenceClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MultipleChoiceModelOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class TokenClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class QAModelOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Seq2SeqQuestionAnsweringModelOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class GreedySearchDecoderOnlyOutput(ModelOutput):

    sequences: torch.LongTensor = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

@dataclass
class GreedySearchEncoderDecoderOutput(ModelOutput):

    sequences: torch.LongTensor = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class SampleDecoderOnlyOutput(ModelOutput):

    sequences: torch.LongTensor = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class SampleEncoderDecoderOutput(ModelOutput):

    sequences: torch.LongTensor = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class BeamSearchDecoderOnlyOutput(ModelOutput):

    sequences: torch.LongTensor = None
    sequences_logits: Optional[torch.FloatTensor] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class BeamSearchEncoderDecoderOutput(ModelOutput):

    sequences: torch.LongTensor = None
    sequences_logits: Optional[torch.FloatTensor] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
