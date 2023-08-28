class GPT2Config:
    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        name_or_path = r'c:\tmp\SI\gpt2\pytorch_model.bin',
        base_model_prefix = 'transformer',
        use_return_dict = True,
        output_attentions = False,
        output_hidden_states = True,
        add_cross_attention = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.base_model_prefix = base_model_prefix
        self.name_or_path = name_or_path
        self.n_positions = n_positions
        self.hidden_size =n_embd
        self.use_return_dict = use_return_dict
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.num_hidden_layers = n_layer
        self.max_position_embeddings = n_positions
        self.num_attention_heads = n_head
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.pruned_heads = kwargs.pop("pruned_heads", {})
        self.add_cross_attention = add_cross_attention
        self.torchscript = kwargs.pop("torchscript", False)


        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

class GPTNeoConfig:
    model_type = "gpt_neo"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=50257,
        max_position_embeddings=2048,
        hidden_size=2048,
        num_layers=24,
        attention_types=[[["global", "local"], 12]],
        num_heads=16,
        intermediate_size=None,
        window_size=256,
        activation_function="gelu_new",
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        classifier_dropout=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        output_attentions = False,
        output_hidden_states = True,
        add_cross_attention = False,
        use_return_dict = True,
        **kwargs,
    ):
        self.add_cross_attention = add_cross_attention
        self.use_return_dict = use_return_dict
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.vocab_size = vocab_size
        self.torchscript = kwargs.pop("torchscript", False)
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.window_size = window_size
        self.activation_function = activation_function
        self.resid_dropout = resid_dropout
        self.embed_dropout = embed_dropout
        self.attention_dropout = attention_dropout
        self.classifier_dropout = classifier_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.tie_word_embeddings = kwargs.pop("tie_word_embeddings", True)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)
        self.attention_types = attention_types
        self.attention_layers = self.expand_attention_types_params(attention_types)

        if len(self.attention_layers) != self.num_layers:
            raise ValueError(
                "Configuration for convolutional module is incorrect. "
                "It is required that `len(config.attention_layers)` == `config.num_layers` "
                f"but is `len(config.attention_layers) = {len(self.attention_layers)}`, "
                f"`config.num_layers = {self.num_layers}`. "
                "`config.attention_layers` is prepared using `config.attention_types`. "
                "Please verify the value of `config.attention_types` argument."
            )
    @staticmethod
    def expand_attention_types_params(attention_types):
        attentions = []
        for item in attention_types:
            for _ in range(item[1]):
                attentions.extend(item[0])
        return attentions


class NanoConfig(object):
    def __init__(
            self,
            vocab_size_or_config_json_file=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range