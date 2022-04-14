import numpy as np
from ._model import Model
from ..utils import safe_isinstance
from .._serializable import Serializer, Deserializer

class TextGeneration(Model):
    """ Generates target sentence/ids using a base model.

    It generates target sentence/ids for a model (a pretrained transformer model or a function).
    """

    def __init__(self, model=None, tokenizer=None, target_sentences=None, device=None):
        """ Create a text generator model from a pretrained transformer model or a function.

        For a pretrained transformer model, a tokenizer should be passed.

        Parameters
        ----------
        model: object or function
            A object of any pretrained transformer model or function for which target sentence/ids are to be generated.

        tokenizer: object
            A tokenizer object(PreTrainedTokenizer/PreTrainedTokenizerFast) which is used to tokenize sentence.

        target_sentences: list
            A target sentence for every explanation row.

        device: str
            By default, it infers if system has a gpu and accordingly sets device. Should be 'cpu' or 'cuda' or pytorch models.

        Returns
        -------
        numpy.ndarray
            Array of target sentence/ids.
        """
        super().__init__(model)

        self.explanation_row = 0
        if target_sentences is not None:
            self.inner_model = lambda _: np.array([target_sentences[self.explanation_row]])

        self.tokenizer = tokenizer
        self.device = device
        if self.device is None:
            self.device = getattr(self.inner_model, "device", None)
        if safe_isinstance(model, "transformers.PreTrainedModel"):
            self.model_agnostic = False
            self.model_type = "pt"
        elif safe_isinstance(model, "transformers.TFPreTrainedModel"):
            self.model_agnostic = False
            self.model_type = "tf"
        else:
            self.model_agnostic = True
            self.model_type = None
        # X is input used to generate target sentence used for caching
        self.X = None
        # target sentence/ids generated from the model using X
        self.target_X = None

    def __call__(self, X):
        """ Generates target sentence/ids from X.

        Parameters
        ----------
        X: str or numpy.ndarray
            Input in the form of text or image.

        Returns
        -------
        numpy.ndarray
            Array of target sentence/ids.
        """
        if (self.X is None) or (isinstance(self.X, np.ndarray) and not np.array_equal(self.X, X)) or \
                (isinstance(self.X, str) and (self.X != X)):
            self.X = X
            # wrap text input in a numpy array
            if isinstance(X, str):
                X = np.array([X])
            # generate target sentence ids in model agnostic scenario
            if self.model_agnostic:
                self.target_X = self.inner_model(X)
            else:
                self.target_X = self.model_generate(X)
            # update explanation row count
            self.explanation_row += 1
        return np.array(self.target_X)

    def get_inputs(self, X, padding_side='right'):
        """ The function tokenizes source sentences.

        In model agnostic case, the function calls model(X) which is expected to
        return a batch of output sentences which is tokenized to compute inputs.

        Parameters
        ----------
        X: numpy.ndarray
            X is a batch of sentences.

        Returns
        -------
        dict
            Dictionary of padded source sentence ids and attention mask as tensors("pt" or "tf" based on model_type).
        """
        # set tokenizer padding to prepare inputs for batch inferencing
        # padding_side="left" for only decoder models text generation eg. GPT2
        self.tokenizer.padding_side = padding_side
        """
        Here I modify the code in order to have max length padding on encoder inputs.
        """
#         inputs = self.tokenizer(X.tolist(), return_tensors=self.model_type, padding=True)
        inputs = self.tokenizer(X.tolist(), return_tensors=self.model_type, padding="max_length")
        # set tokenizer padding to default
        self.tokenizer.padding_side = 'right'
        return inputs

    def model_generate(self, X):
        """ This function performs text generation for tensorflow and pytorch models.

        Parameters
        ----------
        X: dict
            Dictionary of padded source sentence ids and attention mask as tensors.

        Returns
        -------
        numpy.ndarray
            Returns target sentence ids.
        """
        if (hasattr(self.inner_model.config, "is_encoder_decoder") and not self.inner_model.config.is_encoder_decoder) \
                and (hasattr(self.inner_model.config, "is_decoder") and not self.inner_model.config.is_decoder):
            pass
            # TODOmaybe: Is this okay? I am just assuming we want is_decoder when neither are set
            #self.inner_model.config.is_decoder = True
            # raise ValueError(
            #     "Please assign either of is_encoder_decoder or is_decoder to True in model config for extracting target sentence ids"
            # )
        # check if user assigned any text generation specific kwargs
        text_generation_params = {}
        if self.inner_model.config.__dict__.get("task_specific_params") is not None and \
                self.inner_model.config.task_specific_params.get("text-generation") is not None:
            text_generation_params = self.inner_model.config.task_specific_params["text-generation"]
            if not isinstance(text_generation_params, dict):
                raise ValueError(
                    "Please assign text generation params as a dictionary under task_specific_params with key 'text-generation' "
                )
            # remove keys that are overridden by params on the model itself
            # (this is to mimic how precedence works for transformers pipelines)
            for k in list(text_generation_params.keys()):
                if hasattr(self.inner_model.config, k):
                    del text_generation_params[k]
        if self.model_type == "pt":
            # create torch tensors and move to device
            # TODOmaybe: SML: why move the model from where it was? the could mess with the user env (i.e. it breaks piplines)
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if self.device is None else self.device
            # self.inner_model = self.inner_model.to(device)
            # self.inner_model.eval()
            import torch # pylint: disable=import-outside-toplevel
            with torch.no_grad():
                if self.inner_model.config.is_encoder_decoder:
                    inputs = self.get_inputs(X)
                else:
                    inputs = self.get_inputs(X, padding_side="left")
                if self.device is not None:
                    inputs = inputs.to(self.device)
                
                """
                Here I define the temporary inference script for the current model (inner_model) to match the expectations.
                """
                def _model_inference(inputs):
                    # temporary variable for storing next tokens predicted by the model which is <start> token at the beginning
                    next_token = 0

                    # list of tokens as an input to the model
                    start_tokens_list = [0]

                    # we take the result from tokenizer and extract `input_ids` from it (which is just a sequence of integers representing token ids)
                    smile_input_ids = inputs['input_ids']

                    # we also take the attention mask which actually masks the input sequence length (required for padded sequences)
                    smile_attention_mask = inputs['attention_mask']

                    # iterate untill some special symbols are generated or max sequence length is achieved
                    while (
                            next_token != self.inner_model.config.eos_token_id and
                            next_token != self.inner_model.config.pad_token_id and
                            len(start_tokens_list) <= self.tokenizer.model_max_length
                    ):
                        # start tokens are the tokens that will be given as an input at current step to the BERT decoder
                        start_tokens = torch.tensor(start_tokens_list, device=self.device).expand((1, -1))

                        # cut-off gradient calculations and run model inference to get logits for the next token
                        with torch.no_grad():
                            # model will return logits for each decoder output position
                            output_logits = self.inner_model(
                                input_ids=smile_input_ids,
                                decoder_input_ids=start_tokens,
                                encoder_attention_mask=smile_attention_mask
                            ).logits

                        # here we apply softmax on these logits and we get probability distribution over tokens ids for each decoder output position
                        tokenized_sentence_prediction_array = torch.softmax(output_logits, dim=-1)

                        # now we choose the tokens with highest probabilities
                        prediction = torch.argmax(tokenized_sentence_prediction_array, dim=2)

                        # we just take out an integer value from the last position of decoder outputs
                        #   because the previous positions were already considered and are in `start_tokens_list`
                        next_token = int(prediction[0][-1].cpu().detach().numpy())

                        # we append this last predicted token into start tokens list and prepare the input for BERT decoder for the next iteration
                        start_tokens_list.append(next_token)

                    result = np.array(start_tokens_list).reshape(1, -1)

                    return result
                
                outputs = _model_inference(inputs)
                # outputs = self.inner_model.generate(**inputs, **text_generation_params).detach().cpu().numpy()
                
        elif self.model_type == "tf":
            if self.inner_model.config.is_encoder_decoder:
                inputs = self.get_inputs(X)
            else:
                inputs = self.get_inputs(X, padding_side="left")
            if self.device is None:
                outputs = self.inner_model.generate(inputs, **text_generation_params).numpy()
            else:
                try:
                    import tensorflow as tf # pylint: disable=import-outside-toplevel
                    with tf.device(self.device):
                        outputs = self.inner_model.generate(inputs, **text_generation_params).numpy()
                except RuntimeError as err:
                    print(err)
        if getattr(self.inner_model.config, "is_decoder", True):
            # slice the output ids after the input ids
            outputs = outputs[:, inputs["input_ids"].shape[1]:]
        # parse output ids to find special tokens in prefix and suffix
        parsed_tokenizer_dict = self.parse_prefix_suffix_for_model_generate_output(outputs[0, :].tolist())
        keep_prefix, keep_suffix = parsed_tokenizer_dict['keep_prefix'], parsed_tokenizer_dict['keep_suffix']
        # extract target sentence ids by slicing off prefix and suffix
        if keep_suffix > 0:
            target_X = outputs[:, keep_prefix:-keep_suffix]
        else:
            target_X = outputs[:, keep_prefix:]
        return target_X

    def parse_prefix_suffix_for_model_generate_output(self, output):
        """ Calculates if special tokens are present in the begining/end of the model generated output.

        Parameters
        ----------
        output: list
            A list of output token ids.

        Returns
        -------
        dict
            Dictionary of prefix and suffix lengths concerning special tokens in output ids.
        """
        keep_prefix, keep_suffix = 0, 0
        if self.tokenizer.convert_ids_to_tokens(output[0]) in self.tokenizer.special_tokens_map.values():
            keep_prefix = 1
        if len(output) > 1 and self.tokenizer.convert_ids_to_tokens(output[-1]) in self.tokenizer.special_tokens_map.values():
            keep_suffix = 1
        return {
            'keep_prefix' : keep_prefix,
            'keep_suffix' : keep_suffix
        }

    def save(self, out_file):
        super().save(out_file)

        # Increment the verison number when the encoding changes!
        with Serializer(out_file, "shap.models.TextGeneration", version=0) as s:
            s.save("tokenizer", self.tokenizer)
            s.save("device", self.device)

    @classmethod
    def load(cls, in_file, instantiate=True):
        if instantiate:
            return cls._instantiated_load(in_file)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.models.TextGeneration", min_version=0, max_version=0) as s:
            kwargs["tokenizer"] = s.load("tokenizer")
            kwargs["device"] = s.load("device")
        return kwargs
