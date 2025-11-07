import torch
from typing import Optional
from einops import rearrange
from ..train.models import AudioTransformerMAE_Encoder

PRETRAINED_CHECKPOINTS = {
    "dasheng_base": "https://zenodo.org/records/11511780/files/dasheng_base.pt?download=1",
    "dasheng_06B": "https://zenodo.org/records/11511780/files/dasheng_06b.pt?download=1",
    "dasheng_12B": "https://zenodo.org/records/11511780/files/dasheng_12b.pt?download=1",
}


# Using the pretrained encoders, remove all masking
class Dasheng(AudioTransformerMAE_Encoder):
    # need the *args, **kwargs otherwise we get a linter warning
    def forward_features(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        *_, t = x.shape
        x = x + self.time_pos_embed[:, :, :, :t]
        x = x + self.freq_pos_embed[:, :, :, :]
        x = rearrange(x, "b c f t -> b (f t) c")
        if self.pooling == "token":
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.token_pos_embed[:, :]
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x, **kwargs)
        x = self.norm(x)
        return x

    def _to_mask(self, lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        batch_size = len(lengths)
        idx = torch.arange(max_length, device=lengths.device)
        idx = idx.repeat(batch_size).view(batch_size, max_length)
        mask = (idx >= lengths.unsqueeze(-1)).bool()
        return mask

    def forward_spectrogram(self, x: torch.Tensor, x_length:Optional[torch.Tensor] = None) -> torch.Tensor:
        # For dasheng, target-length is 40 ms
        target_length_in_patches = self.target_length // self.patch_stride[-1]
        x = self.patch_embed(x)
        b, c, f, t = x.shape
        input_splits = x.split(target_length_in_patches, dim=-1)
        mask = None # Single mask
        masks = [None for _ in range(len(input_splits))]
        if x_length is not None:
            assert len(x_length) == len(x),"batchsizes of input x and x_length need to be same"
            assert x_length.ndim == 1, "Lengths are of size (B,)"
            scaled_lengths = (x_length / (self.hop_size * 4)).long() # 40ms for all dasheng models
            # Note that the mask is in (t f) format, but transformers here use (f t) format
            mask = self._to_mask(
                max_length=t,
                lengths=scaled_lengths,
                )
            # Trim mask to only use valid "patches", since x.shape[-1] is based on the possibly padded input
            masks = mask.split(target_length_in_patches, dim=-1)
        outputs = []

        for split_x,mask in zip(input_splits, masks):
            forward_kwargs = dict(mask = mask)
            split_x = self.forward_features(split_x, **forward_kwargs)
            outputs.append(split_x)
        x = torch.cat(outputs, dim =1 )
        return x


    def forward(self, x, x_length : Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.forward_to_spec(x)
        return self.forward_spectrogram(x,x_length=x_length)

    @classmethod
    def from_pretrained(
        cls, pretrained_url: str, **additional_model_kwargs
    ) -> AudioTransformerMAE_Encoder:
        """
        Class method to create a new Dasheng model instance from a pre-trained model stored in the Hugging Face model hub.
        """
        if "http" in pretrained_url:
            dump = torch.hub.load_state_dict_from_url(
                pretrained_url, map_location="cpu"
            )
        else:
            dump = torch.load(pretrained_url, map_location="cpu")
        model_parmeters, model_config = dump["model"], dump["config"]
        instance = cls(**{**model_config, **additional_model_kwargs})
        instance.load_state_dict(model_parmeters, strict=True)
        return instance


def dasheng_base(path = None, **model_kwargs):
    model_kwargs["embed_dim"] = 768
    model_kwargs["depth"] = 12
    model_kwargs["num_heads"] = 12
    return Dasheng.from_pretrained(
        pretrained_url=path or PRETRAINED_CHECKPOINTS["dasheng_base"], **model_kwargs
    )


def dasheng_06B(path = None, **model_kwargs):
    model_kwargs["embed_dim"] = 1280
    model_kwargs["depth"] = 32
    model_kwargs["num_heads"] = 16
    return Dasheng.from_pretrained(
        pretrained_url=path or PRETRAINED_CHECKPOINTS["dasheng_06B"], **model_kwargs
    )


def dasheng_12B(path = None, **model_kwargs):
    model_kwargs["embed_dim"] = 1536
    model_kwargs["depth"] = 40
    model_kwargs["num_heads"] = 24
    return Dasheng.from_pretrained(
        pretrained_url=path or PRETRAINED_CHECKPOINTS["dasheng_12B"], **model_kwargs
    )


if __name__ == "__main__":
    mdl = dasheng_base()
    print(mdl(torch.randn(1, 168499)).shape)
