from model.dna_xlstm.xlstm.xlstm_lm_model import xLSTMLMModelConfig
from model.dna_xlstm.xlstm.blocks.mlstm.block import mLSTMBlockConfig
from model.dna_xlstm.xlstm.blocks.mlstm.layer import mLSTMLayerConfig
from model.dna_xlstm.xlstm.blocks.slstm.block import sLSTMBlockConfig
from model.dna_xlstm.xlstm.blocks.slstm.layer import sLSTMLayerConfig
from model.dna_xlstm.xlstm.components.feedforward import FeedForwardConfig
import ml_collections as mlc

from data.alphabet import Alphabet
from data.constants import *


def model_cfg(name: str):
    c = {}
    if name == "smallS":
        c["context_length"] = 1024
        c["num_blocks"] = 5
        c["slstm_at"] = [0, 1, 2, 3, 4]
        c["embedding_dim"] = 128
        c["proj_factor"] = 1.25
        c["bidirectional_alternating"] = True
        c["m_backend_bidirectional"] = False
        c["dropout"] = 0.2

    if name == "bigS":
        c["context_length"] = 1024
        c["num_blocks"] = 6
        c["slstm_at"] = [0, 1, 2, 3, 4, 5]
        c["embedding_dim"] = 256
        c["proj_factor"] = 1
        c["bidirectional_alternating"] = True
        c["m_backend_bidirectional"] = False
        c["dropout"] = 0.2

    if name == "M":
        c["context_length"] = 1024
        c["num_blocks"] = 15
        c["slstm_at"] = []
        c["embedding_dim"] = 600
        c["proj_factor"] = 1  # filler, this is only for slstm
        c["bidirectional_alternating"] = True
        c["m_backend_bidirectional"] = False
        c["dropout"] = 0.0

    cfg = xLSTMLMModelConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4,
                        qkv_proj_blocksize=4, num_heads=4
                    )  # mlstm has default projection ratio of 2
                ),
                slstm_block=sLSTMBlockConfig(
                    slstm=sLSTMLayerConfig(
                        backend="cuda",
                        num_heads=4,
                        conv1d_kernel_size=4,
                        bias_init="powerlaw_blockdependent",
                    ),
                    feedforward=FeedForwardConfig(proj_factor=c["proj_factor"],
                                                  act_fn="gelu"),
                ),
                context_length=c["context_length"],  # context length needed for positional embeddings
                num_blocks=c["num_blocks"],
                embedding_dim=c["embedding_dim"],
                add_post_blocks_norm=True,
                bidirectional=True,
                bidirectional_alternating=c["bidirectional_alternating"],
                m_backend_bidirectional=c["m_backend_bidirectional"],
                s_position_embeddings=False,
                vocab_size=22,
                tie_weights=False,
                slstm_at=c["slstm_at"],
                weight_decay_on_embedding=False,
                add_embedding_dropout=False,
                padding_idx=1,
                dropout=c["dropout"]

            )

    embed_dim = mlc.FieldReference(c["embedding_dim"], field_type=int)

    default_alphabet = Alphabet()
    alphabet_size = mlc.FieldReference(len(default_alphabet), field_type=int)
    mask_tkn_idx = mlc.FieldReference(default_alphabet.get_idx(MASK_TKN), field_type=int)
    pad_tkn_idx = mlc.FieldReference(default_alphabet.get_idx(PAD_TKN), field_type=int)

    mask_ratio = mlc.FieldReference(0.15, field_type=float)
    mask_tkn_prob = mlc.FieldReference(0.8, field_type=float)
    default_config = mlc.ConfigDict(
        {
            "globals": {
                "embed_dim": embed_dim,
                "alphabet_size": alphabet_size,
                "mask_tkn_idx": mask_tkn_idx,
                "pad_tkn_idx": pad_tkn_idx,
                "mask_ratio": mask_ratio,
                "mask_tkn_prob": mask_tkn_prob,
            },
            "alphabet": {
                "standard_tkns": RNA_TOKENS,
                "special_tkns": [CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
            },
            "training": {
                "optimizer": {
                    "lr": 5e-4,
                    "weight_decay": 0.1,
                },
                "lr_scheduler": {
                    "warm_up": {
                        "iters": 1000,
                    },
                    "cosine_decay": {
                        "T_max": 40000,
                        "eta_min": 5e-5,
                    },
                },
                "masking": {
                    "bert_masking": {
                        "mask_ratio": mask_ratio,
                        "mask_tkn_prob": mask_tkn_prob,
                        "random_tkn_prob": 0.1,
                    }
                },
            },
            "model": {
                "embedding": {
                    "num_embeddings": alphabet_size,
                    "embedding_dim": embed_dim,
                    "padding_idx": pad_tkn_idx,
                }
            }
        }
    )

    return cfg, default_config