import os
import struct
import sys

import torch
from transformers import AutoConfig, AutoTokenizer
from safetensors import safe_open


# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1

    cs = [chr(n) for n in cs]

    return dict(zip(bs, cs))


def count_model_parts(dir_model: str) -> int:
    """Returns the number of model parts in the model directory."""
    num_parts = 0
    for filename in os.listdir(dir_model):
        if filename.startswith("model-"):
            num_parts += 1

    if num_parts > 0:
        print(f"Found {num_parts} model parts in {dir_model}")
    return num_parts



# SCRIPT START
if len(sys.argv) < 4:
    print("Usage: convert-h5-to-ggml.py dir-model [use-f32] out-gguf\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)


dir_model = sys.argv[1]
num_parts = count_model_parts(dir_model)
if num_parts != 2:
    print("2 model parts were not found\n")
    sys.exit(1)

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]

ftype = 1
fname_out = ""
if len(sys.argv) > 2:
    ftype = int(sys.argv[2])
    if ftype < 0 or ftype > 1:
        print("Invalid ftype: " + str(ftype))
        sys.exit(1)
    dir_gguf = sys.argv[3]
    fname_out = dir_gguf + ftype_str[ftype] + ".gguf"


config = AutoConfig.from_pretrained(dir_model, trust_remote_code=True)
hparams = config.to_dict()

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676D6C))  # magic: ggml in hex
fout.write(struct.pack("i", hparams["d_model"]))
fout.write(struct.pack("i", hparams["max_seq_len"]))
fout.write(struct.pack("i", hparams["n_heads"]))
fout.write(struct.pack("i", hparams["n_layers"]))
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("f", hparams["attn_config"]["alibi_bias_max"]))
fout.write(struct.pack("f", hparams["attn_config"]["clip_qkv"] or 0.0))
fout.write(struct.pack("?", hparams["attn_config"]["qk_ln"]))
fout.write(struct.pack("i", ftype))

vocab_size = hparams["vocab_size"]

tokenizer = AutoTokenizer.from_pretrained(dir_model, trust_remote_code=True)
encoder = tokenizer.get_vocab()
# Add added_tokens (special tokens) to the encoder
encoder.update(tokenizer.get_added_vocab())

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

counter = 0
# sort by value
text = bytearray("", encoding="utf-8")
for key in sorted(encoder, key=encoder.get):
    # workaround for key error when c not found
    text = ""
    for c in key:
        if c not in byte_decoder:
            text += c
        else:
            text += chr(byte_decoder[c])
    text = bytearray(text, encoding="utf-8")
    fout.write(struct.pack("i", len(text)))
    fout.write(text)
    counter += 1

# Repeat last token until vocab_size (making sure bytes are packed till the expected mark)
while counter < vocab_size:
    fout.write(struct.pack("i", len(text)))
    fout.write(text)
    counter += 1

part_names = (
        f"model-{n:05}-of-{num_parts:05}.safetensors" for n in range(1, num_parts + 1)
        )

for part_name in part_names:
    print(f"\n* Loading part: {part_name}")
    with safe_open(f"{dir_model}/{part_name}", framework="pt", device=0) as model_part:
        for name in model_part.keys():
            data = model_part.get_tensor(name)
            n_dims = len(data.shape)
    
            # ftype == 0 -> float32, ftype == 1 -> float16
            # default type is fp32
            ftype_cur = 0
            if ftype == 1 and name[-7:] == ".weight" and n_dims > 1:
                ftype_cur = 1

            # forcing data type to f32 or f16 if not already f32 or f16
            if data.dtype not in (torch.float16, torch.float32):
                if (ftype_cur == 1):
                    data = data.to(torch.float16)
                else:
                    data = data.to(torch.float32)

            # .cpu() to copy over data from gpu to cpu
            data = data.squeeze().cpu().numpy()
    
            print(
                "Processing variable: " + name + " with shape: ",
                data.shape, "->", data.dtype
            )
    
            # header
            str = name.encode("utf-8")
            fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
            for i in range(n_dims):
                fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
            fout.write(str)
    
            # data
            data.tofile(fout)

    # release memory
    del model_part

fout.close()

print("Done. Output file: " + fname_out)
print("")
