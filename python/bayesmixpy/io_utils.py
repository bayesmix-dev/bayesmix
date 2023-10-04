import os
from pathlib import Path
from google.protobuf.internal.decoder import _DecodeVarint32


def _is_file(a: str):
    out = False
    try:
        p = Path(a)
        out = p.exists() and p.is_file()
    except Exception as e:
        out = False
    return out

def maybe_print_proto_to_file(maybe_proto: str,
                         proto_name: str = None,
                         out_dir: str = None):
    """If maybe_proto is a file, returns the file name.
    If maybe_proto is a string representing a message, prints the message to
    a file and returns the file name.
    """
    if _is_file(maybe_proto):
        return maybe_proto

    proto_file = os.path.join(out_dir, proto_name + ".asciipb")

    with open(proto_file, "w") as f:
        print(maybe_proto, file=f)

    return proto_file

def read_many_protos_from_file(filename, MsgType):
    out = []
    with open(filename, "rb") as fp:
        buf = fp.read()

    n = 0
    while n < len(buf):
        msg_len, new_pos = _DecodeVarint32(buf, n)
        n = new_pos
        msg_buf = buf[n:n+msg_len]
        try:
            msg = MsgType()
            msg.ParseFromString(msg_buf)
            out.append(msg)
            n += msg_len
        except Exception as e:
            break

    return out
