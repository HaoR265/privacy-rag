"""Export PRCONST1 public constants with explicit fixed shape and f20 scale."""
import hashlib
import json
from pathlib import Path
import struct


def export(root=Path("artifacts/constants")):
    source=root/"constants_v1.json"
    constants=json.loads(source.read_text())
    mask=(1<<59)-1
    values=[constants["attention_scale_f20"]]
    values += [value for table in constants["exp_f24"] for value in table]
    for name in ("rope_cos_f20","rope_sin_f20"):
        values += [constants[name][position][coordinate] & mask for coordinate in range(64) for position in range(1280)]
    path=root/"constants_v1.bin"
    data=b"PRCONST1"+struct.pack("<"+"Q"*len(values),*values)
    if len(data)!=1316880:raise ValueError("NATIVE_CONSTANT_LENGTH")
    path.write_bytes(data)
    reread=path.read_bytes()
    if reread[:8]!=b"PRCONST1" or list(struct.unpack("<"+"Q"*len(values),reread[8:]))!=values:
        raise ValueError("NATIVE_CONSTANT_ROUNDTRIP")
    manifest={"format_version":"PRCONST1","path":path.name,"bytes":len(data),
              "sha256":hashlib.sha256(data).hexdigest(),"source_json_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
              "endianness":"little","word_type":"uint64","ring_bits":59,
              "layout":[{"name":"magic","offset":0,"bytes":8,"value":"PRCONST1"},
                        {"name":"attention_scale_f20","offset":8,"count":1,"fractional_bits":20},
                        {"name":"exp_f24","offset":16,"shape":[3,256],"fractional_bits":24},
                        {"name":"rope_cos_f20","offset":6160,"shape":[64,1280],"fractional_bits":20},
                        {"name":"rope_sin_f20","offset":661520,"shape":[64,1280],"fractional_bits":20}],
              "rope_order":"coordinate outer, position inner; transpose of JSON [position][coordinate]",
              "negative_encoding":"integer modulo 2^59","roundtrip_all_words_checked":True}
    (root/"native_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"path":str(path),"bytes":len(data),"sha256":manifest["sha256"]}))


if __name__=="__main__":export()
