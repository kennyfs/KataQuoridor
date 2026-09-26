import os
from typing import Optional, Dict
import torch
import torch.nn

try:
    import onnx
except ImportError:
    onnx = None


class QuoridorOnnxExportWrapper(torch.nn.Module):
    """
    Wrapper for KataQuoridor neural net inference export.
    Exposes InputSpatial (B, 17, 9, 9) and InputGlobal (B, 15, 1, 1) as inputs,
    and OutputPolicy (B, 3, 9, 9) and OutputValue (B, 2, 1, 1) as outputs.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_spatial: torch.Tensor, input_global: torch.Tensor):
        # input_spatial: (B, 17, 9, 9)
        # input_global:  (B, 15, 1, 1) or (B, 15)
        if input_global.dim() == 4:
            input_global_2d = input_global.view(input_global.shape[0], -1)
        else:
            input_global_2d = input_global
        outputs_byheads = self.model(input_spatial, input_global_2d)
        # Extract main heads
        policy_out = outputs_byheads[0][0]  # (B, 18, 9, 9)
        value_out = outputs_byheads[0][1]   # (B, 2)

        # Extract strictly Target 0 (first 3 channels: pawn, v-wall, h-wall)
        raw_policy = policy_out[:, 0:3, :, :].contiguous()  # (B, 3, 9, 9)
        raw_value = value_out.view(value_out.shape[0], 2, 1, 1).contiguous()  # (B, 2, 1, 1)
        return raw_policy, raw_value


def export_quoridor_onnx(
    model: torch.nn.Module,
    export_path: str,
    model_name: str = "kataquoridor",
    opset_version: int = 18,
    verbose: bool = False,
) -> str:
    """
    Exports a KataQuoridor PyTorch model to ONNX format with KataGo metadata props.
    """
    model.eval()
    wrapper = QuoridorOnnxExportWrapper(model)
    wrapper.eval()

    device = next(model.parameters()).device
    dummy_spatial = torch.zeros(1, 17, 9, 9, dtype=torch.float32, device=device)
    dummy_global = torch.zeros(1, 15, 1, 1, dtype=torch.float32, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_spatial, dummy_global),
        export_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["InputSpatial", "InputGlobal"],
        output_names=["OutputPolicy", "OutputValue"],
        dynamic_axes={
            "InputSpatial": {0: "batch"},
            "InputGlobal": {0: "batch"},
            "OutputPolicy": {0: "batch"},
            "OutputValue": {0: "batch"},
        },
        verbose=verbose,
    )

    if onnx is not None:
        model_proto = onnx.load(export_path)

        meta: Dict[str, str] = {
            "katago.metadataVersion": "1",
            "katago.name": model_name,
            "katago.modelVersion": "1",
            "katago.numInputChannels": "17",
            "katago.numInputGlobalChannels": "15",
            "katago.numInputMetaChannels": "0",
            "katago.numPolicyChannels": "3",
            "katago.numValueChannels": "2",
            "katago.numScoreValueChannels": "0",
            "katago.numOwnershipChannels": "0",
            "katago.build.nnXLen": "9",
            "katago.build.nnYLen": "9",
            "katago.build.requireExactNNLen": "true",
        }

        # Clear existing metadata if present
        del model_proto.metadata_props[:]
        for k, v in meta.items():
            entry = model_proto.metadata_props.add()
            entry.key = k
            entry.value = v

        onnx.checker.check_model(model_proto)
        onnx.save(model_proto, export_path)

    return export_path
