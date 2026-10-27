"""Render a source-checked architecture diagram without importing model code.

This authoring helper parses modules.py as text/AST only. It never imports
PyTorch or project modules, loads data/checkpoints, or executes a model. Both
SVG and PNG are drawn from the same vector primitives. All visible additions
use purple text. Outputs are exclusively created and previous files preserved.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import sys
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parent
MODULES = PROJECT / "modules.py"
STEM = PROJECT / "evidence/model_architecture_20261026"
WIDTH, HEIGHT = 1400, 1600
PURPLE = "#800080"
EDGE = "#92769c"
BORDER = "#c7b7ce"
PANEL = "#faf7fc"
INK = PURPLE


def require(condition, message):
    if not condition:
        raise ValueError(message)


def structural(value):
    return ast.dump(value, include_attributes=False)


def parsed(expression):
    return ast.parse(expression, mode="eval").body


def source_facts(source):
    """Check every channel/operator fact represented by the diagram using AST."""
    tree = ast.parse(source, filename="modules.py")
    classes = {item.name: item for item in tree.body if isinstance(item, ast.ClassDef)}
    checks = []

    def method(name, function):
        node = next((item for item in classes[name].body
                     if isinstance(item, ast.FunctionDef) and item.name == function), None)
        require(node is not None, f"Missing {name}.{function}")
        return node

    def assignment(name, function, target, expected):
        function_node = method(name, function)
        assignments = [item for item in function_node.body if isinstance(item, ast.Assign)
                       and any(structural(left) == structural(ast.parse(target + " = None").body[0].targets[0])
                               for left in item.targets)]
        require(len(assignments) == 1, f"Expected one assignment: {name}.{function}:{target}")
        require(structural(assignments[0].value) == structural(parsed(expected)),
                f"Architecture changed: {name}.{function}:{target}")
        checks.append(f"{name}.{function}:{target} = {expected}")

    def returns(name, function, expression):
        values = [item.value for item in method(name, function).body
                  if isinstance(item, ast.Return)]
        require(len(values) == 1 and structural(values[0]) == structural(parsed(expression)),
                f"Architecture changed: {name}.{function} return")
        checks.append(f"{name}.{function}:return")

    for model in ("UNet", "ImprovedUNet"):
        init = method(model, "__init__")
        require([item.arg for item in init.args.args] ==
                ["self", "in_channels", "num_classes", "base_channels"],
                "Model constructor signature changed")
        require([ast.literal_eval(item) for item in init.args.defaults] == [1, 6, 16],
                "Model defaults changed")
        required = [item for item in classes[model].body if isinstance(item, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "required_multiple"
                            for target in item.targets)]
        require(len(required) == 1 and ast.literal_eval(required[0].value) == 16,
                "Input multiple changed")
        checks.append(f"{model}:defaults=(1,6,16),required_multiple=16")

    for index, incoming, outgoing in ((1, "in_channels", "width"),
                                     (2, "width", "width * 2"),
                                     (3, "width * 2", "width * 4"),
                                     (4, "width * 4", "width * 8")):
        assignment("UNet", "__init__", f"self.enc{index}",
                   f"DoubleConv({incoming}, {outgoing})")
    assignment("UNet", "__init__", "self.pool", "nn.MaxPool2d(kernel_size=2, stride=2)")
    assignment("UNet", "__init__", "self.bottleneck", "DoubleConv(width * 8, width * 16)")
    for index, previous, channels in ((4, 16, 8), (3, 8, 4), (2, 4, 2), (1, 2, 1)):
        output = "width" if channels == 1 else f"width * {channels}"
        assignment("UNet", "__init__", f"self.up{index}",
                   f"nn.ConvTranspose2d(width * {previous}, {output}, kernel_size=2, stride=2)")
        assignment("UNet", "__init__", f"self.dec{index}",
                   f"DoubleConv(width * {channels * 2}, {output})")
    assignment("UNet", "__init__", "self.output", "nn.Conv2d(width, num_classes, kernel_size=1)")
    returns("UNet", "forward", "self.output(decoded1)")
    for index, previous in ((4, "encoded"), (3, "decoded4"), (2, "decoded3"), (1, "decoded2")):
        assignment("UNet", "forward", f"decoded{index}",
                   f"self.dec{index}(torch.cat((self.up{index}({previous}), skip{index}), dim=1))")
    assignment("DoubleConv", "__init__", "self.layers",
               "nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True))")

    assignment("ImprovedUNet", "__init__", "self.stem", "nn.Conv2d(in_channels, width, kernel_size=3, padding=1)")
    for index, factor in ((1, 1), (2, 2), (3, 4), (4, 8), (5, 16)):
        channel = "width" if factor == 1 else f"width * {factor}"
        assignment("ImprovedUNet", "__init__", f"self.context{index}",
                   f"ResidualContext({channel})")
        if index > 1:
            previous = "width" if factor == 2 else f"width * {factor // 2}"
            assignment("ImprovedUNet", "__init__", f"self.down{index}",
                       f"PreActConv({previous}, {channel}, stride=2)")
    for index, incoming, factor in ((4, 16, 8), (3, 8, 4), (2, 4, 2), (1, 2, 1)):
        outgoing = "width" if factor == 1 else f"width * {factor}"
        assignment("ImprovedUNet", "__init__", f"self.up{index}",
                   f"NearestUpConv(width * {incoming}, {outgoing})")
        assignment("ImprovedUNet", "__init__", f"self.localize{index}",
                   f"LocalizationBlock(width * {factor * 2}, {outgoing})")
        prior = "encoded" if index == 4 else f"decoded{index + 1}"
        assignment("ImprovedUNet", "forward_heads", f"decoded{index}",
                   f"self.localize{index}(torch.cat((self.up{index}({prior}), skip{index}), dim=1))")
    for name, factor in (("quarter", 4), ("half", 2), ("full", 1)):
        incoming = "width" if factor == 1 else f"width * {factor}"
        assignment("ImprovedUNet", "__init__", f"self.head_{name}",
                   f"nn.Conv2d({incoming}, num_classes, kernel_size=1)")
    returns("ImprovedUNet", "forward_heads",
            "{'quarter': self.head_quarter(decoded3), 'half': self.head_half(decoded2), 'full': self.head_full(decoded1)}")
    assignment("ImprovedUNet", "forward", "half",
               "heads['half'] + torch.nn.functional.interpolate(heads['quarter'], size=heads['half'].shape[-2:], mode='nearest')")
    returns("ImprovedUNet", "forward",
            "heads['full'] + torch.nn.functional.interpolate(half, size=heads['full'].shape[-2:], mode='nearest')")
    assignment("PreActConv", "__init__", "self.norm",
               "nn.InstanceNorm2d(in_channels, affine=True, track_running_stats=False)")
    assignment("PreActConv", "__init__", "self.activation",
               "nn.LeakyReLU(negative_slope=0.01, inplace=False)")
    returns("PreActConv", "forward", "self.conv(self.activation(self.norm(values)))")
    assignment("ResidualContext", "__init__", "self.dropout", "nn.Dropout2d(p=0.3)")
    returns("ResidualContext", "forward", "values + self.second(self.dropout(self.first(values)))")
    assignment("NearestUpConv", "forward", "upsampled",
               "torch.nn.functional.interpolate(values, scale_factor=2, mode='nearest')")
    assignment("LocalizationBlock", "__init__", "self.spatial", "PreActConv(in_channels, in_channels)")
    assignment("LocalizationBlock", "__init__", "self.channel", "PreActConv(in_channels, out_channels, kernel_size=1)")
    return {"configuration": {"in_channels": 1, "num_classes": 6, "base_channels": 16,
                               "spatial_multiple": 16},
            "encoder_channels": [16, 32, 64, 128, 256],
            "decoder_output_channels": [128, 64, 32, 16],
            "head_feature_channels": {"quarter": 64, "half": 32, "full": 16},
            "fusion": "half + nearest(quarter); full + nearest(fused_half)",
            "ast_pattern_check_count": len(checks), "ast_pattern_checks": checks}


class Drawing:
    def __init__(self):
        self.items = []

    def rectangle(self, x, y, w, h, fill="white", radius=12):
        self.items.append({"type": "rect", "x": x, "y": y, "w": w, "h": h,
                           "fill": fill, "radius": radius, "stroke": BORDER})

    def text(self, x, y, value, size=24, bold=False, anchor="left"):
        self.items.append({"type": "text", "x": x, "y": y, "value": value,
                           "size": size, "bold": bold, "anchor": anchor, "fill": INK})

    def lines(self, x, y, values, size=24, step=32, bold=False, anchor="left"):
        for index, value in enumerate(values):
            self.text(x, y + index * step, value, size, bold, anchor)

    def arrow(self, points, dashed=False):
        self.items.append({"type": "arrow", "points": points, "dashed": dashed,
                           "stroke": EDGE})

    def box(self, x, y, w, h, title, lines):
        self.rectangle(x, y, w, h)
        self.text(x + 16, y + 17, title, size=25, bold=True)
        self.lines(x + 16, y + 58, lines, size=23, step=30)


def diagram():
    d = Drawing()
    d.text(50, 30, "HipMRI 2D models: source-checked information flow", size=39, bold=True)
    d.text(50, 80, "Default experiment widths: base=16, input=1 channel, output=6 class logits", size=25)
    d.rectangle(40, 125, 1320, 112, PANEL)
    d.lines(60, 144, ["SHARED INPUT  |  MRI [B,1,H,W], H/W padded to multiples of 16",
                      "Per-slice min-max; bottom/right zero padding; valid mask excludes artificial padding.",
                      "Class IDs stay unchanged; the loss encodes [B,6,H,W] one-hot targets."], size=24, step=28)

    d.rectangle(40, 266, 1320, 369, PANEL)
    d.text(60, 282, "BASELINE U-NET", size=30, bold=True)
    d.text(60, 321, "DoubleConv = two same-padding 3x3 Conv + ReLU blocks", size=23)
    d.box(65, 384, 410, 163, "Encoder / MaxPool x2", ["16@1 -> 32@1/2 -> 64@1/4", "-> 128@1/8 (four skip scales)", "@ denotes spatial resolution vs. input"])
    d.box(515, 384, 220, 163, "Bottleneck", ["256 channels", "H/16 x W/16", "DoubleConv"])
    d.box(775, 384, 560, 163, "Decoder: transpose-conv x2 + skip concat", ["DoubleConv outputs, deepest to shallowest:", "128@1/8 -> 64@1/4 -> 32@1/2 -> 16@1", "Final 1x1 Conv: 16 -> 6 channels"])
    d.arrow([(475, 466), (515, 466)])
    d.arrow([(735, 466), (775, 466)])
    d.arrow([(270, 384), (270, 364), (1055, 364), (1055, 384)], dashed=True)
    d.text(665, 340, "same-scale skips (4)", size=22, anchor="center")
    d.text(700, 580, "OUTPUT: one [B,6,H,W] raw-logits tensor", size=28, bold=True, anchor="center")

    d.rectangle(40, 665, 1320, 611, PANEL)
    d.text(60, 682, "IMPROVED U-NET  |  2D residual / multi-scale adaptation", size=30, bold=True)
    d.text(60, 723, "PreAct = affine InstanceNorm (no running stats) -> LeakyReLU(0.01) -> Conv", size=23)
    d.text(60, 754, "ResidualContext = input + [PreActConv -> Dropout2d(0.3) -> PreActConv]", size=23)
    d.box(65, 827, 410, 178, "Encoder / stride-2 Conv", ["3x3 stem: 1 -> 16; residual context", "16@1 -> 32@1/2 -> 64@1/4", "-> 128@1/8 (four skip scales)", "Residual context at every scale"])
    d.box(515, 827, 220, 178, "Bottleneck", ["256 channels", "H/16 x W/16", "5th residual", "context scale"])
    d.box(775, 827, 560, 178, "Nearest x2 -> PreActConv -> concat skip", ["Localization: 3x3 mix -> 1x1 channel halve", "Decoder outputs: 128@1/8 -> 64@1/4", "-> 32@1/2 -> 16@1", "Three 1x1 heads tap the last three scales"])
    d.arrow([(475, 916), (515, 916)])
    d.arrow([(735, 916), (775, 916)])
    d.arrow([(270, 827), (270, 807), (1055, 807), (1055, 827)], dashed=True)
    d.text(665, 782, "same-scale skips (4)", size=22, anchor="center")
    for x, title, lines in [(65, "Quarter head: from C64", ["q = [B,6,H/4,W/4]"]),
                             (495, "Half head: from C32", ["h = [B,6,H/2,W/2]"]),
                             (925, "Full head: from C16", ["f = [B,6,H,W]"])]:
        d.box(x, 1052, 410, 89, title, lines)
    d.arrow([(1055, 1005), (1055, 1027), (270, 1027), (270, 1052)])
    d.arrow([(1055, 1027), (700, 1027), (700, 1052)])
    d.arrow([(1055, 1027), (1130, 1027), (1130, 1052)])
    d.rectangle(65, 1180, 1270, 69)
    d.text(700, 1193, "Fused logits = f + nearest_x2( h + nearest_x2(q) )", size=28, bold=True, anchor="center")
    d.text(700, 1229, "One full-resolution output; one shared CE; no auxiliary loss or final softmax", size=22, anchor="center")
    d.arrow([(270, 1141), (270, 1180)])
    d.arrow([(700, 1141), (700, 1180)])
    d.arrow([(1130, 1141), (1130, 1180)])

    d.rectangle(40, 1302, 1320, 151, PANEL)
    d.text(60, 1321, "SHARED LOSS AND EVALUATION", size=29, bold=True)
    d.lines(60, 1365, ["Train: valid-region one-hot CE -> backward / Adam update.",
                       "Validation/test: eval + no gradient / optimizer updates; argmax -> valid-pixel confusion -> Dice / IoU.",
                       "Both models use the same preprocessing, batching, valid support and evaluation rules."], size=23, step=29)
    d.lines(50, 1470, ["Source: modules.py (static AST checks only). No model import, forward pass, data or checkpoint read.",
                      "Improved is inspired by Isensee et al. (2018); this is not the complete 3D paper recipe.",
                      "This compares complete models with unequal parameter counts; no module-wise ablation claim.",
                      "Schematic groups layers; tensor spatial labels are relative to the padded model input."], size=23, step=28)
    return d


def head_points(points):
    end_x, end_y = points[-1]
    previous_x, previous_y = points[-2]
    angle = math.atan2(end_y - previous_y, end_x - previous_x)
    size = 10
    return [(end_x, end_y),
            (end_x - size * math.cos(angle - math.pi / 6),
             end_y - size * math.sin(angle - math.pi / 6)),
            (end_x - size * math.cos(angle + math.pi / 6),
             end_y - size * math.sin(angle + math.pi / 6))]


def svg_bytes(drawing):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
           '<title>HipMRI baseline and Improved U-Net architecture</title>',
           '<desc>Static source-checked schematic; no model execution. All new text is purple.</desc>',
           f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>']
    for item in drawing.items:
        if item["type"] == "rect":
            out.append(f'<rect x="{item["x"]}" y="{item["y"]}" width="{item["w"]}" height="{item["h"]}" rx="{item["radius"]}" fill="{item["fill"]}" stroke="{item["stroke"]}" stroke-width="2"/>')
        elif item["type"] == "text":
            anchor = {"left": "start", "center": "middle"}[item["anchor"]]
            out.append(f'<text x="{item["x"]}" y="{item["y"]}" dominant-baseline="text-before-edge" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif" font-size="{item["size"]}" font-weight="{700 if item["bold"] else 400}" fill="{item["fill"]}">{escape(item["value"])}</text>')
        else:
            points = " ".join(f"{x},{y}" for x, y in item["points"])
            dashed = ' stroke-dasharray="9 6"' if item["dashed"] else ""
            out.append(f'<polyline points="{points}" fill="none" stroke="{item["stroke"]}" stroke-width="2.5"{dashed}/>')
            triangle = " ".join(f"{x:.2f},{y:.2f}" for x, y in head_points(item["points"]))
            out.append(f'<polygon points="{triangle}" fill="{item["stroke"]}"/>')
    out.append('</svg>')
    return ("\n".join(out) + "\n").encode("utf-8")


def png_bytes(drawing):
    fonts = Path("C:/Windows/Fonts")
    regular, bold = fonts / "arial.ttf", fonts / "arialbd.ttf"
    require(regular.is_file() and bold.is_file(), "Arial fonts required for this documented renderer")
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    canvas = ImageDraw.Draw(image)
    font_cache = {}
    for item in drawing.items:
        if item["type"] == "rect":
            canvas.rounded_rectangle([item["x"], item["y"], item["x"] + item["w"], item["y"] + item["h"]],
                                     radius=item["radius"], fill=item["fill"], outline=item["stroke"], width=2)
        elif item["type"] == "text":
            key = (item["size"], item["bold"])
            font = font_cache.setdefault(key, ImageFont.truetype(str(bold if item["bold"] else regular), item["size"]))
            left = item["x"]
            if item["anchor"] == "center":
                left -= canvas.textlength(item["value"], font=font) / 2
            bounds = canvas.textbbox((left, item["y"]), item["value"], font=font, anchor="lt")
            require(bounds[0] >= 0 and bounds[2] <= WIDTH - 20 and bounds[3] <= HEIGHT - 15,
                    "Figure text exceeds canvas: " + item["value"])
            canvas.text((left, item["y"]), item["value"], font=font, fill=item["fill"], anchor="lt")
        else:
            if item["dashed"]:
                for (x0, y0), (x1, y1) in zip(item["points"], item["points"][1:]):
                    length = math.hypot(x1 - x0, y1 - y0)
                    for begin in range(0, math.ceil(length), 15):
                        end = min(begin + 9, length)
                        canvas.line([(x0 + (x1 - x0) * begin / length, y0 + (y1 - y0) * begin / length),
                                     (x0 + (x1 - x0) * end / length, y0 + (y1 - y0) * end / length)], fill=item["stroke"], width=3)
            else:
                canvas.line(item["points"], fill=item["stroke"], width=3, joint="curve")
            canvas.polygon(head_points(item["points"]), fill=item["stroke"])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def main():
    paths = [STEM.with_suffix(extension) for extension in (".svg", ".png", ".json")]
    require(not any(path.exists() for path in paths), "Refusing to overwrite existing architecture outputs")
    source = MODULES.read_bytes()
    facts = source_facts(source.decode("utf-8"))
    drawing = diagram()
    svg, png = svg_bytes(drawing), png_bytes(drawing)
    require(MODULES.read_bytes() == source, "Model source changed during authoring")
    require(not any(name in sys.modules for name in ("torch", "numpy", "nibabel", "matplotlib")),
            "A model/scientific package was unexpectedly imported")
    record = {"kind": "static_model_architecture_schematic", "created_at": datetime.now(timezone.utc).isoformat(),
              "performed_by": "Codex static source reading/AST validation and Pillow authoring",
              "source_file": "modules.py", "source_sha256": sha256(source).hexdigest(),
              "renderer_file": Path(__file__).name, "renderer_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "facts": facts, "canvas": {"width": WIDTH, "height": HEIGHT},
              "same_drawing_primitives": True, "primitive_count": len(drawing.items),
              "visible_text": [item["value"] for item in drawing.items if item["type"] == "text"],
              "all_new_text_color": PURPLE,
              "outputs": {paths[0].name: {"sha256": sha256(svg).hexdigest(), "bytes": len(svg)},
                          paths[1].name: {"sha256": sha256(png).hexdigest(), "bytes": len(png)}},
              "scope": "Default base16 2D model architecture; grouped layers and relative spatial resolution. Common pipeline captions summarize the documented shared flow; they are not inferred by executing modules.py.",
              "limitations": ["Not a full 3D Isensee recipe reproduction", "No module-wise ablation or clinical claim", "Schematic is not an executed tensor trace", "No new training/test evaluation or original experiment changes"],
              "execution": {"project_imports": 0, "model_calls": 0, "training": False, "test_evaluation": False,
                            "data_reads": 0, "checkpoint_reads": 0, "gpu_calls": 0,
                            "source_unchanged": True, "previous_outputs_overwritten": False},
              "visual_review_required": True}
    for path, content in zip(paths, (svg, png, (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))):
        with path.open("xb") as handle:
            handle.write(content)
    print(json.dumps({"outputs": [path.name for path in paths],
                      "ast_pattern_checks": facts["ast_pattern_check_count"],
                      "text_count": len(record["visible_text"]), "source_unchanged": True,
                      "model_calls": 0}))


if __name__ == "__main__":
    main()
