"""Stack the per-model failure panels into one figure per case (README autopsies)."""
from PIL import Image

# Each case stacks each model's worst slice: 2D U-Net, 2D Improved U-Net, 3D Improved U-Net.
# File names are <model>_<case>_z<slice>.png, as written by predict.py --figures.
CASES = {
    "1_W029_Week5": ["unet2d_W029_Week5_z70", "improved2d_W029_Week5_z68", "improved3d_W029_Week5_z65"],
    "2_G021_Week5": ["unet2d_G021_Week5_z61", "improved2d_G021_Week5_z60", "improved3d_G021_Week5_z65"],
    "3_G021_Week7": ["unet2d_G021_Week7_z43", "improved2d_G021_Week7_z54", "improved3d_G021_Week7_z62"],
    "4_T009_Week0": ["unet2d_T009_Week0_z48", "improved2d_T009_Week0_z48", "improved3d_T009_Week0_z76"],
}

# Paste the three panels under each other on a white sheet.
for name, panels in CASES.items():
    ims = [Image.open(f"images/{p}.png").convert("RGB") for p in panels]
    sheet = Image.new("RGB", (max(i.width for i in ims), sum(i.height for i in ims)), "white")
    y = 0
    for im in ims:
        sheet.paste(im, (0, y))
        y += im.height
    sheet.save(f"images/autopsy_{name}.png")
    print("saved", f"images/autopsy_{name}.png", sheet.size)
