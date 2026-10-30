import os, sys
import torch
import numpy as np
from tqdm import tqdm #to help visualise loading
import time
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
 
from dataset import get_dataloaders, NUM_CLASSES, PROSTATE_CLASS, pad_collate
from modules import UNet2D, DiceLoss, dice_counts
 
#define HIPMRI_ROOT in job file when running on remote HPC
ROOT = os.environ.get("HIPMRI_ROOT", "/Users/alyxgoller/data/HipMRI_data")  #changes per machine
WEIGHTS = "best_model.pth"
NUM_OF_WORKERS = 0   # raise to around 4 on the HPC


#set up gpu access
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Warning: CUDA not found. Using CPU")

@torch.no_grad()
def general_predict(batch_size = 1):
    """
    Main test function for segmenting images and evaluating performance and cost
    
    """
    
    
 
    # -- fetch test images & their labels --
    _, _, test_loader = get_dataloaders(ROOT, batch_size=batch_size, num_workers=NUM_OF_WORKERS)
 
    #prepare model and load weights
    model = UNet2D(in_channels=1, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval() #turn off training mode/flag
 
    # -- predict and evaluate --
    best_dice = 0
    worst_dice = np.inf
    best_batch = ()
    worst_batch = ()
    total = 0.0
    intersection = torch.zeros(NUM_CLASSES, device=device)
    total = torch.zeros(NUM_CLASSES, device=device)   
    prostate_size = 0  #number of true prostate pixels in the whole test set 
    #(needed for the over / under contoured percentages)
    for image, label in tqdm(test_loader, desc="test", leave=False, disable=not sys.stdout.isatty()):
        image, label = image.to(device), label.to(device)
        
        #predict bacth
        predicted_mask = model(image)   # raw scores (logits) - argmax later turns them into a mask
        
        # -- Evaluating --
 
        #store overall dice data for averaging - for every class
        i, t = dice_counts(predicted_mask, label)
        intersection += i
        total += t
        prostate_size += label[:, PROSTATE_CLASS].sum()
 
        #determine if best or worst dice score so far & store the best and worst batches for final visualisation
        #(compared on the prostate class only, so each batch gives one number)
        cur_batch_dice = ((2 * i[PROSTATE_CLASS] + 1e-5) / (t[PROSTATE_CLASS] + 1e-5)).item()
        if label[:, PROSTATE_CLASS].sum() == 0:
            continue   # a batch with no prostate gives a meaningless score, so it can't be best or worst
        if cur_batch_dice > best_dice:
            best_dice = cur_batch_dice
            best_batch = (image, label, predicted_mask, i, t)
        if cur_batch_dice < worst_dice:
            worst_dice = cur_batch_dice
            worst_batch = (image, label, predicted_mask, i, t)
 
    # -- Reporting --
    ave_dice = (2 * intersection + 1e-5) / (total + 1e-5)
    
    print("Dice figures:")
    print("Average (per class): ", ave_dice, " Best batch prostate Dice: ", best_dice, " Worst batch prostate Dice: ", worst_dice)
    print("Prostate average: ", ave_dice[PROSTATE_CLASS].item())
 
    #calculate percentage of over & under contouring of the whole test set
    #total = predicted + true pixels -> predicted = total - true
    #false_pos = predicted - correct, false_neg = true - correct)
    if prostate_size > 0:
        false_pos = total[PROSTATE_CLASS] - prostate_size - intersection[PROSTATE_CLASS]
        false_neg = prostate_size - intersection[PROSTATE_CLASS]
        print(f"Over-contoured (false positive): {100 * false_pos.item() / prostate_size.item():.1f}% of the true prostate")
        print(f"Under-contoured (false negative): {100 * false_neg.item() / prostate_size.item():.1f}% of the true prostate")
 
    if len(best_batch) == 0:
        print("No test batch contained prostate, nothing to visualise")
        return
 
    #visualise slices from best and worst batches
    visualise(best_batch, worst_batch, num_of_slices = 3, prostate = True)



@torch.no_grad()
def imbalance_test(num_slices = 60, indexes = None):
    """
    Visualise how the model prediction precision changes with different amounts of prostate (changing class balances).
    Uses one patients slices - finds the min (but not zero), max and no prostate instances from the set of patient slices
 
    Args:
        num_slices = how many slices from the start of the test set to search (the first 60 are one patient in the test file)
        indexes = optional dictionary input in form {"name": slice index} - only these indexes will be visualised for hard coding 
                  e.g. {"MAX prostate": 25, "MIN prostate": 8, "NO prostate": 0}
    """
 
    # -- fetch test slices and prepare the model, as in general_predict --
    _, _, test_loader = get_dataloaders(ROOT, batch_size=16, num_workers=NUM_OF_WORKERS, verbose=False)
    ds = test_loader.dataset
    model = UNet2D(in_channels=1, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()
 
    # -- find the slices: count the prostate pixels in every label --
    if indexes is None:
        sizes = [int(ds[s][1][PROSTATE_CLASS].sum()) for s in range(num_slices)]   # ds[s] = (image, one-hot label)
        print("Prostate pixels in each slice index:", sizes)
        with_prostate = [s for s in range(num_slices) if sizes[s] > 0]
        if len(with_prostate) == 0 or 0 not in sizes:
            print("Need slices with and without prostate - hard code the indexes instead")
            return
        indexes = {"MAX prostate": max(with_prostate, key=lambda s: sizes[s]),
                   "MIN prostate": min(with_prostate, key=lambda s: sizes[s]),
                   "NO prostate": sizes.index(0)}
 
    # -- predict those slices together as one batch --
    image, label = pad_collate([ds[s] for s in indexes.values()])
    image, label = image.to(device), label.to(device)
    predicted_mask = model(image)
    true = label.argmax(dim=1).cpu().numpy()             # one-hot channels -> one class number per pixel
    pred = predicted_mask.argmax(dim=1).cpu().numpy()    # highest scoring class per pixel
    image = image.cpu().numpy()
 
    # -- report each slice and collect the rows for plotting --
    rows = []
    for k, (name, s) in enumerate(indexes.items()):
        fp, fn, over, under = contour_errors(pred[k], true[k])
        print(f"{name} (slice index {s}):")
        print("   Dice per class:", [round(d, 3) for d in dice_per_class(pred[k], true[k])])
        print(f"   Over-contoured (false +): {fp} px = {over:.1f}% of prostate   Under-contoured (false -): {fn} px = {under:.1f}% of prostate")
        rows.append((f"{name}\nslice index {s}", image[k, 0], true[k], pred[k]))
    show_figure(draw_rows(rows), "imbalance_test.png")
 
 
 
 




# -- Visualisation Helpers --
 
def dice_per_class(pred, true):
    """Calculate Dice of every class for ONE slice.
        
       Args:
           pred = predicted segmentation
           true = true mask - segmented label of image
        Returns:
            List of Dice scores for each class in the slice - nan for classes not present in slice
        """
    scores = []
    for c in range(NUM_CLASSES):
        both = ((pred == c) & (true == c)).sum()
        total = (pred == c).sum() + (true == c).sum()
        scores.append(float("nan") if total == 0 else float(2 * both / total))
    return scores
 
 
def contour_errors(pred, true):
    """False positive and false negative prostate pixels in ONE slice, and each as a percentage of the true prostate size.
       False positive % = over-contoured, false negative % = under-contoured.
 
       Args:
           pred = predicted segmentation
           true = true mask - segmented label of image
       Returns:
           (false positive pixels, false negative pixels, over-contoured %, under-contoured %)
           the percentages are nan if there is no prostate in the label (nothing to take a percentage of)
    """
    p, t = pred == PROSTATE_CLASS, true == PROSTATE_CLASS
    fp, fn, size = int((p & ~t).sum()), int((~p & t).sum()), int(t.sum())
    if size == 0:
        return fp, fn, float("nan"), float("nan")
    return fp, fn, 100 * fp / size, 100 * fn / size
 
 
def pick_slices(batch, title, num_of_slices, prostate):
    """Obtain key information for plotting and add to easily accessible list:
        Num_of_slices slices from one stored batch -> list of (title, image, true mask, predicted mask)
 
        Args:
            batch = store information (image, label, predicted mask of image, intersection, total area)
            title = name of image - either BEST or WORST in case of predict plotting
            num_of_slices = the number of slices that are to be plotted/visualised
            prostate (Bool) = True if only selecting slices with a prostate present in label, else False (ignore missing prostate)
 
        Return:
            List of (title, image, true mask, predicted mask)
    """
    image, label, predicted_mask, i, t = batch
    image = image.cpu().numpy()
    true = label.argmax(dim=1).cpu().numpy()             # one-hot channels -> one class number per pixel
    pred = predicted_mask.argmax(dim=1).cpu().numpy()    # highest scoring class per pixel
    batch_dice = ((2 * i[PROSTATE_CLASS] + 1e-5) / (t[PROSTATE_CLASS] + 1e-5)).item()
 
    rows = []
    for s in range(len(image)):
        # skip slices where the prostate isn't in the label
        if prostate and not (true[s] == PROSTATE_CLASS).any():
            continue   
        
        rows.append((f"{title} batch (prostate Dice {batch_dice:.3f})\nslice {s}", image[s, 0], true[s], pred[s]))
        if len(rows) == num_of_slices:
            break
    return rows
 
 
#Visualising Functions written primarily by AI
 
def draw_rows(rows):
    """One figure, one row per slice: input | ground truth | prediction | prostate errors | Dice per class"""
    class_colours = ["black", "tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]
    classes = ListedColormap(class_colours)
    error_colours = ListedColormap(["black", "green", "red", "blue"])
    height = 5.5 * len(rows)
    fig = Figure(figsize=(14, height), dpi=100)   # tall figure, scrolled in the window
 
    for r, (title, image, true, pred) in enumerate(rows):
        ax = fig.add_subplot(len(rows), 5, 5 * r + 1)
        ax.imshow(image, cmap="gray")
        ax.set_title("Input")
 
        ax = fig.add_subplot(len(rows), 5, 5 * r + 2)
        ax.imshow(true, cmap=classes, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        ax.set_title("Ground truth")
 
        ax = fig.add_subplot(len(rows), 5, 5 * r + 3)
        ax.imshow(pred, cmap=classes, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        ax.set_title("Prediction")
 
        #prostate errors: 1 = correct (green), 2 = false positive (red), 3 = missed (blue), 0 = nothing (hidden)
        p, t = pred == PROSTATE_CLASS, true == PROSTATE_CLASS
        errors = 1 * (p & t) + 2 * (p & ~t) + 3 * (~p & t)
        ax = fig.add_subplot(len(rows), 5, 5 * r + 4)
        ax.imshow(image, cmap="gray")
        ax.imshow(np.ma.masked_equal(errors, 0), cmap=error_colours, vmin=0, vmax=3, alpha=0.8, interpolation="nearest")
        ax.set_title("Prostate: green correct,\nred false +, blue missed")
 
        ax = fig.add_subplot(len(rows), 5, 5 * r + 5)
        text = title + "\n\nDice per class\n"
        for c, d in enumerate(dice_per_class(pred, true)):
            text += f"class {c}: {d:.3f}" + ("  <- prostate" if c == PROSTATE_CLASS else "") + "\n"
        #over / under contoured prostate, as a % of the true prostate (nan = no prostate in this slice's label)
        fp, fn, over, under = contour_errors(pred, true)
        text += f"\nOver-contoured:\n  false + {fp} px\n  = {over:.1f}%\nUnder-contoured:\n  false - {fn} px\n  = {under:.1f}%\n"
        ax.text(0, 1, text, va="top", family="monospace", fontsize=11)
 
    for ax in fig.axes:
        ax.axis("off")
    fig.legend(handles=[Patch(color=col, label=f"class {c}") for c, col in enumerate(class_colours)],
               loc="upper center", ncol=NUM_CLASSES)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.0 / height, bottom=0.01, wspace=0.05, hspace=0.2)
    return fig
 
 
def show_scrollable(fig):
    """Put the figure in a window with a scrollbar (mouse wheel works too)"""
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
 
    window = tk.Tk()
    window.geometry("1300x800")
    canvas = tk.Canvas(window)
    scrollbar = tk.Scrollbar(window, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
 
    figure_canvas = FigureCanvasTkAgg(fig, master=canvas)
    widget = figure_canvas.get_tk_widget()
    canvas.create_window((0, 0), window=widget, anchor="nw")
    figure_canvas.draw()
    canvas.configure(scrollregion=(0, 0, widget.winfo_reqwidth(), widget.winfo_reqheight()))
 
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))  # Mac / Windows
    canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))                           # Linux up
    canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))                            # Linux down
    window.mainloop()   # stays open until the window is closed
 
 
def show_figure(fig, png_name):
    """Show the figure in the scrollable window, or save it as a picture if no window is possible (e.g. terminal-only HPC)"""
    try:
        show_scrollable(fig)
    except Exception:
        fig.savefig(png_name, bbox_inches="tight")
        print(f"No window available, saved {png_name} instead")
 
 
def visualise(best_batch, worst_batch, num_of_slices = 3, prostate = True):
    """Show num_of_slices slices from the best batch, then from the worst batch, in one scrollable window.
    prostate = True -> only slices where the prostate is in the label"""
    rows = pick_slices(best_batch, "BEST", num_of_slices, prostate) + pick_slices(worst_batch, "WORST", num_of_slices, prostate)
    fig = draw_rows(rows)
    show_figure(fig, "best_and_worst.png")
 
 
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "imbalance":   # python predict.py imbalance
        imbalance_test(60, indexes = {"BEST": 1, "MAX PROSTATE": 20, "BASE": 50})
    else:
        general_predict(batch_size=1)
 