import matplotlib.pyplot as plt

CLASS_NAMES = ["background", "body", "skeleton", "bladder", "rectum", "prostate"]

with open("../results/can2d_20epoch.out", "r") as f:
	text = f.read()

loss = []
fg_dice = []
per_class = []

for line in text.splitlines():
	if "val fg-dice" in line and "per-class" in line:
		loss_value = line.split("loss")[1].split()[0]
		loss.append(float(loss_value))

		fg = line.split("val fg-dice")[1]
		fg_dice.append(float(fg.split()[0]))

		classes = line.split("per-class")[1]
		classes = classes.split("]")[0]
		classes = classes.strip(" [")

		per_class.append([float(x) for x in classes.split(",")])

epochs = range(1, len(fg_dice) + 1)

fig, ax = plt.subplots(nrows=2, tight_layout=True, figsize=(5, 8))

""" Plot loss"""
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].plot(epochs, loss, label="Loss")
ax[0].set_title("CAN2D Training")
ax[0].set_xticks(range(0, len(fg_dice) + 1, 5))
ax[0].legend()

""" Plot per-class dice """
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Dice")
ax[1].plot(epochs, fg_dice, label="Average")
ax[1].set_xticks(range(0, len(fg_dice) + 1, 5))
for i in range(1, len(CLASS_NAMES)):
	ax[1].plot(
		epochs,
		[row[i] for row in per_class],
		"--",
		label=CLASS_NAMES[i]
	)
ax[1].legend()
ax[1].grid(True)

plt.savefig("can2d_training")
# plt.show()