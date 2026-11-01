#  following process from https://docs.ultralytics.com/usage/python
from modules import load_initial_model, load_trained
from config import DATA_DIR, BASE_MODEL, IMGSZ, EPOCHS, BATCH, WORKERS, SEED, write_data_yaml, PROJECT_NAME, BOX, DFL, CLS

def main():
    data_yaml = write_data_yaml() # creation of data_yaml file - now automated
    model = load_initial_model(BASE_MODEL) # loads pre-trained YOLOv8

    # training of model
    results = model.train(# documentation at https://docs.ultralytics.com/modes/train
        data = str(DATA_DIR / "data.yaml"), 
        epochs = EPOCHS,
        imgsz = IMGSZ,
        batch = BATCH,
        device = 0, # GPU 
        workers = WORKERS, # Rangpur recommendation - avoids slowdown/freeze
	    box = BOX,
	    dfl = DFL,
        cls = CLS,
        seed = SEED, #  for fair comparison
        project = "runs", # output directory
        name = PROJECT_NAME# subfolder in output directory using pre-defined name in config.py
    )

    

    best = results.save_dir / "weights" / "best.pt"
    
    trained = load_trained(best)  # loading the BEST trained model
    val_metrics = trained.val(data = str(DATA_DIR / "data.yaml"), split = "val", imgsz = IMGSZ, workers = WORKERS)

    # NOTE: Written by Claude Sonnet 5.0 (Medium)
    print(f"Val: mAP50 {val_metrics.box.map50:.3f}, mAP50-95 {val_metrics.box.map:.3f}, "
          f"P {val_metrics.box.mp:.3f}, R {val_metrics.box.mr:.3f}")
    

if __name__ == "__main__":
    main()

