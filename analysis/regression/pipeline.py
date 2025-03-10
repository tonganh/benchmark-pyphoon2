from train import train
from argparse import ArgumentParser


def save_log(file, str):
    """Write brief logs for every training of the pipeline"""
    pipeline_log = open(file, "a")
    pipeline_log.write(str)
    pipeline_log.close()

if __name__ == "__main__":
    """Pipeline which directly call the train function of the train.py file, with the necessary arguments to reproduce the paper results """
    parser = ArgumentParser()
    parser.add_argument("--model_name")
    parser.add_argument("--size")
    parser.add_argument("--cropped")
    parser.add_argument("--device")
    parser.add_argument("--labels")
    parser.add_argument("--image_dirs", action="append", help="List of image directories (can specify multiple)")
    parser.add_argument("--metadata_dirs", action="append", help="List of metadata directories (can specify multiple)")
    parser.add_argument("--metadata_jsons", action="append", help="List of metadata JSON files (can specify multiple)")
    args = parser.parse_args()

    # Make sure all path arguments are defined
    if not hasattr(args, 'image_dirs') or args.image_dirs is None:
        args.image_dirs = []
    if not hasattr(args, 'metadata_dirs') or args.metadata_dirs is None:
        args.metadata_dirs = []
    if not hasattr(args, 'metadata_jsons') or args.metadata_jsons is None:
        args.metadata_jsons = []
        
    print(f"Using {len(args.image_dirs)} image directories, {len(args.metadata_dirs)} metadata directories, and {len(args.metadata_jsons)} metadata JSONs")
    
    # Check if any directories are specified
    if len(args.image_dirs) == 0 and len(args.metadata_dirs) == 0 and len(args.metadata_jsons) == 0:
        print("WARNING: No image or metadata directories specified. The dataset will be empty!")
        print("Use --image_dirs, --metadata_dirs, or --metadata_jsons arguments to specify data sources.")

    # Pipeline launched for 5 sessions training 
    for i in range(5):
        for label in ["pressure", "wind"]:
            for model in ["resnet18", "resnet50"]:
                args.model_name, args.size, args.cropped, args.device, args.labels = model, "512", False, 0, label
                train_log = train(args)
                save_log("pipeline_logs.txt", "training session " + str(i*3) + " : " + str(args) + " " + train_log + "\n")

                args.model_name, args.size, args.cropped, args.device, args.labels = model, "224", "False", 0, label
                train_log = train(args)
                save_log("pipeline_logs.txt", "training session " + str(i*3 +1) + " : " + str(args) + " " + train_log + "\n")

                args.model_name, args.size, args.cropped, args.device, args.labels = model, "224", "True", 0, label
                train_log = train(args)
                save_log("pipeline_logs.txt", "training session " + str(i*3 +2) + " : " + str(args) + " " + train_log + "\n")
