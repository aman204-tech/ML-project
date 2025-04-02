from src.preprocessing.video import VideoPreprocessor

def main():
    # Instantiate the preprocessor (you can provide cascade paths if needed)
    preprocessor = VideoPreprocessor()

    # Process the dataset
    # Change these paths if your raw dataset is somewhere else.
    input_dir = "Dataset/Raw"
    output_dir = "Dataset/Processed"
    preprocessor.process_dataset(input_dir, output_dir)

if __name__ == "__main__":
    main()