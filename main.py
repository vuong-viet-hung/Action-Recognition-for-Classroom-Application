from annotation import get_class_indices, get_train_samples, get_test_samples

# Annotation files
CLASS_INDEX_FILE = (
    "data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt"
)
TRAIN_SAMPLE_FILES = [
    "data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist01.txt",
    "data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist02.txt",
    "data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist03.txt",
]
VALID_SAMPLE_FILES = [
    "data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist01.txt",
    "data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist02.txt",
]
TEST_SAMPLE_FILES = [
    "data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist03.txt",
]


def main() -> None:
    class_indices = get_class_indices(CLASS_INDEX_FILE)
    selected_classes = {"BalanceBeam", "BaseballPitch"}

    train_sample_generator = get_train_samples(TRAIN_SAMPLE_FILES, selected_classes)
    valid_sample_generator = get_test_samples(
        VALID_SAMPLE_FILES, class_indices, selected_classes
    )
    test_sample_generator = get_test_samples(
        TEST_SAMPLE_FILES, class_indices, selected_classes
    )


if __name__ == "__main__":
    main()
