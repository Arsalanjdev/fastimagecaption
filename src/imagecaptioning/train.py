import torch
import torch.nn as nn

from src.imagecaptioning.model import HybridModelAttention
from torch.utils.data import DataLoader, Dataset, random_split
from src.imagecaptioning.token_utils import split_imagefile_captions, tokenize
from PIL import Image

from src.imagecaptioning.vocab import Vocabulary
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence


CAPTION_PATH = "captions.txt"
IMAGE_DIR = "Images"
BATCH_SIZE = 32
VAL_RATIO = 0.1
TEST_RATIO = 0.1
EPOCHS = 50
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


class ImageCaptionDataset(Dataset):
    """
    PyTorch Dataset for loading image–caption pairs for image captioning
    tasks.

    Each dataset item returns a transformed image tensor and a tensor of
    caption token indices, including <SOS> (start-of-sequence) and
    <EOS> (end-of-sequence) tokens. Captions are tokenized using the
    `tokenize` function and mapped to indices using a provided
    `Vocabulary` object.

    Args:
        captions_filepath (str): Path to the captions file containing
            image filenames and captions.
        image_directory (str): Directory where image files are stored.
        vocabulary (Vocabulary): Vocabulary instance used to convert
            tokens to indices.
        transforms (transforms.Compose | None): Optional torchvision
            transformations applied to each image.
    """

    def __init__(
        self,
        captions_filepath: str,
        image_directory: str,
        vocabulary: Vocabulary,
        transforms: transforms.Compose | None = None,
    ):
        """
        Initialize the dataset by loading image–caption pairs and storing
        preprocessing configuration.
        """
        self.imagecaptionpairs = split_imagefile_captions(captions_filepath)
        self.vocab = vocabulary
        self.image_directory = image_directory
        self.transforms = transforms

    def __len__(self):
        """
        Return the total number of image–caption pairs in the dataset.

        Returns:
            int: Dataset size.
        """
        return len(self.imagecaptionpairs)

    def __getitem__(self, idx):
        """
        Retrieve a single image–caption pair by index.

        The image is loaded, converted to RGB, optionally transformed,
        and the caption is tokenized and converted to a tensor of token
        indices with <SOS> and <EOS> markers.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Tensor, Tensor]: Transformed image tensor and caption
            index tensor (dtype=torch.long).
        """
        imagefile, caption = self.imagecaptionpairs[idx]
        img = Image.open(f"{self.image_directory}/{imagefile}").convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        tokens = tokenize(caption)
        caption_indices = (
            [self.vocab.word2idx["<SOS>"]]
            + [self.vocab.word2idx.get(t, self.vocab.word2idx["<UNK>"]) for t in tokens]
            + [self.vocab.word2idx["<EOS>"]]
        )
        caption_indices = torch.tensor(caption_indices, dtype=torch.long)
        return img, caption_indices


class CaptionCollate:
    """
    Custom collate function for batching image–caption samples with
    variable-length captions.

    This collate class is intended to be used with a PyTorch DataLoader.
    It stacks image tensors into a batch and pads caption sequences to
    the maximum caption length in the batch using a specified padding
    index.

    Args:
        pad_idx (int): Vocabulary index used for padding caption
            sequences.
    """

    def __init__(self, pad_idx):
        """
        Initialize the collate object with the padding token index.

        Args:
            pad_idx (int): Padding token index used when padding caption
                sequences.
        """
        self.pad_idx = pad_idx

    def __call__(self, batch):
        """
        Collate a batch of image–caption pairs into batched tensors.

        Images are stacked along the batch dimension, while captions are
        padded to equal length using `torch.nn.utils.rnn.pad_sequence`
        with the specified padding index.

        Args:
            batch (List[Tuple[Tensor, Tensor]]): A list of dataset samples,
                where each sample contains an image tensor and a caption
                index tensor.

        Returns:
            Tuple[Tensor, Tensor]:
                - images (Tensor): Batched image tensor of shape
                  (batch_size, C, H, W).
                - captions (Tensor): Padded caption tensor of shape
                  (batch_size, max_caption_length).
        """
        images = torch.stack([item[0] for item in batch], dim=0)
        captions = pad_sequence(
            [item[1] for item in batch], batch_first=True, padding_value=self.pad_idx
        )
        return images, captions


all_captions = [caption[1] for caption in split_imagefile_captions(CAPTION_PATH)]
vocab = Vocabulary(all_captions)
vocab.dump_to_json("vocab.json")
pad_idx = vocab.word2idx["<PAD>"]

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

dataset = ImageCaptionDataset(CAPTION_PATH, IMAGE_DIR, vocab, transforms=transform)

# Split dataset
total_len = len(dataset)
test_len = int(TEST_RATIO * total_len)
val_len = int(VAL_RATIO * total_len)
train_len = total_len - val_len - test_len
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_len, val_len, test_len]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=CaptionCollate(pad_idx),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=CaptionCollate(pad_idx),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=CaptionCollate(pad_idx),
)


model = HybridModelAttention(len(vocab)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


def train_one_epoch(model, loader):
    model.train()
    total_loss = 0

    for images, captions in loader:
        images, captions = images.to(DEVICE), captions.to(DEVICE)
        inputs, targets = captions[:, :-1], captions[:, 1:]

        optimizer.zero_grad()

        outputs = model(images, inputs)
        vocab_size = outputs.size(-1)

        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, captions in loader:
            images, captions = images.to(DEVICE), captions.to(DEVICE)
            inputs, targets = captions[:, :-1], captions[:, 1:]

            outputs = model(images, inputs)  # ← same change here
            vocab_size = outputs.size(-1)

            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

            total_loss += loss.item()

    return total_loss / len(loader)


for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)
    print(
        f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )

test_loss = evaluate(model, test_loader)
print(f"Test Loss: {test_loss:.4f}")
