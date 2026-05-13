# Data

The image datasets used in this project are too large to include directly in the repository. Follow the instructions below to download them.

## Datasets

| Dataset   | Images | Captions/Image | Role               | Source                                                                 |
|-----------|--------|----------------|---------------------|------------------------------------------------------------------------|
| Flickr8k  | 8,000  | 5              | Development / Debug | [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)         |
| Flickr30k | 31,000 | 5              | Primary Evaluation  | [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr30k)        |

## Download Instructions

### Option 1: Kaggle CLI (used in our notebooks)

1. Create a [Kaggle](https://www.kaggle.com/) account if you don't have one.
2. Go to **Account → API → Create New Token** to download your `kaggle.json`.
3. Place `kaggle.json` in `~/.kaggle/` and set permissions:
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
4. Download the dataset:
   ```bash
   # Flickr8k (for quick development/debugging)
   kaggle datasets download -d adityajn105/flickr8k -p data/ --unzip

   # Flickr30k (primary benchmark)
   kaggle datasets download -d adityajn105/flickr30k -p data/ --unzip
   ```

### Option 2: Manual Download

1. Visit the Kaggle dataset pages linked above.
2. Click **Download** and extract the zip into this `data/` directory.

## Expected Directory Structure

After downloading, the `data/` directory should look like:

```
data/
├── README.md
├── Images/              # All .jpg image files
└── captions.txt         # CSV with columns: image, caption
```

For Flickr30k, the caption file may be named `results.csv` instead of `captions.txt`. The notebook handles both formats automatically.

## Preprocessing Notes

- **Vocabulary:** A fixed vocabulary of 10,000 words is built from training captions at runtime. Words outside this vocabulary are mapped to `<unk>`.
- **Splits:** We use a random 85/10/5 train/val/test split. The split is seeded (`SEED = 42` in the main notebook) for reproducibility.
- **Image preprocessing:** Images are resized so the shorter side is 256 pixels, then center-cropped to 224×224 to match VGG-19 input requirements.
- **VGG-19 features:** The encoder extracts 14×14×512 feature maps from the `conv4` layer (before max-pooling) of a pretrained VGG-19. These are extracted on-the-fly during training (no separate pre-extraction step required, though caching to disk is recommended for Flickr30k to speed up training).

## Pretrained Encoder Weights

VGG-19 weights are downloaded automatically via `torchvision.models.vgg19(pretrained=True)` — no manual download is needed.
