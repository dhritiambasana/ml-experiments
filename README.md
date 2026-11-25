# Machine Learning Experiment Codes

## Setup
Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

## Notes
- The examples use built-in or generated datasets, so most do not need a separate dataset file.
- The CNN script downloads MNIST on its first run and requires TensorFlow.
- Kernel-based image processing creates a sample image internally. Replace that section with `cv2.imread("your_image.jpg", cv2.IMREAD_GRAYSCALE)` if you need to process your own image.
- Data Analysis uses the built-in Iris dataset.
