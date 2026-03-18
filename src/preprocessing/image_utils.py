"""
Image Validation & Transformation Utilities
=============================================

Reusable image processing functions used across the pipeline.

CONCEPT: Defensive Image Processing
--------------------------------------
Real-world images are messy:
  - Corrupted files that PIL can't open
  - Wrong formats (.pdf renamed to .jpg)
  - EXIF rotation (phone photos appear rotated)
  - Extremely large images that exhaust memory
  - CMYK color space (from print workflows)
  - Animated GIFs (multiple frames)

Every image function should handle these gracefully.
"""

from PIL import Image, ExifTags, ImageFile
import io
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Allow loading of truncated images (common with network downloads)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Maximum image dimensions (prevent memory bombs)
MAX_IMAGE_PIXELS = 50_000_000  # ~7000x7000 pixels
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

VALID_FORMATS = {'JPEG', 'PNG', 'WEBP', 'BMP', 'TIFF'}
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}


def validate_image(image_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate that bytes represent a valid image.
    
    Returns:
        Tuple of (is_valid, error_message)
    
    CONCEPT: Image Validation Layers
    ----------------------------------
    We check multiple things in order:
    1. Can PIL open it? (basic format validity)
    2. Is it a supported format? (not SVG, PDF, etc.)
    3. Is it a reasonable size? (not 1x1 or 10000x10000)
    4. Can we actually read the pixel data? (not just headers)
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Check format
        if image.format not in VALID_FORMATS:
            return False, f"Unsupported format: {image.format}. Use JPEG, PNG, or WEBP."
        
        # Check size
        width, height = image.size
        if width < 10 or height < 10:
            return False, f"Image too small: {width}x{height}. Minimum 10x10."
        if width * height > MAX_IMAGE_PIXELS:
            return False, f"Image too large: {width}x{height}. Maximum ~7000x7000."
        
        # Try to load pixel data (catches truncated/corrupted images)
        image.load()
        
        return True, "Valid"
    
    except Exception as e:
        return False, f"Invalid image: {str(e)}"


def fix_orientation(image: Image.Image) -> Image.Image:
    """
    Fix image orientation based on EXIF data.
    
    CONCEPT: EXIF Orientation
    --------------------------
    Phone cameras store the physical orientation in EXIF metadata
    instead of rotating the actual pixels. This means:
    
    - The image file stores pixels in landscape orientation
    - EXIF tag says "this was taken in portrait mode"
    - Image viewers rotate automatically, but PIL doesn't!
    
    If we don't fix this, portrait photos appear sideways to our model.
    This is a COMMON production bug — easy to miss in testing because
    most test images don't have EXIF rotation.
    """
    try:
        exif = image._getexif()
        if exif is None:
            return image
        
        # Find the orientation tag
        orientation_key = None
        for key, val in ExifTags.TAGS.items():
            if val == 'Orientation':
                orientation_key = key
                break
        
        if orientation_key is None or orientation_key not in exif:
            return image
        
        orientation = exif[orientation_key]
        
        # Apply rotation based on EXIF orientation value
        rotations = {
            3: 180,
            6: 270,
            8: 90,
        }
        
        if orientation in rotations:
            image = image.rotate(rotations[orientation], expand=True)
        
        # Handle mirroring (orientation 2, 4, 5, 7)
        if orientation in (2, 5):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation in (4, 7):
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        return image
    
    except (AttributeError, KeyError, IndexError):
        return image


def preprocess_for_inference(
    image_bytes: bytes,
    target_size: Tuple[int, int] = (224, 224),
    quality: int = 95
) -> bytes:
    """
    Full preprocessing pipeline for inference.
    
    1. Open and validate
    2. Fix EXIF orientation
    3. Convert to RGB
    4. Resize to target size
    5. Return as JPEG bytes
    
    Args:
        image_bytes: Raw image bytes
        target_size: Output dimensions (width, height)
        quality: JPEG quality (1-100)
    
    Returns:
        Processed JPEG bytes
    """
    # Open
    image = Image.open(io.BytesIO(image_bytes))
    
    # Fix orientation
    image = fix_orientation(image)
    
    # Convert to RGB (handles RGBA, grayscale, CMYK, palette mode, etc.)
    if image.mode != 'RGB':
        # Handle RGBA by compositing on white background
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert('RGB')
    
    # Resize with high-quality resampling
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to JPEG bytes
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality, optimize=True)
    
    return buffer.getvalue()


def get_image_info(image_bytes: bytes) -> dict:
    """
    Extract metadata from image bytes.
    
    Useful for logging, debugging, and data quality monitoring.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return {
            'format': image.format,
            'mode': image.mode,
            'width': image.size[0],
            'height': image.size[1],
            'size_bytes': len(image_bytes),
            'size_kb': round(len(image_bytes) / 1024, 1),
            'has_exif': hasattr(image, '_getexif') and image._getexif() is not None,
        }
    except Exception as e:
        return {'error': str(e)}
