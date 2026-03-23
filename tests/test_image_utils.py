import io
from PIL import Image
from src.utils.image_utils import is_mostly_black, is_too_black, is_too_white


def test_is_mostly_black_false(tmp_path):
    img_path = tmp_path / "glioma" / "original" / "img.jpg"

    img_path.parent.mkdir(parents=True)

    white_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    white_img.save(img_path)

    result = is_mostly_black(img_path)

    assert result is False


def test_is_mostly_black_true(tmp_path):
    img_path = tmp_path / "glioma" / "original" / "img.jpg"

    img_path.parent.mkdir(parents=True)

    white_img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    white_img.save(img_path)

    result = is_mostly_black(img_path)

    assert result is True


def test_is_mostly_black_when_image_missing(tmp_path):
    img_path = tmp_path / "glioma" / "original" / "img.jpg"

    result = is_mostly_black(img_path)

    assert result is True


def create_image(color=(0, 0, 0), size=(100, 100)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_is_too_black():
    black_image_bytes = create_image((0, 0, 0))
    assert is_too_black(black_image_bytes)


def test_is_too_white():
    white_image_bytes = create_image((255, 255, 255))
    assert is_too_white(white_image_bytes)
