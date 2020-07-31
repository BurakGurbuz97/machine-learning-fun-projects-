import NST_net as Net
from PIL import Image

def reshape_img(image_path):
    img = Image.open(image_path)
    img = img.resize((400,300), Image.ANTIALIAS)
    img.save(image_path.split(".")[0] + ".png")
    return image_path.split(".")[0] + ".png"

CONTENT_IMG = "images/burek.jpg"
STYLE_IMG = "images/picasso_harem.jpg"
CONTENT_IMG = reshape_img(CONTENT_IMG)
STYLE_IMG = reshape_img(STYLE_IMG)

artist = Net.Artist()
artist.generate_image(CONTENT_IMG,STYLE_IMG,"burak_cubic", num_iterations=500)

