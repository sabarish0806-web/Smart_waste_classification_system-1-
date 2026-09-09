import os
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

CLASSES = ["Glass", "Metal", "Organic", "Paper", "Plastic"]
BASE_DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")

def create_paper_image(width=224, height=224):
    """Generate realistic paper / cardboard image with fibrous texture, folds, and tan/white paper hues."""
    paper_type = random.choice(["cardboard", "newspaper", "kraft_paper", "carton"])
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    if paper_type == "cardboard":
        # Brown / tan cardboard shade
        base_color = (random.randint(180, 205), random.randint(135, 160), random.randint(85, 110))
        img.paste(base_color, [0, 0, width, height])
        # Add distinct cardboard fluting / lines
        for y in range(0, height, random.randint(5, 8)):
            draw.line([(0, y), (width, y)], fill=(base_color[0]-30, base_color[1]-25, base_color[2]-20), width=2)
            draw.line([(0, y+2), (width, y+2)], fill=(base_color[0]+15, base_color[1]+15, base_color[2]+10), width=1)
    elif paper_type == "newspaper":
        # Off-white with printed text lines
        img.paste((235, 230, 215), [0, 0, width, height])
        for y in range(12, height - 12, random.randint(6, 12)):
            x_end = random.randint(width//2, width - 15)
            draw.line([(15, y), (x_end, y)], fill=(35, 35, 35), width=random.randint(1, 2))
    elif paper_type == "carton":
        # Printed paper carton packaging
        img.paste((215, 190, 160), [0, 0, width, height])
        draw.rectangle([15, 15, width-15, height-15], outline=(140, 70, 40), width=4)
        draw.text((30, 30), "CARDBOARD BOX", fill=(100, 40, 20))
    else:
        # Kraft paper / brown paper bag
        img.paste((195, 155, 105), [0, 0, width, height])
        for _ in range(10):
            y = random.randint(0, height)
            draw.line([(0, y), (width, y)], fill=(170, 130, 85), width=2)

    # Add paper fiber grain noise
    noise = np.random.randint(-12, 12, (height, width, 3), dtype=np.int16)
    img_arr = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_arr)

def create_plastic_image(width=224, height=224):
    """Generate realistic plastic image (clear bottles, blue water bottles, colorful packaging)."""
    plastic_type = random.choice(["clear_bottle", "blue_bottle", "plastic_wrapper", "plastic_jug"])
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    if plastic_type == "clear_bottle":
        # Cyan/clear transparent highlights with ridges
        img.paste((210, 230, 245), [0, 0, width, height])
        # Bottle cylinder shape outline
        draw.rectangle([50, 20, 174, 204], outline=(140, 180, 220), width=6)
        # Specular highlight vertical stripes
        draw.rectangle([70, 30, 90, 190], fill=(255, 255, 255))
        draw.rectangle([130, 30, 140, 190], fill=(240, 250, 255))
        # Cap
        draw.rectangle([85, 5, 139, 20], fill=(30, 120, 220))
    elif plastic_type == "blue_bottle":
        img.paste((30, 120, 210), [0, 0, width, height])
        draw.ellipse([30, 30, 194, 194], outline=(200, 235, 255), width=8)
        draw.rectangle([80, 50, 120, 180], fill=(230, 245, 255))
    elif plastic_type == "plastic_wrapper":
        img.paste((240, 60, 120), [0, 0, width, height])
        # Crinkle highlights
        for _ in range(12):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 255), width=random.randint(2, 5))
    else:
        # Plastic jug
        img.paste((240, 240, 240), [0, 0, width, height])
        draw.rectangle([40, 30, 184, 190], fill=(220, 235, 245), outline=(100, 150, 200), width=5)
        draw.rectangle([60, 10, 110, 30], fill=(230, 50, 50))

    return img

def create_metal_image(width=224, height=224):
    """Generate metal waste image (soda cans, tin cans, aluminum foil)."""
    metal_type = random.choice(["can", "foil", "tin"])
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    if metal_type == "can":
        # Metallic silver/red soda can
        img.paste((180, 185, 190), [0, 0, width, height])
        draw.rectangle([40, 20, 184, 204], fill=(200, 30, 30), outline=(220, 225, 230), width=6)
        # Specular glare down center
        draw.rectangle([95, 20, 125, 204], fill=(240, 245, 250))
        # Top rim
        draw.ellipse([40, 10, 184, 30], fill=(210, 215, 220))
    elif metal_type == "foil":
        # Crinkled silver foil
        img.paste((190, 195, 200), [0, 0, width, height])
        for _ in range(25):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            draw.line([(x1, y1), (x2, y2)], fill=(250, 250, 255), width=random.randint(1, 4))
            draw.line([(x1, y1), (x2, y2)], fill=(100, 105, 110), width=1)
    else:
        # Tin can with ridges
        img.paste((160, 165, 170), [0, 0, width, height])
        for y in range(30, 190, 15):
            draw.line([(30, y), (194, y)], fill=(230, 235, 240), width=3)
            draw.line([(30, y+4), (194, y+4)], fill=(100, 105, 110), width=2)

    return img

def create_glass_image(width=224, height=224):
    """Generate glass container image (green bottle, brown bottle, glass jar)."""
    glass_type = random.choice(["green_glass", "brown_glass", "clear_jar"])
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    if glass_type == "green_glass":
        img.paste((20, 120, 60), [0, 0, width, height])
        draw.rectangle([60, 15, 164, 209], outline=(100, 220, 140), width=7)
        draw.rectangle([80, 25, 95, 195], fill=(180, 255, 200))
    elif glass_type == "brown_glass":
        img.paste((130, 70, 20), [0, 0, width, height])
        draw.rectangle([60, 15, 164, 209], outline=(210, 140, 60), width=7)
        draw.rectangle([80, 25, 95, 195], fill=(255, 200, 120))
    else:
        # Glass jar
        img.paste((220, 240, 235), [0, 0, width, height])
        draw.rectangle([35, 40, 189, 195], outline=(120, 180, 170), width=8)
        draw.rectangle([55, 55, 75, 180], fill=(255, 255, 255))
        draw.rectangle([45, 20, 179, 40], fill=(160, 160, 160))

    return img

def create_organic_image(width=224, height=224):
    """Generate organic waste image (food waste, fruit peels, leaves, coffee grounds)."""
    organic_type = random.choice(["leaf_food", "banana_peel", "apple_waste", "coffee_soil"])
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    if organic_type == "leaf_food":
        img.paste((35, 130, 40), [0, 0, width, height])
        for _ in range(25):
            x = random.randint(10, width-10)
            y = random.randint(10, height-10)
            r = random.randint(15, 35)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(random.randint(50, 140), random.randint(140, 210), 25))
    elif organic_type == "banana_peel":
        img.paste((45, 125, 35), [0, 0, width, height])
        draw.polygon([(20, 30), (200, 70), (170, 190), (50, 170)], fill=(210, 180, 30))
        draw.polygon([(40, 50), (180, 85), (150, 175), (70, 155)], fill=(150, 110, 20))
        draw.line([(20, 30), (200, 70)], fill=(70, 45, 10), width=6)
    elif organic_type == "apple_waste":
        img.paste((40, 120, 35), [0, 0, width, height])
        draw.ellipse([30, 30, 194, 194], fill=(220, 200, 130), outline=(170, 40, 30), width=8)
        draw.ellipse([80, 80, 144, 144], fill=(90, 50, 20))
    else:
        # Coffee grounds / soil
        img.paste((70, 45, 25), [0, 0, width, height])
        for _ in range(60):
            x = random.randint(0, width)
            y = random.randint(0, height)
            draw.ellipse([x-2, y-2, x+2, y+2], fill=(random.randint(40, 100), random.randint(120, 180), 30))

    return img

def generate_dataset(num_train_per_class=120, num_val_per_class=30, num_test_per_class=30):
    """Populates train, val, and test splits for all 5 waste classes."""
    random.seed(42)
    np.random.seed(42)

    creator_map = {
        "Paper": create_paper_image,
        "Plastic": create_plastic_image,
        "Metal": create_metal_image,
        "Glass": create_glass_image,
        "Organic": create_organic_image
    }

    splits = {
        "train": num_train_per_class,
        "val": num_val_per_class,
        "test": num_test_per_class
    }

    total_count = 0
    for split_name, count in splits.items():
        for cls in CLASSES:
            split_dir = os.path.join(BASE_DATASET_DIR, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            creator = creator_map[cls]

            for i in range(count):
                img = creator(224, 224)
                img_path = os.path.join(split_dir, f"{cls.lower()}_{i+1:03d}.jpg")
                img.save(img_path, quality=95)
                total_count += 1

    print(f"Dataset generated successfully! Total images created: {total_count}")
    print(f"Location: {BASE_DATASET_DIR}")

if __name__ == "__main__":
    generate_dataset()
