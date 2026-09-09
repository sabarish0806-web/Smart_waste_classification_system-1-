import os

def allowed_file(filename, allowed_extensions={'jpg', 'jpeg', 'png'}):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

WASTE_CATEGORIES = {
    "Plastic": {
        "display_name": "Plastic Waste",
        "color": "#EAB308",  # Amber/Yellow
        "bin_type": "Dry Recycling Bin",
        "bin_color": "Yellow",
        "icon": "fa-bottle-water",
        "guidance": "Rinse containers to remove food residue. Flatten plastic bottles and replace caps before placing them in the yellow dry recycling bin. Avoid shredding.",
        "examples": ["Water bottles", "Food packaging containers", "Shampoo bottles", "Plastic jugs"]
    },
    "Paper": {
        "display_name": "Paper & Cardboard",
        "color": "#3B82F6",  # Blue
        "bin_type": "Paper Recycling Bin",
        "bin_color": "Blue",
        "icon": "fa-newspaper",
        "guidance": "Ensure paper and cardboard are clean and dry. Flatten boxes to save space. Do not recycle wet, grease-stained, or wax-coated paper.",
        "examples": ["Newspapers", "Cardboard boxes", "Office paper", "Magazines", "Cartons"]
    },
    "Metal": {
        "display_name": "Metal Waste",
        "color": "#6B7280",  # Grey
        "bin_type": "Metal Recycling Bin",
        "bin_color": "Grey",
        "icon": "fa-can-food",
        "guidance": "Empty and rinse metal cans and foil. Crushing cans is optional but helps save space. Metal aerosol cans should be completely empty.",
        "examples": ["Aluminum soda cans", "Tin food cans", "Clean aluminum foil", "Metal jar lids"]
    },
    "Glass": {
        "display_name": "Glass Container",
        "color": "#10B981",  # Green
        "bin_type": "Glass Recycling Bin",
        "bin_color": "Green",
        "icon": "fa-wine-bottle",
        "guidance": "Rinse glass bottles and jars. Metal/plastic caps can be removed and recycled separately. Do not include ceramics, cookware, or light bulbs.",
        "examples": ["Glass bottles", "Food jars", "Beverage containers"]
    },
    "Organic": {
        "display_name": "Organic / Bio-Waste",
        "color": "#8B5CF6",  # Purple/Brown
        "bin_type": "Organic Waste Bin / Compost",
        "bin_color": "Green/Brown",
        "icon": "fa-seedling",
        "guidance": "Dispose of in designated organic waste bins or home compost. Avoid mixing with plastic wrappers or non-biodegradable bags.",
        "examples": ["Food waste", "Fruit peels", "Coffee grounds", "Garden clippings", "Eggshells"]
    },
    "Other/Unknown": {
        "display_name": "Other / General Waste",
        "color": "#EF4444",  # Red
        "bin_type": "General Waste Bin",
        "bin_color": "Black/Red",
        "icon": "fa-trash",
        "guidance": "Item could not be confidently identified into a primary recyclable category. Check local municipal guidelines or dispose in general landfill waste.",
        "examples": ["Composite materials", "Hazardous items", "E-waste", "Mixed materials"]
    }
}

def get_category_guidance(category_name):
    return WASTE_CATEGORIES.get(category_name, WASTE_CATEGORIES["Other/Unknown"])
