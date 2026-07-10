#!/usr/bin/env python3
"""Import or generate assets/products.json for mini-recsys."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "products.json"

DATABLIST_CSV_URL = (
    "https://github.com/datablist/sample-csv-files/raw/main/files/products/products-10000.csv"
)
DATABLIST_DRIVE_CSV_URL = (
    "https://drive.google.com/uc?id=1BE-dfkrb6oyLKDuqXAq2fDYMkDz2f9hM&export=download"
)

# Map external dataset categories onto mini-recsys categories.
CATEGORY_MAP = {
    "electronics": "Electronics",
    "electronic": "Electronics",
    "computer": "Electronics",
    "computers": "Electronics",
    "phone": "Electronics",
    "phones": "Electronics",
    "audio": "Electronics",
    "camera": "Electronics",
    "gaming": "Electronics",
    "book": "Books",
    "books": "Books",
    "media": "Books",
    "stationery": "Books",
    "home": "Home",
    "kitchen": "Home",
    "furniture": "Home",
    "garden": "Home",
    "appliance": "Home",
    "appliances": "Home",
    "clothing": "Clothing",
    "clothes": "Clothing",
    "fashion": "Clothing",
    "apparel": "Clothing",
    "shoes": "Clothing",
    "sport": "Sports",
    "sports": "Sports",
    "outdoor": "Sports",
    "beauty": "Beauty",
    "health": "Beauty",
    "toy": "Toys",
    "toys": "Toys",
    "kids": "Toys",
    "baby": "Toys",
    "food": "Food",
    "grocery": "Food",
    "automotive": "Automotive",
    "auto": "Automotive",
}

TARGET_CATEGORIES = [
    "Electronics",
    "Books",
    "Home",
    "Clothing",
    "Sports",
    "Beauty",
    "Toys",
    "Food",
    "Automotive",
    "Misc",
]

IMAGE_BY_CATEGORY = {
    "Electronics": "photo-1518770660439-4636190af475",
    "Books": "photo-1544947950-fa07a98d237f",
    "Home": "photo-1556911220-bff31c812dba",
    "Clothing": "photo-1445205170230-053b83016050",
    "Sports": "photo-1461896836934-ffe607ba8131",
    "Beauty": "photo-1596462502278-27bfdd403f2c",
    "Toys": "photo-1558062435-c77e45c2e3db",
    "Food": "photo-1546069901-ba9599a7e63c",
    "Automotive": "photo-1492144534655-ae79c964c9d7",
    "Misc": "photo-1472851294608-062f824d29cc",
}

BRANDS = [
    "Apex",
    "Nova",
    "Zenith",
    "Orbit",
    "Pulse",
    "Vertex",
    "Lumen",
    "Craft",
    "Summit",
    "Harbor",
    "Nimbus",
    "Forge",
    "Atlas",
    "Echo",
    "Prism",
]

ADJECTIVES = [
    "Pro",
    "Lite",
    "Ultra",
    "Smart",
    "Classic",
    "Premium",
    "Compact",
    "Essential",
    "Deluxe",
    "Active",
]

PRODUCT_TYPES = {
    "Electronics": [
        "Wireless Earbuds",
        "Bluetooth Speaker",
        "USB-C Hub",
        "Mechanical Keyboard",
        "4K Monitor",
        "Action Camera",
        "Portable Charger",
        "Smart Watch",
        "Noise-Canceling Headphones",
        "Tablet Stand",
        "Webcam",
        "SSD Drive",
        "Router",
        "Graphics Tablet",
        "Drone",
    ],
    "Books": [
        "Programming Guide",
        "Science Fiction Novel",
        "History Collection",
        "Cookbook",
        "Self-Help Handbook",
        "Poetry Anthology",
        "Biography",
        "Design Patterns Manual",
        "Language Course",
        "Art Album",
        "Mystery Thriller",
        "Philosophy Reader",
        "Travel Journal",
        "Business Strategy Book",
        "Children's Storybook",
    ],
    "Home": [
        "Ceramic Mug Set",
        "Desk Lamp",
        "Throw Pillow",
        "Storage Basket",
        "Coffee Maker",
        "Air Purifier",
        "Bamboo Cutting Board",
        "Wall Clock",
        "Scented Candle",
        "Vacuum Cleaner",
        "Bed Sheet Set",
        "Kitchen Knife Set",
        "Plant Pot",
        "Shower Curtain",
        "Tool Organizer",
    ],
    "Clothing": [
        "Hoodie",
        "Running Shoes",
        "Denim Jacket",
        "Wool Sweater",
        "Cargo Pants",
        "Baseball Cap",
        "Leather Belt",
        "Summer Dress",
        "Windbreaker",
        "Socks Pack",
        "Canvas Sneakers",
        "Flannel Shirt",
        "Yoga Leggings",
        "Winter Gloves",
        "Polo Shirt",
    ],
    "Sports": [
        "Yoga Mat",
        "Dumbbell Set",
        "Tennis Racket",
        "Camping Tent",
        "Hiking Backpack",
        "Resistance Bands",
        "Cycling Helmet",
        "Swim Goggles",
        "Basketball",
        "Fitness Tracker Band",
        "Foam Roller",
        "Soccer Ball",
        "Water Bottle",
        "Jump Rope",
        "Golf Balls",
    ],
    "Beauty": [
        "Face Moisturizer",
        "Shampoo Set",
        "Lip Balm Pack",
        "Sunscreen Lotion",
        "Perfume",
        "Makeup Brush Kit",
        "Hair Dryer",
        "Body Wash",
        "Serum",
        "Hand Cream",
        "Beard Oil",
        "Nail Polish Set",
        "Face Mask Pack",
        "Eye Cream",
        "Cleansing Wipes",
    ],
    "Toys": [
        "Building Blocks Set",
        "Puzzle Game",
        "Plush Toy",
        "Board Game",
        "RC Car",
        "Art Supply Kit",
        "Educational Tablet",
        "Action Figure",
        "Card Game",
        "STEM Kit",
        "Doll House",
        "Train Set",
        "Craft Kit",
        "Outdoor Play Set",
        "Musical Toy",
    ],
    "Food": [
        "Organic Coffee Beans",
        "Green Tea Box",
        "Protein Bar Pack",
        "Olive Oil",
        "Granola Mix",
        "Dark Chocolate",
        "Instant Noodles",
        "Sparkling Water Case",
        "Pasta Sauce",
        "Honey Jar",
        "Snack Variety Pack",
        "Rice Bag",
        "Hot Sauce",
        "Nut Butter",
        "Dried Fruit Mix",
    ],
    "Automotive": [
        "Car Phone Mount",
        "Dash Camera",
        "Seat Cover Set",
        "Tire Pressure Gauge",
        "Jump Starter",
        "Floor Mats",
        "Air Freshener Pack",
        "Wiper Blades",
        "USB Car Charger",
        "Trunk Organizer",
        "LED Headlight Bulbs",
        "Cleaning Kit",
        "Roof Rack Straps",
        "Emergency Kit",
        "Polish Wax",
    ],
    "Misc": [
        "Gift Card Holder",
        "Notebook Set",
        "Desk Organizer",
        "Reusable Tote Bag",
        "Keychain",
        "Umbrella",
        "Phone Case",
        "Sticker Pack",
        "Calendar Planner",
        "Cable Organizer",
    ],
}


def normalize_category(raw: str) -> str:
    key = re.sub(r"[^a-z]", "", raw.lower())
    if key in CATEGORY_MAP:
        return CATEGORY_MAP[key]
    for token in re.split(r"[\s/&,-]+", raw.lower()):
        token = re.sub(r"[^a-z]", "", token)
        if token in CATEGORY_MAP:
            return CATEGORY_MAP[token]
    return "Misc"


def image_url(category: str, item_id: int) -> str:
    photo = IMAGE_BY_CATEGORY.get(category, IMAGE_BY_CATEGORY["Misc"])
    return f"https://images.unsplash.com/{photo}?w=400&h=300&fit=crop&sig={item_id}"


def download_csv(url: str, dest: Path) -> None:
    print(f"Downloading CSV from {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "mini-recsys-import/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        dest.write_bytes(response.read())


def load_csv_rows(csv_path: Path, limit: int) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break
        return rows


def rows_from_datablist(rows: list[dict[str, str]], limit: int) -> list[dict]:
    products = []
    for index, row in enumerate(rows[:limit], start=1):
        name = (row.get("Name") or row.get("name") or f"Product {index}").strip()
        category = normalize_category((row.get("Category") or row.get("category") or "Misc").strip())
        price_raw = row.get("Price") or row.get("price") or "0"
        try:
            price = round(float(str(price_raw).replace(",", "")), 2)
        except ValueError:
            price = round(random.uniform(5.0, 499.0), 2)
        products.append(
            {
                "id": index,
                "title": name,
                "category": category,
                "image_url": image_url(category, index),
                "price": max(price, 0.99),
            }
        )
    return products


def generate_products(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    per_category = count // len(TARGET_CATEGORIES)
    remainder = count % len(TARGET_CATEGORIES)
    products: list[dict] = []
    item_id = 1
    seen_titles: set[str] = set()

    for category_index, category in enumerate(TARGET_CATEGORIES):
        category_count = per_category + (1 if category_index < remainder else 0)
        types = PRODUCT_TYPES[category]
        for _ in range(category_count):
            for _attempt in range(20):
                title = (
                    f"{rng.choice(BRANDS)} {rng.choice(ADJECTIVES)} "
                    f"{rng.choice(types)} {rng.randint(100, 9999)}"
                )
                if title not in seen_titles:
                    seen_titles.add(title)
                    break
            products.append(
                {
                    "id": item_id,
                    "title": title,
                    "category": category,
                    "image_url": image_url(category, item_id),
                    "price": round(rng.uniform(4.99, 799.99), 2),
                }
            )
            item_id += 1

    rng.shuffle(products)
    for index, product in enumerate(products, start=1):
        product["id"] = index
    return products


def summarize(products: list[dict]) -> None:
    counts: dict[str, int] = {}
    for product in products:
        counts[product["category"]] = counts.get(product["category"], 0) + 1
    print(f"Generated {len(products)} products")
    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")


def write_products(products: list[dict]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(products, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
    print(f"Wrote {OUTPUT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10000, help="Number of products")
    parser.add_argument(
        "--source",
        choices=("generate", "datablist"),
        default="generate",
        help="generate = local synthetic data; datablist = download CSV",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generate mode")
    parser.add_argument(
        "--csv",
        type=Path,
        help="Use a local CSV file instead of downloading (datablist mode)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        print("count must be positive", file=sys.stderr)
        return 1

    if args.source == "generate":
        products = generate_products(args.count, args.seed)
    else:
        csv_path = args.csv
        if csv_path is None:
            csv_path = ROOT / "scripts" / ".cache" / "products-10000.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not csv_path.exists():
                try:
                    download_csv(DATABLIST_CSV_URL, csv_path)
                except Exception as error:
                    print(f"GitHub download failed: {error}")
                    print("Trying Google Drive mirror...")
                    download_csv(DATABLIST_DRIVE_CSV_URL, csv_path)
        rows = load_csv_rows(csv_path, args.count)
        if not rows:
            print("CSV is empty", file=sys.stderr)
            return 1
        products = rows_from_datablist(rows, args.count)

    summarize(products)
    write_products(products)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
