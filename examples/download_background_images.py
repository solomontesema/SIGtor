from icrawler import ImageDownloader
from icrawler.builtin import GoogleImageCrawler

def download_images(keywords, num_images=100, folder_path='images'):
    for keyword in keywords:
        print(f"Downloading images for keyword: {keyword}")
        crawler = GoogleImageCrawler(storage={'root_dir': f"{folder_path}/{keyword}"})
        crawler.crawl(keyword=keyword, max_num=num_images)

if __name__ == "__main__":
    keywords = [
        "sky",
        "countryside",
        "sunset",
        "beach",
        "mountains",
        "forest",
        "desert",
        "city skyline",
        "abstract background",
        "nature"
    ]
    download_images(keywords, num_images=100, folder_path="./Datasets/BackgroundImages/")