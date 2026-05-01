# import argparse
# import concurrent.futures
# import os
# import sys
# import warnings
# from pathlib import Path
# from urllib.parse import unquote, urlparse

# sys.path.insert(0, os.getcwd())


# import requests
# from bs4 import BeautifulSoup
# from tqdm import tqdm

# # Suppress SSL verification warnings
# warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# from _configs.constants import DATA_DIR


# def get_filename_from_url(url: str) -> str:
#     return unquote(os.path.basename(urlparse(url).path))


# def get_file_links(base_url: str) -> list[str]:
#     print(f"Fetching links from: {base_url}")
#     try:
#         response = requests.get(base_url, verify=False, timeout=10)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.text, "html.parser")

#         file_links = []
#         for link in soup.find_all("a"):
#             href = link.get("href")
#             if href and any(href.endswith(ext) for ext in [".tar", ".json", ".feather"]):
#                 file_links.append(
#                     base_url.rstrip("/") + ("/" if not href.startswith("/") else "") + href
#                 )

#         print(f"Found {len(file_links)} files to download")
#         return file_links
#     except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
#         print(f"Error fetching links from {base_url}: {e}")
#         return []


# def download_file(file_url: str, save_path: Path) -> bool:
#     save_path.parent.mkdir(parents=True, exist_ok=True)

#     try:
#         # Check if file exists
#         response = requests.head(file_url, verify=False, timeout=10)
#         if response.status_code != 200:
#             raise ValueError(f"File not found (status code: {response.status_code})")

#         # Get file size if available
#         file_size = int(response.headers.get("content-length", 0))

#         # Download with tqdm progress bar
#         with requests.get(file_url, stream=True, verify=False, timeout=30) as response:
#             response.raise_for_status()
#             with open(save_path, "wb") as file:
#                 with tqdm(
#                     total=file_size,
#                     unit="B",
#                     unit_scale=True,
#                     desc=os.path.basename(save_path),
#                     leave=False,
#                 ) as pbar:
#                     for chunk in response.iter_content(chunk_size=8192):
#                         if chunk:
#                             file.write(chunk)
#                             pbar.update(len(chunk))

#         return True
#     except requests.exceptions.SSLError:
#         print(f"SSL Error for {os.path.basename(save_path)}, retrying without verification...")
#         with requests.get(file_url, stream=True, verify=False, timeout=30) as response:
#             response.raise_for_status()
#             with open(save_path, "wb") as file:
#                 with tqdm(
#                     total=file_size,
#                     unit="B",
#                     unit_scale=True,
#                     desc=os.path.basename(save_path),
#                     leave=False,
#                 ) as pbar:
#                     for chunk in response.iter_content(chunk_size=8192):
#                         if chunk:
#                             file.write(chunk)
#                             pbar.update(len(chunk))
#         return True
#     except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
#         print(f"Download failed for {os.path.basename(save_path)}: {e}")
#         return False


# def download_files_parallel(
#     file_urls: list[str], save_paths: list[Path], max_workers: int
# ) -> None:
#     if not file_urls:
#         return

#     print(f"Starting parallel download of {len(file_urls)} files with {max_workers} workers")

#     # Use a thread pool with a progress bar for overall progress
#     with tqdm(total=len(file_urls), desc="Overall progress", unit="file") as pbar:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#             # Submit all tasks
#             future_to_url = {
#                 executor.submit(download_file, url, path): (url, path)
#                 for url, path in zip(file_urls, save_paths, strict=True)
#             }

#             # Process completed tasks
#             for future in concurrent.futures.as_completed(future_to_url):
#                 url, path = future_to_url[future]
#                 try:
#                     success = future.result()
#                     if success:
#                         pbar.update(1)
#                 except Exception as e:
#                     print(f"Error processing {url}: {e}")


# def download_webdatasets(dataset_name: str, base_url: str, max_workers: int) -> None:
#     url = f"{base_url}/webdatasets/{dataset_name}_1M"
#     file_links = get_file_links(url)

#     if not file_links:
#         return

#     save_paths = [
#         Path(DATA_DIR) / dataset_name / "train" / get_filename_from_url(file_link)
#         for file_link in file_links
#     ]
#     download_files_parallel(file_links, save_paths, max_workers)


# def download_data_object_models(dataset_name: str, base_url: str) -> None:
#     dataset_urls = {
#         "gso": f"{base_url}/tars/google_scanned_objects.zip",
#         "shapenet": f"{base_url}/tars/shapenetcorev2.zip",
#     }

#     if dataset_name not in dataset_urls:
#         print(f"Warning: Unsupported dataset '{dataset_name}' for data_object_models. Skipping.")
#         return

#     url = dataset_urls[dataset_name]
#     save_path = Path(DATA_DIR) / dataset_name / get_filename_from_url(url)
#     download_file(url, save_path)


# def main() -> None:
#     parser = argparse.ArgumentParser(description="Download training datasets")
#     parser.add_argument("--webdatasets", action="store_true", help="Download webdatasets")
#     parser.add_argument(
#         "--data_object_models", action="store_true", help="Download data object models"
#     )
#     parser.add_argument("--dataset_name", type=str, help="Specific dataset to download")
#     parser.add_argument(
#         "--workers", type=int, default=16, help="Number of parallel download workers"
#     )
#     args = parser.parse_args()

#     # Exit early if no download options are specified
#     if not args.webdatasets and not args.data_object_models:
#         print("Error: No download options specified. Use --webdatasets or --data_object_models")
#         parser.print_help()
#         sys.exit(1)

#     # # Download selected datasets
#     # if args.dataset_name in TRAIN_DATASETS:
#     #     if args.webdatasets:
#     #         print(f"\nDownloading webdatasets for {args.dataset_name}")
#     #         download_webdatasets(args.dataset_name, TRAIN_DATA_BASE_URL, args.workers)

#     #     if args.data_object_models:
#     #         print(f"\nDownloading data object models for {args.dataset_name}")
#     #         download_data_object_models(args.dataset_name, TRAIN_DATA_BASE_URL)
#     else:
#         print(f"Warning: Unsupported dataset '{args.dataset_name}' for training. Skipping.")


# if __name__ == "__main__":
#     main()
