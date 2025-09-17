import os
import requests
from tqdm import tqdm
import zipfile

""" 
Download and unzip Prostatectomy dataset 
from https://doi.org/10.5522/04/17796640.v1
"""

FILES = {
    "SAR-RARP50_train_set.zip": "https://rdr.ucl.ac.uk/ndownloader/articles/24932529/versions/1",
    "SAR-RARP50_test_set.zip": "https://rdr.ucl.ac.uk/ndownloader/articles/24932499/versions/1",
}
    
def download_file(url, save_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))

        with open(save_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=os.path.basename(save_path),
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    print(f"Saved to {save_path}")
    
    
def unzip_file(zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unzipped {os.path.basename(zip_path)} to {extract_to}")
    os.remove(zip_path)
    
def unzip_recursive(save_path, folder_path):
    unzip_file(save_path, folder_path)
        
    for root, dirs, files in tqdm(os.walk(folder_path)):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                extract_folder = os.path.join(root, file.replace(".zip", ""))
                unzip_file(zip_path, extract_folder)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for filename, url in FILES.items():
        save_path  = os.path.join(script_dir, filename)
        print(f"Starting download: {filename}")
        download_file(url, save_path )
        
        # unzip recursively videos inside the extracted folder
        dest_folder = os.path.join(script_dir, filename.replace('.zip', ''))
        unzip_recursive(dest_folder)

if __name__ == "__main__":
    main()