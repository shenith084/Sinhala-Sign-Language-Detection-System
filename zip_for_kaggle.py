import os
import zipfile

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            # Make sure we use relative paths in the zip, and force forward slashes for Kaggle (Linux)
            arcname = os.path.relpath(file_path, path).replace("\\", "/")
            ziph.write(file_path, arcname)

print("Creating Kaggle-compatible zip file...")
with zipfile.ZipFile('ssl400_full_kaggle_upload.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipdir('kaggle_staging', zipf)
print("Done! Safe to upload to Kaggle.")
