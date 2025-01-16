from recognizer import download, unzip
import numpy as np

link = "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"

download(link)

zip_path = "./traffic-signs-data.zip"
extract_path = "./data/"

unzip(zip_path, extract_path)
