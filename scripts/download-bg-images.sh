#! /bin/bash
mkdir -p ~/.EasyOCR/bg-images

for ((i=0 ; i <= 1000; i++)); do
  wget -q https://source.unsplash.com/random -O ~/.EasyOCR/bg-images/$i.jpg
  echo "Downloaded $i.jpg"
done | tqdm --total 1001
