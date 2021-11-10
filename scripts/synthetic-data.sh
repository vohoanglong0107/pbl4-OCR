#! /bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset name>"
    exit 1
fi

train_dir=~/.EasyOCR/data/$1/train
valid_dir=~/.EasyOCR/data/$1/valid
bg_dir=~/.EasyOCR/bg-images
app_output_dir=/app/out/
app_bg_dir=/app/bg-images
image=belval/trdg:latest

# 1% normal black text on white background
docker run -v $train_dir:$app_output_dir -v $bg_dir:$app_bg_dir -t $image trdg \
-t 2 -f 64 -w 3 -r -b 1 -c 200

# 1% normal black text on white background
docker run -v $train_dir:$app_output_dir -v $bg_dir:$app_bg_dir -t $image trdg \
-t 2 -f 64 -w 3 -r -b 1 -fi -c 200

# White, gaussian noise, Quasicrystal background
for i in {0..2}; do
    # 1% normal text
    docker run -v $train_dir:$app_output_dir \
    -v $bg_dir:$app_bg_dir -t $image trdg \
    -t 2 -f 64 -w 3 -r -b $i -tc '#000000,#FFFFFF' -c 200
    # 1% random skewness to 5 degree
    docker run -v $train_dir:$app_output_dir \
    -v $bg_dir:$app_bg_dir -t $image trdg \
    -t 2 -f 64 -w 3 -r -b $i -tc '#000000,#FFFFFF' -k 5 -rk -c 200
    # 1% blur
    docker run -v $train_dir:$app_output_dir \
    -v $bg_dir:$app_bg_dir -t $image trdg \
    -t 2 -f 64 -w 3 -r -b $i -tc '#000000,#FFFFFF' -k 5 -rk -bl 2 -rbl -c 300

    # 9% distortion
    for j in {1..3}; do
        # distorsion orientation
        for z in {0..2}; do
            docker run -v $train_dir:$app_output_dir \
            -v $bg_dir:$app_bg_dir -t $image trdg \
            -t 2 -f 64 -w 3 -r -b $i -tc '#000000,#FFFFFF' -k 5 -rk -bl 2 -rbl -d $j -do $z -c 200
        done
    done
done

# image background
# random font
docker run -v $train_dir:$app_output_dir -v $bg_dir:$app_bg_dir -t $image trdg \
-t 2 -f 64 -w 3 -r -b 3 -id bg-images/ -tc '#000000,#FFFFFF' -c 500
# tight crop
docker run -v $train_dir:$app_output_dir -v $bg_dir:$app_bg_dir -t $image trdg \
-t 2 -f 64 -w 3 -r -b 3 -id bg-images/ -tc '#000000,#FFFFFF' -fi -c 500

# 5 degree skew
docker run -v $train_dir:$app_output_dir -v $bg_dir:$app_bg_dir -t $image trdg \
-t 2 -f 64 -w 3 -r -b 3 -id bg-images/ -tc '#000000,#FFFFFF' -k 5 -rk -c 1000

# blur
docker run -v $train_dir:$app_output_dir -v $bg_dir:$app_bg_dir -t $image trdg \
-t 2 -f 64 -w 3 -r -b 3 -id bg-images/ -tc '#000000,#FFFFFF' -k 5 -rk -bl 1 -rbl -c 1000


for j in {1..3}; do
    # distorsion orientation
    for z in {0..2}; do
        docker run -v $train_dir:$app_output_dir \
        -v $bg_dir:$app_bg_dir -t $image trdg \
        -t 2 -f 64 -w 3 -r -b 3 -id bg-images/ -tc '#000000,#FFFFFF' \
        -k 5 -rk -bl 1 -rbl -d $j -do $z -c 1500
    done
done

docker run -v $train_dir:$app_output_dir \
-v $bg_dir:$app_bg_dir -t $image trdg \
-t 2 -f 64 -w 3 -r -b 3 -id bg-images/ -tc '#000000,#FFFFFF' \
-k 5 -rk -bl 2 -rbl -d 3 -do 2 -c 500

# ------------------------------------------------------------
# validation

# 0.5% normal black text on white background
docker run -v $valid_dir:$app_output_dir -v $bg_dir:$app_bg_dir -t $image trdg \
-t 2 -f 64 -w 3 -r -b 1 -c 20

# 0.5% normal black text on white background
docker run -v $valid_dir:$app_output_dir -v $bg_dir:$app_bg_dir -t $image trdg \
-t 2 -f 64 -w 3 -r -b 1 -fi -c 20

# White, gaussian noise, Quasicrystal background
for i in {0..2}; do
    # 0.5% normal text
    docker run -v $valid_dir:$app_output_dir \
    -v $bg_dir:$app_bg_dir -t $image trdg \
    -t 2 -f 64 -w 3 -r -b $i -tc '#000000,#FFFFFF' -c 20
    # 0.5% random skewness to 5 degree
    docker run -v $valid_dir:$app_output_dir \
    -v $bg_dir:$app_bg_dir -t $image trdg \
    -t 2 -f 64 -w 3 -r -b $i -tc '#000000,#FFFFFF' -k 5 -rk -c 20
    # blur
    docker run -v $valid_dir:$app_output_dir \
    -v $bg_dir:$app_bg_dir -t $image trdg \
    -t 2 -f 64 -w 3 -r -b $i -tc '#000000,#FFFFFF' -k 5 -rk -bl 2 -rbl -c 30

    # distortion
    for j in {1..3}; do
        # distorsion orientation
        for z in {0..2}; do
            docker run -v $valid_dir:$app_output_dir \
            -v $bg_dir:$app_bg_dir -t $image trdg \
            -t 2 -f 64 -w 3 -r -b $i -tc '#000000,#FFFFFF' -k 5 -rk -bl 2 -rbl -d $j -do $z -c 20
        done
    done
done

# image background
# random font
docker run -v $valid_dir:$app_output_dir -v $bg_dir:$app_bg_dir -t $image trdg \
-t 2 -f 64 -w 3 -r -b 3 -id bg-images/ -tc '#000000,#FFFFFF' -c 50
# tight crop
docker run -v $valid_dir:$app_output_dir -v $bg_dir:$app_bg_dir -t $image trdg \
-t 2 -f 64 -w 3 -r -b 3 -id bg-images/ -tc '#000000,#FFFFFF' -fi -c 50

# 5 degree skew
docker run -v $valid_dir:$app_output_dir -v $bg_dir:$app_bg_dir -t $image trdg \
-t 2 -f 64 -w 3 -r -b 3 -id bg-images/ -tc '#000000,#FFFFFF' -k 5 -rk -c 100

# blur
docker run -v $valid_dir:$app_output_dir -v $bg_dir:$app_bg_dir -t $image trdg \
-t 2 -f 64 -w 3 -r -b 3 -id bg-images/ -tc '#000000,#FFFFFF' -k 5 -rk -bl 1 -rbl -c 100


for j in {1..3}; do
    # distorsion orientation
    for z in {0..2}; do
        docker run -v $valid_dir:$app_output_dir \
        -v $bg_dir:$app_bg_dir -t $image trdg \
        -t 2 -f 64 -w 3 -r -b 3 -id bg-images/ -tc '#000000,#FFFFFF' \
        -k 5 -rk -bl 1 -rbl -d $j -do $z -c 150
    done
done

docker run -v $valid_dir:$app_output_dir \
-v $bg_dir:$app_bg_dir -t $image trdg \
-t 2 -f 64 -w 3 -r -b 3 -id bg-images/ -tc '#000000,#FFFFFF' \
-k 5 -rk -bl 2 -rbl -d 3 -do 2 -c 50
