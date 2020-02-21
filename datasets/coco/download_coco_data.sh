wget "http://images.cocodataset.org/zips/train2017.zip" | tr -d '\r'
wget "http://images.cocodataset.org/zips/val2017.zip" | tr -d '\r'
wget "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" | tr -d '\r'
unzip "train2017.zip" | tr -d '\r'
unzip "val2017.zip" | tr -d '\r'
unzip "annotations_trainval2017.zip" | tr -d '\r'
rm "train2017.zip" | tr -d '\r'
rm "val2017.zip" | tr -d '\r'
rm "annotations_trainval2017.zip" | tr -d '\r'
mkdir images
mv train2017 images/
mv val2017 images/

