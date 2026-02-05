#install git-lfs , pre-req for geneformer clone
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
apt-get install git-lfs
git lfs install

#install geneformer (with retry for transient errors)
cd /
for i in 1 2 3; do
  git clone https://huggingface.co/ctheodoris/Geneformer && break
  echo "Clone attempt $i failed, retrying in 15s..."
  sleep 15
done

if [ ! -d "Geneformer" ]; then
  echo "ERROR: Failed to clone Geneformer after 3 attempts"
  exit 1
fi

cd Geneformer
git checkout b07f4b1e8893a0923a8fde223fe3b5a60b976d99
pip install .

#Download training data and converting to streaming dataset
#commenting since we already have it in s3
#sh ./download_dataset.sh 
#python  create_mds.py
