Typical steps for Ubuntu 20.04

```bash
sudo apt update
sudo apt upgrade -y
```

```bash
sudo apt install git python-is-python3 python3-pip -y
```
```bash
pip install -qqU pip
. .profile
```

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
```
```bash
sudo apt install git-lfs
```
```bash
git config --global filter.lfs.smudge "git-lfs smudge --skip -- %f"
git config --global filter.lfs.process "git-lfs filter-process --skip"
```
```bash
git config --global credential.helper store
git config --global user.email "foo@bar"
git config --global user.name "Foo"
```

```bash
git clone -b code_branch_name https://github.com/bigscience-workshop/metadata.git
cd bigscience-metadata/
pip install -qqe .[preprocessing]
```
```bash
huggingface-cli login
```

```bash
python experiments/gcp/dataset/c4/tag_metadata.py task_id=0 out_dir=../../ metadata_to_include=["clean_website_description","entity_paragraph","title"] dataset_name="bs-modeling-metadata/c4-en-html-with-metadata" dataset_revision="data_branch_name" file_pq_serial_range=[0,0,19]
```

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone -b data_branch_name https://huggingface.co/datasets/bs-modeling-metadata/c4-en-html-with-metadata
cd c4-en-html-with-metadata/
git lfs install --skip-smudge
```

```bash
git checkout -b new_data_branch_name
git push -u origin HEAD
```

```bash
cp experiments/gcp/dataset/c4/upload.sh .
chmod u+x upload.sh
./upload.sh "00-0[01]*"
```
