# Upload a custom dataset to the hub

This page aims to show a way to upload a custom dataset on the Hub.

Since the target data format is json lines, let's say we have the following files: `train-01.jsonl`, `train-02.jsonl`, `train-03.jsonl`, `train-04.jsonl`, `dev-01.jsonl` and `dev-02.jsonl` in the folder at this path `path/to/custom/dataset/files`. 

1. Create the dataset repo on the Hub

```bash
huggingface-cli repo create <your-dataset-id>
```

or if you want to create the dataset repo under our WG organization:
```bash
huggingface-cli repo create <your-dataset-id> --organization bs-modeling-metadata
```
Note: don't forget to replace `<your-dataset-id>` by the actual name of your custom dataset

2. Clone the dataset repo locally

*Please make sure you have both git and git-lfs installed on your system. See [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [Install Git LFS](https://git-lfs.github.com/)*

Start by going to where you want to clone the repository:
```bash
cd path/to/clone/hub/repo
```

Than run git lfs install to initialize git-lfs:

```bash
git lfs install
```
Now, you can clone the dataset with the following command if the previously dataset repo is on your account

```bash
git clone https://huggingface.co/<your-username>/<your-dataset-id>
```
otherwise if the repo is under the WG organization use:
```bash
git clone https://huggingface.co/bs-modeling-metadata/<your-dataset-id>
```

3. Move the custom files in cloned the dataset repo

```bash
mv path/to/custom/dataset/files/train-* path/to/clone/hub/repo/<your-dataset-id>
mv path/to/custom/dataset/files/dev-* path/to/clone/hub/repo/<your-dataset-id>
```

4. Compress the files (if they are not already compressed)

Just run the command to compress all your `jsonl` files:
```bash
cd path/to/clone/hub/repo/<your-dataset-id>
gzip *.jsonl
```

5. Add and commit the files

```bash
git add train*
git add dev*
git commit -m "add dataset"
```

6. Push the changes

And now you're ready to share your dataset on the hub  🚀

```bash
git push
```