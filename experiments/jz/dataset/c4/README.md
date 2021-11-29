1. Download `wiki_en_dump.db` 

```bash
cd $SCRATCH/
mkdir modeling-metadata-artefacts
cd modeling-metadata-artefacts
gsutil cp gs://bigscience/modeling-metadata/preprocessing/wiki_en_dump.db /gpfsssd/scratch/rech/six/uue59kq/modeling-metadata-artefacts

mkdir entity_preprocessing
cd entity_preprocessing/

wget http://gem.cs.ru.nl/generic.tar.gz
wget http://gem.cs.ru.nl/wiki_2019.tar.gz
wget http://gem.cs.ru.nl/ed-wiki-2019.tar.gz

tar -xzf generic.tar.gz
tar -xzf wiki_2019.tar.gz
tar -xzf ed-wiki-2019.tar.gz

rm generic.tar.gz
rm wiki_2019.tar.gz
rm ed-wiki-2019.tar.gz
```

