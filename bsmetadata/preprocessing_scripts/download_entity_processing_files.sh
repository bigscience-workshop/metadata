# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


### This script downloads all the files required for the entity linking process in metadata preprocessing.
### The Entity Linker used here is REL: Radboud Entity Linker [(Github URL:https://github.com/informagi/REL), (arxiv URL: https://arxiv.org/abs/2006.01969)]

mkdir entity_linking_files
cd entity_linking_files

wget http://gem.cs.ru.nl/generic.tar.gz
wget http://gem.cs.ru.nl/wiki_2019.tar.gz
wget http://gem.cs.ru.nl/ed-wiki-2019.tar.gz

tar -xzf generic.tar.gz
tar -xzf wiki_2019.tar.gz
tar -xzf ed-wiki-2019.tar.gz

rm generic.tar.gz
rm wiki_2019.tar.gz
rm ed-wiki-2019.tar.gz

cd ..
